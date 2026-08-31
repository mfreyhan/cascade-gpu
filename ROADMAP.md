# cascade — GPU-Accelerated Turbomachinery Cascade Solver

**Goal:** A compressible RANS solver for turbomachinery cascades written in C++/CUDA. Two objectives: (1) a concrete, validated portfolio project, (2) hands-on experience writing GPU solvers.

**Phase 1 (1 month, ~160 h):** 2D linear cascade, compressible RANS (Spalart–Allmaras), structured C-grid, in-house mesh generator, CUDA implementation, validation against published experimental data.

**Phase 2 (afterwards):** 3D, unstructured, implicit, multi-GPU, unsteady. The Phase 1 architecture is designed so these can be added *without rewriting the numerics*.

---

## 1. Fixed technical decisions

| Topic | Decision | Rationale |
|---|---|---|
| Equations | Compressible RANS, density-based | Turbomachinery validation data is compressible; an explicit density-based scheme requires no global solve, making it fully GPU-friendly |
| Discretization | Cell-centered finite volume, 2nd-order MUSCL | Standard and verifiable |
| Flux scheme | Roe with entropy fix + van Albada limiter (JST as an alternative) | Shock capturing in transonic cascades |
| Time integration | 5-stage explicit Runge–Kutta, local time stepping, implicit residual smoothing | Steady state; fully explicit |
| Turbulence | Spalart–Allmaras (negative variant); k-ω SST as a Phase 1 stretch goal | One equation, robust |
| Precision | `template<typename Real>`, **FP32 by default** | Rented consumer GPUs run FP64 at 1/64 rate. A deliberate decision, to be measured and reported |
| Mesh | In-house C-grid generator: TFI (algebraic) + elliptic (Poisson) smoothing | Full control over cascade geometry; a self-contained module |

---

## 2. Extensibility architecture (the 5 decisions that enable Phase 2)

Roughly 10% extra work in Phase 1; avoids a rewrite in Phase 2.

### K1 — Dimension as a compile-time parameter

`constexpr int NDIM` drives `Vec<Real,NDIM>` and `NVAR = NDIM+2`. The physical core (Roe flux, viscous flux, SA source terms) is written dimension-generically and instantiated for 2D and later 3D from the same code. Going to 3D means a new instantiation plus 3D mesh/BCs — **no new numerics**.

### K2 — Numerical kernels separated from loop drivers

Every scheme is written as a `__host__ __device__` **free function**:

```cpp
flux = roeFlux(stateL, stateR, faceNormalArea);
```

Two distinct drivers call them:

- **Structured driver:** direct i-j index arithmetic (fast, the Phase 1 default)
- **Face-list driver:** explicit face list + indirect indexing (the bridge to unstructured meshes)

A structured mesh can feed both, so the same case is run through both drivers and the **performance difference is measured and reported in the README**. In Phase 2, unstructured support reduces to a new mesh reader plus the existing face-list driver.

### K3 — Multi-block connectivity and halo exchange (even with a single block)

Even though Phase 1 uses one block, a block-connectivity and halo (ghost) exchange layer is written, because:

- The cascade C-grid's **wake cut** and its **pitchwise periodicity** are already connectivity problems
- In Phase 2, **multi-GPU** is the same halo exchange running over P2P/NCCL
- 3D multi-block meshes need it as well

### K4 — Turbulence model behind a policy interface

A `TurbulenceModel` interface: `nEquations`, `computeSource()`, `eddyViscosity()`, `wallBC()`. SA is its first implementation; SST and a transition model can be added later without moving existing code.

### K5 — Boundary conditions as face groups plus a type enum

Boundary conditions are not embedded in the mesh; they are kept as a list of `BoundaryPatch{type, faceRange, params}` and dispatched to kernels through a table. Hub/shroud/tip patches in 3D, non-reflecting BCs, and mixing planes become a new enum value plus a new kernel, leaving existing code untouched.

### Backend layer

Instead of a heavy dependency such as Kokkos, a thin `parallel_for` / `reduce` abstraction over Serial, OpenMP and CUDA. Every kernel compiles under all three, so development and testing are possible on a machine without a GPU.

---

## 3. Directory layout

```
cascade/
├── CMakeLists.txt            # CASCADE_ENABLE_CUDA, CASCADE_REAL=float|double
├── src/
│   ├── core/                 # Vec, State, Real, parallel_for backend
│   ├── mesh/                 # Block, connectivity, halo exchange, Plot3D I/O
│   ├── meshgen/              # C-grid generator: TFI + elliptic smoothing
│   ├── numerics/             # roeFlux, muscl, limiter, viscousFlux, gradient
│   ├── bc/                   # inlet, outlet, wall, periodic, wake-cut
│   ├── turb/                 # SA (and later SST)
│   ├── solver/               # RK, local dt, residual smoothing, convergence
│   ├── io/                   # VTK writer, restart, config (TOML/JSON)
│   └── drivers/              # structured & face-list residual drivers
├── tests/                    # unit + validation regression tests
├── cases/                    # sod, vortex, flatplate, naca0012, vki-ls59
├── tools/                    # Python: post-processing, plotting, comparison
├── docs/                     # theory.md, validation.md, performance.md
└── .github/workflows/        # CPU build + test CI
```

---

## 4. Phase 1 — four-week plan (40 h/week)

### Week 1 — Infrastructure + mesh generator + CPU Euler solver

- [ ] **D1-2:** CMake, `Real` template, `Vec/State`, SoA fields, `parallel_for` (Serial/OpenMP), Catch2, CI, config reader
- [ ] **D2-3:** Mesh data structures (blocks, ghost cells, metrics: face normals/areas, cell volumes), Plot3D + VTK I/O, connectivity & halo exchange
- [ ] **D3-5:** **C-grid generator**: read blade coordinates → initial grid via TFI → elliptic (Poisson) smoothing, wall orthogonality and first-cell-height control (y+ target), parametric pitch/stagger/wake-cut. Grid quality metrics (skewness, aspect ratio, negative-volume check)
- [ ] **D5-7:** Euler solver: Roe + MUSCL, RK5, local time stepping; BCs: slip wall, subsonic inlet (p0/T0/flow angle via Riemann invariants), static-pressure outlet, pitchwise periodic, wake cut
- [ ] **Verification:** Sod tube → 2D isentropic vortex (**order-of-accuracy plot**) → oblique shock → NACA0012 transonic Cp

✅ **Deliverable:** Euler cascade solution on a self-generated grid, with isentropic Mach distribution

### Week 2 — CUDA port + performance engineering

- [ ] **D8-9:** CUDA backend, device memory management, structured residual kernels (gradients, limiter, i/j fluxes, RK update), `cub::DeviceReduce` for time step and residual norms
- [ ] **D9-10:** GPU↔CPU regression test (agreement within tolerance), FP32 vs FP64 comparison
- [ ] **D10-12:** **Optimization pass** — measure, change, measure again:
  - AoS vs SoA
  - Shared-memory tiling vs relying on L2
  - `__launch_bounds__` / register pressure / occupancy
  - **CUDA Graphs** (RK5 means ~25 kernel launches per iteration — eliminate launch overhead)
  - Pinned vs unified memory
- [ ] **D12-13:** Nsight Systems timeline + Nsight Compute **roofline**; implement the face-list driver and benchmark it against the structured one
- [ ] **D13-14:** `docs/performance.md`: speedup table (Serial / OpenMP / GPU), roofline plot, gain from each optimization

✅ **Deliverable:** A measured, profiled, documented GPU Euler solver

### Week 3 — Viscous terms + turbulence

- [ ] **D15-16:** Viscous fluxes (Green–Gauss gradients with face-gradient correction), Sutherland viscosity, adiabatic/isothermal no-slip walls
- [ ] **D16:** **Wall distance** (Phase 1: brute-force GPU kernel is sufficient; KD-tree/Eikonal in Phase 2)
- [ ] **D17-19:** **SA-neg**: convection (first-order upwind), diffusion, source terms, wall BC, separate relaxation; eddy-viscosity limiters
- [ ] **D19-21:** Validation: laminar flat plate → **Blasius cf + velocity profile**; turbulent flat plate → **NASA TMR reference** (cf, log-law u+), grid convergence

✅ **Deliverable:** A validated turbulent RANS solver

### Week 4 — Cascade validation + polish

- [ ] **D22-24:** **VKI LS-59** transonic turbine cascade: mesh, run, isentropic Mach distribution comparison, **total pressure loss coefficient ω**, exit flow angle, mesh refinement study. (Optional second case: Kiock cascade or a NACA-65 compressor cascade)
- [ ] **D25-26:** Convergence acceleration — priority: **FAS multigrid** (the largest gain for an explicit solver). k-ω SST if time remains
- [ ] **D27-28:** `docs/validation.md` (the primary portfolio document), README (theory summary + figures + performance + usage), example cases, clean commit history, release tag

✅ **Deliverable:** A publishable repository

---

## 5. Validation matrix

| Case | What it tests | Reference |
|---|---|---|
| Sod shock tube (1D) | Riemann solver, shock/contact discontinuity | Analytical |
| 2D isentropic vortex | **Order of accuracy (2nd)**, low dissipation | Analytical |
| Oblique shock / wedge | Shock angle, supersonic BCs | Analytical |
| Subsonic channel bump | Inlet/outlet BCs, spurious entropy generation | Reference solution |
| NACA0012 M=0.8, α=1.25° | Transonic Cp, shock position | AGARD |
| Laminar flat plate | Blasius cf, velocity profile | Analytical |
| Turbulent flat plate | SA verification, cf, log-law | NASA Turbulence Modeling Resource |
| **VKI LS-59 cascade** | Isentropic Mach distribution, loss, exit angle | Experimental |
| GPU vs CPU | Backend equivalence | Internal regression |

---

## 6. GPU learning checklist (Week 2 output — each item measured and written up)

- [ ] SoA vs AoS memory access measurement
- [ ] Coalescing + L2 hit rate analysis
- [ ] Shared memory tiling (stencil)
- [ ] Occupancy / register pressure trade-off (`__launch_bounds__`)
- [ ] `cub` device-wide reduction
- [ ] CUDA Graphs to eliminate launch overhead
- [ ] Streams + asynchronous copies
- [ ] FP32 / FP64 / mixed precision comparison
- [ ] Nsight Compute roofline interpretation (demonstrate and prove the solver is memory-bound)
- [ ] (Phase 2) Multi-GPU halo exchange, P2P / NCCL

---

## 7. Development + Vast.ai workflow

**Local (no GPU):**

- With the CUDA Toolkit installed, `nvcc` **compiles fine without a GPU**. Only execution requires one.
- Day-to-day development on the Serial/OpenMP backend; move to the GPU once tests are green.

**Remote (Vast.ai):**

- Image: `nvidia/cuda:12.x-devel-ubuntu22.04`. RTX 3090 ≈ $0.20/h, RTX 4090 ≈ $0.35/h.
- Workflow: `git push` → on the instance `git pull && cmake --build` → run → download the Nsight report and open it locally.
- Heaviest in Week 2 (~15-20 h), 1-2 h/day otherwise. **Monthly total ≈ $30-50.**
- Never leave an instance running without producing output; solution files stay out of the repo — only summary CSVs and figures are committed.

---

## 8. Phase 2 backlog (after month 1, in priority order)

1. **3D** — spanwise direction, 3D mesh generator, hub/shroud/tip BCs (thanks to K1, no numerics are rewritten)
2. **Implicit** — LU-SGS / point-implicit, much larger CFL
3. **Multi-GPU** — moving the K3 halo exchange onto P2P/NCCL
4. **Unstructured** — CGNS/SU2 reader plus the existing face-list driver (K2)
5. **Unsteady** — dual-time stepping, rotor–stator interaction, mixing plane
6. **k-ω SST + γ-Reθ transition** — required for LPT cascades such as T106A
7. Non-reflecting (Giles) BCs, quasi-3D AVDR / streamtube thickness

---

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Periodic BC + ghost cell indexing errors (the most common source of bugs) | Unit-test halo exchange on its own (transport an analytical field) |
| SA diverging on a poor grid | y+ < 1 enforced; report grid quality metrics from the mesh generator |
| Elliptic smoother producing negative volumes | Jacobian check every iteration, under-relaxation |
| FP32 GPU/CPU differences breaking tests | Define tolerances relatively; run critical tests in FP64 |
| Week 3 overrunning | SST and multigrid are stretch goals; the committed deliverable is SA + one cascade validation |

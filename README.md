# cascade

A GPU-accelerated compressible RANS solver for 2D linear turbomachinery cascades, written in C++17/CUDA, with an in-house structured C-grid mesh generator.

> **Status: under active development (Week 1 of 4).** The core layer, build system and test suite are in place. The Euler solver, mesh generator, CUDA backend and turbulence model are being added in that order — see [ROADMAP.md](ROADMAP.md). Nothing in this repository is validated yet; validation results will appear in `docs/validation.md` as each case passes.

## What it is

A density-based, cell-centred finite volume solver for the compressible RANS equations, targeting blade-to-blade cascade flows:

- 2nd-order MUSCL reconstruction with a Roe approximate Riemann solver
- explicit 5-stage Runge–Kutta with local time stepping
- Spalart–Allmaras turbulence model
- structured C-grid, generated in-repo from blade coordinates (TFI + elliptic smoothing)
- Serial / OpenMP / CUDA execution backends behind one interface, so the same kernel source runs on all three

The solver is templated on the floating-point type and defaults to **FP32**, because the consumer GPUs it targets run FP64 at 1/64 of their FP32 rate. Verification cases that measure order of accuracy are built in FP64.

## Design

The code is 2D today but is structured so that 3D, unstructured meshes and multi-GPU do not require rewriting the numerics. The five decisions that carry this — compile-time dimension, numerics separated from loop drivers, a halo-exchange layer present even with one block, turbulence behind a policy interface, and boundary conditions as data — are described in [ROADMAP.md](ROADMAP.md) §2.

## Build

```bash
# CPU only (default; works on a machine with no GPU)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

# With CUDA (nvcc compiles without a GPU present; only running needs one)
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release \
      -DCASCADE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
```

| Option | Default | Meaning |
|---|---|---|
| `CASCADE_REAL` | `float` | solver floating point type (`float` or `double`) |
| `CASCADE_NDIM` | `2` | spatial dimension |
| `CASCADE_ENABLE_OPENMP` | `ON` | OpenMP execution backend |
| `CASCADE_ENABLE_CUDA` | `OFF` | CUDA execution backend |
| `CASCADE_WERROR` | `OFF` | treat warnings as errors (used in CI) |

## Run

```bash
./build/cascade cases/demo/config.toml
```

## License

MIT

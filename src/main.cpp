#include "core/Backend.hpp"
#include "core/Config.hpp"
#include "core/Field.hpp"
#include "core/State.hpp"
#include "core/Types.hpp"

#include <iostream>
#include <string>
#include <typeinfo>

namespace {

const char* realTypeName() { return sizeof(cascade::Real) == 4 ? "float (FP32)" : "double (FP64)"; }

void printBanner() {
  using namespace cascade;
  std::cout << "cascade " << CASCADE_VERSION_STRING << "\n"
            << "  dimension : " << NDIM << "D  (" << NVAR << " mean-flow variables)\n"
            << "  precision : " << realTypeName() << "\n"
            << "  backend   : " << defaultBackendName() << " (" << backendThreadCount()
            << " threads)\n";
#if defined(CASCADE_HAVE_CUDA)
  std::cout << "  cuda      : compiled in\n";
#else
  std::cout << "  cuda      : not compiled in\n";
#endif
}

}  // namespace

int main(int argc, char** argv) {
  using namespace cascade;

  printBanner();

  if (argc < 2) {
    std::cout << "\nusage: cascade <case.toml>\n";
    return 0;
  }

  try {
    const Config cfg = Config::fromFile(argv[1]);
    std::cout << "\nconfiguration (" << argv[1] << "):\n" << cfg.dump();

    const Gas gas{cfg.getReal("gas.gamma", Real(1.4))};

    // Inlet stagnation state in the solver's non-dimensionalisation:
    // rho01 = 1, T01 = 1, p01 = 1/gamma.  Reported here as a standing check
    // that the scaling in State.hpp and the gas model agree.
    const Real p01 = Real(1) / gas.gamma;
    Prim q{};
    q.rho = Real(1);
    q.u = Vec<Real, NDIM>::zero();
    q.p = p01;

    const Cons w = toConserved(q, gas);
    const Prim back = toPrimitive(w, gas);

    std::cout << "\ninlet stagnation state (non-dimensional):\n"
              << "  rho01 = " << back.rho << "   p01 = " << back.p
              << "   T01 = " << temperature(back.rho, back.p, gas)
              << "   a01 = " << soundSpeed(back.rho, back.p, gas) << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}

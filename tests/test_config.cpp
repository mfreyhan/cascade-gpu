#include "core/Config.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace cascade;
using Catch::Approx;

namespace {
const char* kSample = R"(
# a cascade case file
name = "vki-ls59"

[gas]
gamma = 1.4

[inlet]
total_pressure = 1.0       ; inline comment
total_temperature = 1.0
flow_angle_deg = 30.0
supersonic = false

[solver]
cfl = 2.5
max_iterations = 20000
scheme = "roe"

[mesh]
spacing = [1.0e-5, 0.002, 0.01]
)";
}  // namespace

TEST_CASE("sections become dotted key prefixes", "[config]") {
  const Config cfg = Config::fromString(kSample);

  REQUIRE(cfg.has("gas.gamma"));
  REQUIRE(cfg.has("solver.cfl"));
  REQUIRE(cfg.has("name"));  // top-level key keeps no prefix
  REQUIRE_FALSE(cfg.has("cfl"));
}

TEST_CASE("typed lookups", "[config]") {
  const Config cfg = Config::fromString(kSample);

  REQUIRE(cfg.getReal("gas.gamma", Real(0)) == Approx(1.4));
  REQUIRE(cfg.getInt("solver.max_iterations", 0) == 20000);
  REQUIRE(cfg.getString("solver.scheme", "") == "roe");
  REQUIRE(cfg.getBool("inlet.supersonic", true) == false);
  REQUIRE(cfg.getReal("inlet.flow_angle_deg", Real(0)) == Approx(30.0));
}

TEST_CASE("defaults are used only for absent keys", "[config]") {
  const Config cfg = Config::fromString(kSample);
  REQUIRE(cfg.getReal("solver.cfl", Real(1)) == Approx(2.5));
  REQUIRE(cfg.getReal("solver.does_not_exist", Real(7)) == Approx(7.0));
}

TEST_CASE("comments and quotes are stripped correctly", "[config]") {
  const Config cfg = Config::fromString(kSample);
  // The ';' comment after total_pressure must not end up in the value.
  REQUIRE(cfg.getReal("inlet.total_pressure", Real(0)) == Approx(1.0));
  REQUIRE(cfg.getString("name", "") == "vki-ls59");
}

TEST_CASE("real arrays parse", "[config]") {
  const Config cfg = Config::fromString(kSample);
  const auto spacing = cfg.getRealArray("mesh.spacing");
  REQUIRE(spacing.size() == 3);
  REQUIRE(spacing[0] == Approx(1.0e-5));
  REQUIRE(spacing[2] == Approx(0.01));
}

TEST_CASE("required keys throw instead of silently defaulting", "[config]") {
  // Inlet conditions and mesh paths must never fall back to a plausible
  // default; a wrong-but-running case is worse than a failed launch.
  const Config cfg = Config::fromString(kSample);
  REQUIRE(cfg.requireReal("gas.gamma") == Approx(1.4));
  REQUIRE_THROWS_AS(cfg.requireReal("inlet.mach"), std::runtime_error);
  REQUIRE_THROWS_AS(cfg.requireString("mesh.file"), std::runtime_error);
}

TEST_CASE("malformed input is reported with a line number", "[config]") {
  REQUIRE_THROWS_AS(Config::fromString("[unterminated\n"), std::runtime_error);
  REQUIRE_THROWS_AS(Config::fromString("key_without_value\n"), std::runtime_error);
  REQUIRE_THROWS_AS(Config::fromString("= 5\n"), std::runtime_error);
  REQUIRE_THROWS_AS(Config::fromFile("no/such/file.toml"), std::runtime_error);
}

TEST_CASE("a bad boolean is an error, not a false", "[config]") {
  const Config cfg = Config::fromString("[a]\nflag = maybe\n");
  REQUIRE_THROWS_AS(cfg.getBool("a.flag", false), std::runtime_error);
}

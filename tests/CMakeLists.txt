include(FetchContent)

# Catch2 v3. Prefer an installed copy (offline / CI cache), otherwise fetch.
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.2
  GIT_SHALLOW    TRUE
  FIND_PACKAGE_ARGS 3
)
FetchContent_MakeAvailable(Catch2)

add_executable(cascade_tests
  test_vec.cpp
  test_state.cpp
  test_field.cpp
  test_backend.cpp
  test_config.cpp
  test_mesh.cpp
  test_halo.cpp
)

target_link_libraries(cascade_tests PRIVATE cascade_core Catch2::Catch2WithMain)

list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
include(Catch)
catch_discover_tests(cascade_tests)

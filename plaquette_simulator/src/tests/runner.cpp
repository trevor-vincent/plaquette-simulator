#define CATCH_CONFIG_RUNNER
// #include "Test_CliffordStateKokkos.hpp"
#include "Test_CliffordState.hpp"
#include <Kokkos_Core.hpp>
#include <catch2/catch.hpp>

using namespace Plaquette;

int main(int argc, char *argv[]) {
    int result;
    {
        CliffordState<int> kokkos_state_int(0, 0);
        result = Catch::Session().run(argc, argv);
    }
    return result;
}

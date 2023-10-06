#pragma once

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "BatchCliffordStateKokkos.hpp"

using namespace Plaquette;
namespace {} // namespace


#define PRINT_VECTOR_3D(v) \
    std::cout << #v << ":\n"; \
    for (size_t i = 0; i < v.size(); ++i) { \
        std::cout << "Batch " << i << ":\n"; \
        for (size_t j = 0; j < v[i].size(); ++j) { \
            for (size_t k = 0; k < v[i][j].size(); ++k) { \
                std::cout << v[i][j][k] << " "; \
            } \
            std::cout << "\n"; \
        } \
        std::cout << "\n"; \
    }

#define PRINT_VECTOR_2D(v) \
    std::cout << #v << ":\n"; \
    for (size_t i = 0; i < v.size(); ++i) { \
        std::cout << "Batch " << i << ":\n"; \
        for (size_t j = 0; j < v[i].size(); ++j) { \
            std::cout << v[i][j] << " "; \
        } \
        std::cout << "\n\n"; \
    }

TEMPLATE_TEST_CASE("BatchCliffordStateKokkos::Initialization",
                   "[batch_clifford] [initialization]", int) {

  {
    const std::size_t num_qubits = 3;
    const std::size_t batch_size = 1;
    BatchCliffordStateKokkos<TestType> kokkos_state_1(num_qubits, batch_size);
  }
}

TEMPLATE_TEST_CASE("CliffordStateKokkos::CheckInitialization",
                   "[CliffordStateKokkos]", int) {

  {
    const std::size_t num_qubits = 3;
    const std::size_t batch_size = 1;
    const std::size_t tableau_width = 2 * num_qubits + 1;

    BatchCliffordStateKokkos<TestType> kokkos_state_1{num_qubits, batch_size};

    std::vector<std::vector<std::vector<TestType>>> x_host(
        batch_size, std::vector<std::vector<TestType>>(
                        tableau_width, std::vector<TestType>(num_qubits)));

    std::vector<std::vector<std::vector<TestType>>> z_host(
        batch_size, std::vector<std::vector<TestType>>(
                        tableau_width, std::vector<TestType>(num_qubits)));

    std::vector<std::vector<TestType>> r_host(
        batch_size, std::vector<TestType>(tableau_width));

    kokkos_state_1.DeviceToHost(&x_host[0][0][0], &z_host[0][0][0],
                                &r_host[0][0]);



    PRINT_VECTOR_3D(x_host);
    PRINT_VECTOR_3D(z_host);
    PRINT_VECTOR_2D(r_host);
    
    //  std::vector<std::vector<int>> tableau_x_check = {{
    //      {1, 0, 0},
    //      {0, 1, 0},
    //      {0, 0, 1},
    //      {0, 0, 0},
    //      {0, 0, 0},
    //      {0, 0, 0},
    // 	{0, 0, 0}
    //  }};

    //  std::vector<std::vector<int>> tableau_z_check = {{
    // 	{0, 0, 0},
    // 	{0, 0, 0},
    // 	{0, 0, 0},
    // 	{1, 0, 0},
    // 	{0, 1, 0},
    // 	{0, 0, 1},
    // 	{0, 0, 0}
    //    }};

    // for (int i = 0; i < 2*num_qubits*num_qubits; ++i) {
    //   REQUIRE(tableau_x_host[i] ==
    //   tableau_x_check[i/num_qubits][i%num_qubits]); REQUIRE(tableau_z_host[i]
    //   == tableau_z_check[i/num_qubits][i%num_qubits]);
    // }

    // for (int i = 0; i < 2*num_qubits; ++i) {
    //   REQUIRE(tableau_sign_host[i] == 0);
    // }

    // PRINT_MATRIX(tableau_x_host, 2*num_qubits, num_qubits);
    // PRINT_MATRIX(tableau_z_host, 2*num_qubits, num_qubits);
    // PRINT_MATRIX(tableau_sign_host, 2*num_qubits, 1);
    // REQUIRE(1==0);
  }
}

TEMPLATE_TEST_CASE("BatchCliffordStateKokkos::Hadamard",
                   "[batch_clifford] [hadamard]", int) {

  {
    const std::size_t num_qubits = 3;
    const std::size_t batch_size = 1;
    BatchCliffordStateKokkos<TestType> kokkos_state_1(num_qubits, batch_size);
    kokkos_state_1.ApplyHadamardGate(0);
  }
}

TEMPLATE_TEST_CASE("BatchCliffordStateKokkos::Phase",
                   "[batch_clifford] [hadamard]", int) {

  {
    const std::size_t num_qubits = 3;
    const std::size_t batch_size = 1;
    BatchCliffordStateKokkos<TestType> kokkos_state_1(num_qubits, batch_size);
    kokkos_state_1.ApplyPhaseGate(0);
  }
}

TEMPLATE_TEST_CASE("BatchCliffordStateKokkos::PauliX",
                   "[batch_clifford] [hadamard]", int) {

  {
    const std::size_t num_qubits = 3;
    const std::size_t batch_size = 1;
    BatchCliffordStateKokkos<TestType> kokkos_state_1(num_qubits, batch_size);
    kokkos_state_1.ApplyPauliXGate(0);
  }
}

TEMPLATE_TEST_CASE("BatchCliffordStateKokkos::PauliZ",
                   "[batch_clifford] [hadamard]", int) {

  {
    const std::size_t num_qubits = 3;
    const std::size_t batch_size = 1;
    BatchCliffordStateKokkos<TestType> kokkos_state_1(num_qubits, batch_size);
    kokkos_state_1.ApplyPauliZGate(0);
  }
}

TEMPLATE_TEST_CASE("BatchCliffordStateKokkos::CNOT", "[batch_clifford] [cnot]",
                   int) {

  {
    const std::size_t num_qubits = 3;
    const std::size_t batch_size = 1;
    BatchCliffordStateKokkos<TestType> kokkos_state_1(num_qubits, batch_size);
    kokkos_state_1.ApplyControlNotGate(0, 1);
  }
}

// TEMPLATE_TEST_CASE("CliffordStateKokkos::CopyConstructor",
//                    "[CliffordStateKokkos]", int) {

//   {
//     const std::size_t num_qubits = 3;
//     CliffordStateKokkos<TestType> kokkos_state_1{num_qubits};
//     CliffordStateKokkos<TestType> kokkos_state_2{kokkos_state_1};

//     CHECK(kokkos_state_1.GetNumQubits() == kokkos_state_2.GetNumQubits());

//     std::vector<Kokkos::complex<TestType>>
//     kokkos_state_1_host(kokkos_state_1.getLength());

//     std::vector<Kokkos::complex<TestType>>
//     kokkos_state_2_host(kokkos_state_2.getLength());

//     kokkos_state_1.DeviceToHost(kokkos_state_1_host.data(),
// 				kokkos_state_1.getLength());

//     kokkos_state_2.DeviceToHost(kokkos_state_2_host.data(),
// 				kokkos_state_2.getLength());

//     // for (size_t i = 0; i < kokkos_state_1_host.size(); i++) {
//     //   CHECK(kokkos_state_1_host[i] == kokkos_state_2_host[i]);
//     // }
//   }
// }

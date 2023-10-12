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
#include "Test_Helpers.hpp"

using namespace Plaquette;
namespace {} // namespace

#define PRINT_MATRIX_BATCH(variable, batch_size, tableau_width, num_qubits)    \
  do {                                                                         \
    std::cout << #variable << ":\n";                                           \
    for (size_t batch_id = 0; batch_id < batch_size; ++batch_id) {             \
      std::cout << "Batch " << batch_id << ":\n";                              \
      for (size_t i = 0; i < tableau_width; ++i) {                             \
        for (size_t j = 0; j < num_qubits; ++j) {                              \
          std::cout << variable[batch_id * tableau_width * num_qubits +        \
                                i * num_qubits + j]                            \
                    << " ";                                                    \
        }                                                                      \
        std::cout << "\n";                                                     \
      }                                                                        \
      std::cout << "\n";                                                       \
    }                                                                          \
  } while (0)

#define PRINT_VECTOR_BATCH(variable, batch_size, tableau_width)                \
  do {                                                                         \
    std::cout << #variable << ":\n";                                           \
    for (size_t batch_id = 0; batch_id < batch_size; ++batch_id) {             \
      std::cout << "Batch " << batch_id << ": ";                               \
      for (size_t i = 0; i < tableau_width; ++i) {                             \
        std::cout << variable[batch_id * tableau_width + i] << " ";            \
      }                                                                        \
      std::cout << "\n";                                                       \
    }                                                                          \
  } while (0)

#define PRINT_VECTOR_3D(v)                                                     \
  std::cout << #v << ":\n";                                                    \
  for (size_t i = 0; i < v.size(); ++i) {                                      \
    std::cout << "Batch " << i << ":\n";                                       \
    for (size_t j = 0; j < v[i].size(); ++j) {                                 \
      for (size_t k = 0; k < v[i][j].size(); ++k) {                            \
        std::cout << v[i][j][k] << " ";                                        \
      }                                                                        \
      std::cout << "\n";                                                       \
    }                                                                          \
    std::cout << "\n";                                                         \
  }

#define PRINT_VECTOR_2D(v)                                                     \
  std::cout << #v << ":\n";                                                    \
  for (size_t i = 0; i < v.size(); ++i) {                                      \
    std::cout << "Batch " << i << ":\n";                                       \
    for (size_t j = 0; j < v[i].size(); ++j) {                                 \
      std::cout << v[i][j] << " ";                                             \
    }                                                                          \
    std::cout << "\n\n";                                                       \
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
    const std::size_t batch_size = 2;
    const std::size_t tableau_width = 2 * num_qubits + 1;

    BatchCliffordStateKokkos<TestType> kokkos_state_1{num_qubits, batch_size};
    auto &&[x, z, r] = kokkos_state_1.DeviceToHost();

    std::vector<std::vector<int>> tableau_x_check = {
        {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0},
        {0, 0, 0}, {0, 0, 0}, {0, 0, 0}};

    std::vector<std::vector<int>> tableau_z_check = {
        {0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {1, 0, 0},
        {0, 1, 0}, {0, 0, 1}, {0, 0, 0}};

    for (size_t batch_id = 0; batch_id < batch_size; ++batch_id) {
      for (size_t i = 0; i < tableau_width; ++i) {
        if (i != tableau_width - 1) {
          REQUIRE(r[batch_id * tableau_width + i] == 0);
        } else {
          REQUIRE(r[batch_id * tableau_width + i] == 1);
        }
        for (size_t j = 0; j < num_qubits; ++j) {
          REQUIRE(
              x[batch_id * tableau_width * num_qubits + i * num_qubits + j] ==
              tableau_x_check[i][j]);
          REQUIRE(
              z[batch_id * tableau_width * num_qubits + i * num_qubits + j] ==
              tableau_z_check[i][j]);
        }
      }
    }
  }
}

TEMPLATE_TEST_CASE("CliffordStateKokkos::XZRConstructor",
                   "[CliffordStateKokkos]", int) {

  {
    const std::size_t num_qubits = 5;
    const std::size_t batch_size = 1;
    const std::size_t tableau_width = 2 * num_qubits + 1;

    std::vector<std::vector<int>> x_check = {
        {1, 0, 0, 0, 0}, {1, 1, 0, 0, 0}, {1, 1, 1, 0, 1}, {1, 1, 0, 1, 0},
        {0, 1, 0, 0, 1}, {0, 0, 0, 0, 1}, {0, 1, 0, 0, 0}, {0, 0, 0, 0, 0},
        {1, 1, 0, 1, 0}, {1, 1, 0, 0, 1}, {0, 0, 0, 0, 0}};

    std::vector<std::vector<int>> z_check = {
        {0, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {0, 1, 0, 1, 0}, {0, 0, 0, 0, 1},
        {0, 1, 1, 1, 0}, {1, 1, 0, 0, 1}, {0, 1, 0, 1, 0}, {0, 0, 0, 1, 0},
        {0, 0, 0, 1, 0}, {0, 1, 0, 1, 0}, {0, 0, 0, 0, 0}};

    std::vector<int> r_check = {0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1};

    std::vector<int> x_flat = {1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0,
                               1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                               0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                               0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0};

    std::vector<int> z_flat = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
                               0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0,
                               0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                               0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0};

    BatchCliffordStateKokkos<TestType> kokkos_state_1{x_flat, z_flat, r_check,
                                                      num_qubits, batch_size};

    auto &&[x, z, r] = kokkos_state_1.DeviceToHost();

    for (size_t batch_id = 0; batch_id < batch_size; ++batch_id) {
      for (size_t i = 0; i < tableau_width; ++i) {
        REQUIRE(r[batch_id * tableau_width + i] == r_check[i]);
        for (size_t j = 0; j < num_qubits; ++j) {
          REQUIRE(
              x[batch_id * tableau_width * num_qubits + i * num_qubits + j] ==
              x_check[i][j]);
          REQUIRE(
              z[batch_id * tableau_width * num_qubits + i * num_qubits + j] ==
              z_check[i][j]);
        }
      }
    }
  }
}

TEMPLATE_TEST_CASE("BatchCliffordStateKokkos::Measure0",
                   "[batch_clifford] [measure] [measure0]", int) {
  {
    const std::size_t num_qubits = 1;
    const std::size_t batch_size = 1;
    BatchCliffordStateKokkos<TestType> kokkos_state_1(num_qubits, batch_size);
    kokkos_state_1.ApplyHadamardGate(0);
    kokkos_state_1.ApplyPhaseGate(0);
    kokkos_state_1.ApplyPhaseGate(0);
    kokkos_state_1.ApplyHadamardGate(0);
    kokkos_state_1.MeasureQubit(0);
    auto result = kokkos_state_1.GetMeasurement(0, 0);
    REQUIRE(result.value().first == 1);  // Check the measured value.
    REQUIRE(result.value().second == 1); // Check that the measurement was
  }
}


TEMPLATE_TEST_CASE("BatchCliffordStateKokkos::Measure1",
                   "[batch_clifford] [measure] [measure1]", int) {
  {
    const std::size_t num_qubits = 1;
    const std::size_t batch_size = 1;
    BatchCliffordStateKokkos<TestType> kokkos_state_1(num_qubits, batch_size);
    kokkos_state_1.MeasureQubit(0);
    auto result = kokkos_state_1.GetMeasurement(0, 0);
    REQUIRE(result.value().first == 0);  // Check the measured value.
    REQUIRE(result.value().second == 1); // Check that the measurement was
  }
}


TEMPLATE_TEST_CASE("BatchCliffordStateKokkos::Measure2",
                   "[batch_clifford] [measure] [measure2]", int) {
  {
    const std::size_t num_qubits = 2;
    const std::size_t batch_size = 1;
    BatchCliffordStateKokkos<TestType> kokkos_state_1(num_qubits, batch_size);
    kokkos_state_1.MeasureQubit(0);
    kokkos_state_1.MeasureQubit(1);
    auto result_0 = kokkos_state_1.GetMeasurement(0, 0);
    auto result_1 = kokkos_state_1.GetMeasurement(1, 0);
    REQUIRE(result_0.value().first == 0);  // Check the measured value.
    REQUIRE(result_0.value().second == 1); // Check that the measurement was
    REQUIRE(result_1.value().first == 0);  // Check the measured value.
    REQUIRE(result_1.value().second == 1); // Check that the measurement was    
  }
}


TEMPLATE_TEST_CASE("BatchCliffordStateKokkos::Measure3",
                   "[batch_clifford] [measure] [measure3]", int) {
  {
    const std::size_t num_qubits = 2;
    const std::size_t batch_size = 1;
    BatchCliffordStateKokkos<TestType> kokkos_state_1(num_qubits, batch_size);

    kokkos_state_1.ApplyHadamardGate(0);
    kokkos_state_1.ApplyPhaseGate(0);
    kokkos_state_1.ApplyPhaseGate(0);
    kokkos_state_1.ApplyHadamardGate(0);    
    kokkos_state_1.MeasureQubit(0);
    kokkos_state_1.MeasureQubit(1);
    auto result_0 = kokkos_state_1.GetMeasurement(0, 0);
    auto result_1 = kokkos_state_1.GetMeasurement(1, 0);
    
    REQUIRE(result_0.value().first == 1);  // Check the measured value.
    REQUIRE(result_0.value().second == 1); // Check that the measurement was
    REQUIRE(result_1.value().first == 0);  // Check the measured value.
    REQUIRE(result_1.value().second == 1); // Check that the measurement was    
  }
}


TEMPLATE_TEST_CASE("BatchCliffordStateKokkos::EPR",
                   "[batch_clifford] [epr]", int) {

  {
    const std::size_t num_qubits = 2;
    const std::size_t batch_size = 1;
    BatchCliffordStateKokkos<TestType> kokkos_state_1(num_qubits, batch_size);
    // Apply quantum operations.
    kokkos_state_1.ApplyHadamardGate(0);
    kokkos_state_1.ApplyControlNotGate(0, 1);

    kokkos_state_1.MeasureQubit(0);
    kokkos_state_1.MeasureQubit(1);
    auto result_0 = kokkos_state_1.GetMeasurement(0, 0);
    auto result_1 = kokkos_state_1.GetMeasurement(1, 0);
    
    REQUIRE(result_0.value().second == 0); // Check that the measurement was
    REQUIRE(result_1.value().second == 1); // Check that the measurement was    
    REQUIRE(result_1.value().first == result_0.value().first);  // Check the measured value.

  }
}

TEMPLATE_TEST_CASE("BatchCliffordStateKokkos::PhaseKickback",
                   "[batch_clifford] [hadamard]", int) {

  {
    const std::size_t num_qubits = 2;
    const std::size_t batch_size = 1;
    BatchCliffordStateKokkos<TestType> kokkos_state_1(num_qubits, batch_size);
    kokkos_state_1.ApplyHadamardGate(1);
    kokkos_state_1.ApplyPhaseGate(1);
    kokkos_state_1.ApplyHadamardGate(0);
    kokkos_state_1.ApplyControlNotGate(0, 1);
    kokkos_state_1.MeasureQubit(1);
    auto result_0 = kokkos_state_1.GetMeasurement(0, 0);
    REQUIRE(result_0.value().second == 0); // Check that the measurement was not deterministic.

    if (result_0.value().first){
      kokkos_state_1.ApplyPhaseGate(0);
      kokkos_state_1.ApplyPhaseGate(0);
    }
    kokkos_state_1.ApplyPhaseGate(0);
    kokkos_state_1.ApplyHadamardGate(0);

    kokkos_state_1.MeasureQubit(0);
    auto result_1 = kokkos_state_1.GetMeasurement(1, 0);
    REQUIRE(result_1.value().second == 1); // Check that the measurement was
    REQUIRE(result_1.value().first == 1);  // Check the measured value.
    
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

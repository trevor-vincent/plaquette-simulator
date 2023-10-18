#pragma once

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include <catch2/catch.hpp>

#include "BatchCliffordStateKokkos.hpp"
#include "TableauState.hpp"
#include "Test_Helpers.hpp"

using namespace Plaquette;
namespace {} // namespace
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
    REQUIRE(result.value().first == 0);
    // Check the measured value. REQUIRE(result.value().second == 1); //
    // Check that the measurement was
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

TEMPLATE_TEST_CASE("BatchCliffordStateKokkos::EPR", "[batch_clifford] [epr]",
                   int) {

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
    REQUIRE(result_1.value().first ==
            result_0.value().first); // Check the measured value.
  }
}

TEST_CASE("Test PauliProductPhase", "[product_phase]") {

  std::vector<std::vector<int>> paulis = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  std::vector<std::vector<int>> expected = {
      {0, 0, 0, 0}, {0, 0, 1, -1}, {0, -1, 0, 1}, {0, 1, -1, 0}};

  REQUIRE(PauliProductPhase(1, 0, 1, 1) == 1);

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      REQUIRE(PauliProductPhase(paulis[i][0], paulis[i][1], paulis[j][0],
                                paulis[j][1]) == expected[i][j]);
    }
  }
}

TEST_CASE("Test Checksum", "[checksum]") {

  const std::size_t num_qubits = 5;
  const std::size_t batch_size = 1;
  const std::size_t tableau_width = 2 * num_qubits + 1;

  std::vector<int> r_flat = {0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1};

  std::vector<int> x_flat = {1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0,
                             1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                             0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                             0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0};

  std::vector<int> z_flat = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
                             0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0,
                             0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                             0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0};

  BatchCliffordStateKokkos<int> kokkos_state_1{x_flat, z_flat, r_flat,
                                               num_qubits, batch_size};

  auto checksum = kokkos_state_1.CheckSum();
  auto expected_checksum = std::accumulate(r_flat.begin(), r_flat.end(), 0) +
                           std::accumulate(x_flat.begin(), x_flat.end(), 0) +
                           std::accumulate(z_flat.begin(), z_flat.end(), 0);
  REQUIRE(checksum == expected_checksum);
}

TEMPLATE_TEST_CASE("CliffordStateKokkos::BatchRowProductSignFunctor",
                   "[CliffordStateKokkos]", int) {

  {
    const std::size_t num_qubits = 5;
    const std::size_t batch_size = 1;
    const std::size_t tableau_width = 2 * num_qubits + 1;

    std::vector<int> r_flat = {0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1};

    std::vector<int> x_flat = {1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0,
                               1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                               0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                               0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0};

    std::vector<int> z_flat = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
                               1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0,
                               1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
                               1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0};

    std::vector<std::vector<BatchCliffordStateKokkos<TestType>>> states(
        num_qubits,
        std::vector<BatchCliffordStateKokkos<TestType>>(
            num_qubits, BatchCliffordStateKokkos<TestType>(
                            x_flat, z_flat, r_flat, num_qubits, batch_size)));

    using KokkosExecSpace = Kokkos::DefaultExecutionSpace;
    Kokkos::RangePolicy<KokkosExecSpace> policy(0, batch_size);

    using UnmanagedHostVectorView =
        Kokkos::View<TestType *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    Kokkos::View<TestType *> results("results", batch_size);
    auto unmanaged_results =
        UnmanagedHostVectorView(results.data(), batch_size);

    std::map<std::pair<size_t, size_t>, size_t> expected_map;

    std::vector<std::tuple<int, int, int>> expected = {
        std::make_tuple(0, 1, 0),  std::make_tuple(0, 2, 0),
        std::make_tuple(0, 3, 1),  std::make_tuple(0, 4, 0),
        std::make_tuple(0, 5, 1),  std::make_tuple(0, 6, 1),
        std::make_tuple(0, 7, 0),  std::make_tuple(0, 8, 0),
        std::make_tuple(0, 9, 0),  std::make_tuple(0, 10, 1),
        std::make_tuple(1, 0, 0),  std::make_tuple(1, 2, 0),
        std::make_tuple(1, 3, 1),  std::make_tuple(1, 4, 1),
        std::make_tuple(1, 5, 0),  std::make_tuple(1, 6, 1),
        std::make_tuple(1, 7, 0),  std::make_tuple(1, 8, 0),
        std::make_tuple(1, 9, 0),  std::make_tuple(1, 10, 1),
        std::make_tuple(2, 0, 0),  std::make_tuple(2, 1, 0),
        std::make_tuple(2, 3, 0),  std::make_tuple(2, 4, 1),
        std::make_tuple(2, 5, 1),  std::make_tuple(2, 6, 1),
        std::make_tuple(2, 7, 1),  std::make_tuple(2, 8, 1),
        std::make_tuple(2, 9, 0),  std::make_tuple(2, 10, 1),
        std::make_tuple(3, 0, 1),  std::make_tuple(3, 1, 1),
        std::make_tuple(3, 2, 0),  std::make_tuple(3, 4, 0),
        std::make_tuple(3, 5, 1),  std::make_tuple(3, 6, 1),
        std::make_tuple(3, 7, 1),  std::make_tuple(3, 8, 0),
        std::make_tuple(3, 9, 0),  std::make_tuple(3, 10, 0),
        std::make_tuple(4, 0, 0),  std::make_tuple(4, 1, 1),
        std::make_tuple(4, 2, 1),  std::make_tuple(4, 3, 0),
        std::make_tuple(4, 5, 0),  std::make_tuple(4, 6, 1),
        std::make_tuple(4, 7, 0),  std::make_tuple(4, 8, 0),
        std::make_tuple(4, 9, 0),  std::make_tuple(4, 10, 1),
        std::make_tuple(5, 0, 0),  std::make_tuple(5, 1, 0),
        std::make_tuple(5, 2, 1),  std::make_tuple(5, 3, 1),
        std::make_tuple(5, 4, 0),  std::make_tuple(5, 6, 1),
        std::make_tuple(5, 7, 0),  std::make_tuple(5, 8, 1),
        std::make_tuple(5, 9, 1),  std::make_tuple(5, 10, 1),
        std::make_tuple(6, 0, 1),  std::make_tuple(6, 1, 0),
        std::make_tuple(6, 2, 1),  std::make_tuple(6, 3, 1),
        std::make_tuple(6, 4, 1),  std::make_tuple(6, 5, 1),
        std::make_tuple(6, 7, 1),  std::make_tuple(6, 8, 1),
        std::make_tuple(6, 9, 1),  std::make_tuple(6, 10, 0),
        std::make_tuple(7, 0, 0),  std::make_tuple(7, 1, 0),
        std::make_tuple(7, 2, 0),  std::make_tuple(7, 3, 1),
        std::make_tuple(7, 4, 0),  std::make_tuple(7, 5, 0),
        std::make_tuple(7, 6, 1),  std::make_tuple(7, 8, 0),
        std::make_tuple(7, 9, 0),  std::make_tuple(7, 10, 1),
        std::make_tuple(8, 0, 0),  std::make_tuple(8, 1, 0),
        std::make_tuple(8, 2, 1),  std::make_tuple(8, 3, 1),
        std::make_tuple(8, 4, 0),  std::make_tuple(8, 5, 1),
        std::make_tuple(8, 6, 1),  std::make_tuple(8, 7, 0),
        std::make_tuple(8, 9, 0),  std::make_tuple(8, 10, 1),
        std::make_tuple(9, 0, 0),  std::make_tuple(9, 1, 0),
        std::make_tuple(9, 2, 0),  std::make_tuple(9, 3, 0),
        std::make_tuple(9, 4, 1),  std::make_tuple(9, 5, 1),
        std::make_tuple(9, 6, 1),  std::make_tuple(9, 7, 0),
        std::make_tuple(9, 8, 0),  std::make_tuple(9, 10, 1),
        std::make_tuple(10, 0, 1), std::make_tuple(10, 1, 1),
        std::make_tuple(10, 2, 1), std::make_tuple(10, 3, 0),
        std::make_tuple(10, 4, 1), std::make_tuple(10, 5, 1),
        std::make_tuple(10, 6, 0), std::make_tuple(10, 7, 1),
        std::make_tuple(10, 8, 1), std::make_tuple(10, 9, 1),
    };

    for (const auto &tuple : expected) {
      expected_map[std::make_pair(std::get<0>(tuple), std::get<1>(tuple))] =
          std::get<2>(tuple);
    }

    std::vector<std::vector<int>> tableau{
        {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0},
        {1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0}, {1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1},
        {0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0}, {0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0},
        {0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1}, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0},
        {1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0}, {1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1},
    };

    auto expected_checksum = std::accumulate(r_flat.begin(), r_flat.end(), 0) +
                             std::accumulate(x_flat.begin(), x_flat.end(), 0) +
                             std::accumulate(z_flat.begin(), z_flat.end(), 0);

    // size_t i == 4;
    // size_t j == 3;

    for (size_t i = 0; i < num_qubits; i++) {
      for (size_t j = 0; j < num_qubits; j++) {
        if (i != j) {
          Kokkos::parallel_for(policy,
                               BatchRowProductSignFunctor<TestType>(
                                   states[i][j].GetX(), states[i][j].GetZ(),
                                   states[i][j].GetR(), results, i, j));
          Kokkos::deep_copy(unmanaged_results, results);
          REQUIRE(results[0] == expected_map[{i, j}]);
          REQUIRE(states[i][j].CheckSum() == expected_checksum);
        }
      }
    }
  }
}

TEMPLATE_TEST_CASE("CliffordStateKokkos::BatchRowMultiplyFunctor",
                   "[CliffordStateKokkos]", int) {

  {
    const std::size_t num_qubits = 5;
    const std::size_t batch_size = 1;
    const std::size_t tableau_width = 2 * num_qubits + 1;

    std::vector<int> r_flat = {0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1};

    std::vector<int> x_flat = {1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0,
                               1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                               0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                               0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0};

    std::vector<int> z_flat = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
                               1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0,
                               1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
                               1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0};

    std::vector<std::vector<BatchCliffordStateKokkos<TestType>>> states(
        num_qubits,
        std::vector<BatchCliffordStateKokkos<TestType>>(
            num_qubits, BatchCliffordStateKokkos<TestType>(
                            x_flat, z_flat, r_flat, num_qubits, batch_size)));

    using KokkosExecSpace = Kokkos::DefaultExecutionSpace;
    Kokkos::RangePolicy<KokkosExecSpace> policy(0, batch_size);

    using UnmanagedHostVectorView =
        Kokkos::View<TestType *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    std::vector<std::tuple<int, int, int>> expected = {
        std::make_tuple(0, 1, 41),  std::make_tuple(0, 2, 44),
        std::make_tuple(0, 3, 44),  std::make_tuple(0, 4, 45),
        std::make_tuple(0, 5, 44),  std::make_tuple(0, 6, 44),
        std::make_tuple(0, 7, 41),  std::make_tuple(0, 8, 42),
        std::make_tuple(0, 9, 44),  std::make_tuple(0, 10, 41),
        std::make_tuple(1, 0, 39),  std::make_tuple(1, 2, 40),
        std::make_tuple(1, 3, 42),  std::make_tuple(1, 4, 44),
        std::make_tuple(1, 5, 43),  std::make_tuple(1, 6, 42),
        std::make_tuple(1, 7, 41),  std::make_tuple(1, 8, 40),
        std::make_tuple(1, 9, 40),  std::make_tuple(1, 10, 41),
        std::make_tuple(2, 0, 39),  std::make_tuple(2, 1, 37),
        std::make_tuple(2, 3, 41),  std::make_tuple(2, 4, 40),
        std::make_tuple(2, 5, 42),  std::make_tuple(2, 6, 40),
        std::make_tuple(2, 7, 42),  std::make_tuple(2, 8, 41),
        std::make_tuple(2, 9, 36),  std::make_tuple(2, 10, 41),
        std::make_tuple(3, 0, 39),  std::make_tuple(3, 1, 39),
        std::make_tuple(3, 2, 41),  std::make_tuple(3, 4, 38),
        std::make_tuple(3, 5, 41),  std::make_tuple(3, 6, 39),
        std::make_tuple(3, 7, 39),  std::make_tuple(3, 8, 35),
        std::make_tuple(3, 9, 39),  std::make_tuple(3, 10, 39),
        std::make_tuple(4, 0, 41),  std::make_tuple(4, 1, 42),
        std::make_tuple(4, 2, 41),  std::make_tuple(4, 3, 39),
        std::make_tuple(4, 5, 39),  std::make_tuple(4, 6, 38),
        std::make_tuple(4, 7, 39),  std::make_tuple(4, 8, 40),
        std::make_tuple(4, 9, 38),  std::make_tuple(4, 10, 41),
        std::make_tuple(5, 0, 41),  std::make_tuple(5, 1, 43),
        std::make_tuple(5, 2, 45),  std::make_tuple(5, 3, 44),
        std::make_tuple(5, 4, 41),  std::make_tuple(5, 6, 42),
        std::make_tuple(5, 7, 41),  std::make_tuple(5, 8, 45),
        std::make_tuple(5, 9, 43),  std::make_tuple(5, 10, 41),
        std::make_tuple(6, 0, 41),  std::make_tuple(6, 1, 40),
        std::make_tuple(6, 2, 42),  std::make_tuple(6, 3, 41),
        std::make_tuple(6, 4, 39),  std::make_tuple(6, 5, 41),
        std::make_tuple(6, 7, 41),  std::make_tuple(6, 8, 42),
        std::make_tuple(6, 9, 40),  std::make_tuple(6, 10, 39),
        std::make_tuple(7, 0, 41),  std::make_tuple(7, 1, 43),
        std::make_tuple(7, 2, 46),  std::make_tuple(7, 3, 44),
        std::make_tuple(7, 4, 43),  std::make_tuple(7, 5, 43),
        std::make_tuple(7, 6, 44),  std::make_tuple(7, 8, 42),
        std::make_tuple(7, 9, 46),  std::make_tuple(7, 10, 41),
        std::make_tuple(8, 0, 39),  std::make_tuple(8, 1, 39),
        std::make_tuple(8, 2, 43),  std::make_tuple(8, 3, 38),
        std::make_tuple(8, 4, 41),  std::make_tuple(8, 5, 44),
        std::make_tuple(8, 6, 42),  std::make_tuple(8, 7, 39),
        std::make_tuple(8, 9, 42),  std::make_tuple(8, 10, 41),
        std::make_tuple(9, 0, 39),  std::make_tuple(9, 1, 37),
        std::make_tuple(9, 2, 36),  std::make_tuple(9, 3, 39),
        std::make_tuple(9, 4, 38),  std::make_tuple(9, 5, 40),
        std::make_tuple(9, 6, 38),  std::make_tuple(9, 7, 41),
        std::make_tuple(9, 8, 40),  std::make_tuple(9, 10, 41),
        std::make_tuple(10, 0, 41), std::make_tuple(10, 1, 43),
        std::make_tuple(10, 2, 46), std::make_tuple(10, 3, 44),
        std::make_tuple(10, 4, 45), std::make_tuple(10, 5, 43),
        std::make_tuple(10, 6, 42), std::make_tuple(10, 7, 41),
        std::make_tuple(10, 8, 44), std::make_tuple(10, 9, 46),
    };

    std::map<std::pair<size_t, size_t>, size_t> expected_map;
    for (const auto &tuple : expected) {
      expected_map[std::make_pair(std::get<0>(tuple), std::get<1>(tuple))] =
          std::get<2>(tuple);
    }

    for (size_t i = 0; i < num_qubits; i++) {
      for (size_t j = 0; j < num_qubits; j++) {
        if (i != j) {
          Kokkos::parallel_for(policy,
                               BatchRowMultiplyFunctor<TestType>(
                                   states[i][j].GetX(), states[i][j].GetZ(),
                                   states[i][j].GetR(), i, j));
          auto sum = states[i][j].CheckSum();
          REQUIRE(sum == expected_map[std::make_pair(i, j)]);
        }
      }
    }
  }
}

TEMPLATE_TEST_CASE("CliffordStateKokkos::BatchMeasureDeterminedFunctor",
                   "[CliffordStateKokkos]", int) {

  {
    const std::size_t num_qubits = 5;
    const std::size_t batch_size = 1;
    const std::size_t tableau_width = 2 * num_qubits + 1;

    std::vector<int> r_flat = {0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1};

    std::vector<int> x_flat = {1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0,
                               1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0,
                               0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                               0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0};

    std::vector<int> z_flat = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
                               1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0,
                               1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0,
                               1, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0};

    std::vector<BatchCliffordStateKokkos<TestType>> states(
        num_qubits, BatchCliffordStateKokkos<TestType>(x_flat, z_flat, r_flat,
                                                       num_qubits, batch_size));

    std::vector<Kokkos::View<TestType *, Kokkos::DefaultExecutionSpace>>
        measurement_results(
            num_qubits, Kokkos::View<TestType *, Kokkos::DefaultExecutionSpace>(
                            "measurement_results", batch_size));

    std::vector<std::vector<TestType>> measurement_results_host(
        num_qubits, std::vector<TestType>(batch_size));

    std::vector<TestType> expected_results = {1, 1, 0, 0, 0};

    using KokkosExecSpace = Kokkos::DefaultExecutionSpace;
    Kokkos::RangePolicy<KokkosExecSpace> policy(0, batch_size);

    using UnmanagedHostVectorView =
        Kokkos::View<TestType *, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    for (size_t q = 0; q < num_qubits; q++) {
      Kokkos::parallel_for(policy,
                           BatchMeasureDeterminedFunctor<TestType>(
                               states[q].GetX(), states[q].GetZ(),
                               states[q].GetR(), measurement_results[q], q));
      Kokkos::deep_copy(UnmanagedHostVectorView(
                            measurement_results_host[q].data(), batch_size),
                        measurement_results[q]);

      REQUIRE(measurement_results_host[q][0] == expected_results[q]);
    }
  }
}

TEMPLATE_TEST_CASE("CliffordStateKokkos::MeasureRandom",
                   "[CliffordStateKokkos]", int) {

  {
    const std::size_t num_qubits = 2;
    const std::size_t batch_size = 1;

    float bias_1 = 0;
    int check_sum_1 = 9;
    BatchCliffordStateKokkos<TestType> kokkos_state_1(num_qubits, batch_size);
    kokkos_state_1.ApplyHadamardGate(1);
    kokkos_state_1.ApplyPhaseGate(1);
    kokkos_state_1.ApplyHadamardGate(0);
    kokkos_state_1.ApplyControlNotGate(0, 1);
    // This will be nondeterministic
    kokkos_state_1.MeasureQubit(1, bias_1);
    REQUIRE(kokkos_state_1.CheckSum() == check_sum_1);

    float bias_2 = 1;
    int check_sum_2 = 10;
    BatchCliffordStateKokkos<TestType> kokkos_state_2(num_qubits, batch_size);
    kokkos_state_2.ApplyHadamardGate(1);
    kokkos_state_2.ApplyPhaseGate(1);
    kokkos_state_2.ApplyHadamardGate(0);
    kokkos_state_2.ApplyControlNotGate(0, 1);
    // This will be nondeterministic
    kokkos_state_2.MeasureQubit(1, bias_2);
    REQUIRE(kokkos_state_2.CheckSum() == check_sum_2);
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
    REQUIRE(result_0.value().second == 0);

    if (result_0.value().first) {
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

TEST_CASE("S-State Distillation Low Space Test",
          "[s_state_distillation_low_space]") {

  for (int test_case = 0; test_case < 100; ++test_case) {

    const std::size_t num_qubits = 5;
    const std::size_t batch_size = 1;
    BatchCliffordStateKokkos<int> sim(num_qubits, batch_size, test_case);
    size_t batch_id = 0;
    // Define phasors.
    std::vector<std::vector<int>> phasors = {
        {0}, {1}, {2}, {0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};

    int anc = 4;
    for (const auto &phasor : phasors) {
      sim.ApplyHadamardGate(anc);
      for (int k : phasor) {
        sim.ApplyControlNotGate(anc, k);
      }
      sim.ApplyHadamardGate(anc);
      sim.ApplyPhaseGate(anc);
      sim.ApplyHadamardGate(anc);

      auto measurement_id = sim.MeasureQubit(anc);
      auto v = sim.GetMeasurement(measurement_id, batch_id);

      REQUIRE(v.value().second ==
              false); // Check that the measurement was not // determined.
      if (v.value().first) {
        for (int k : phasor) {
          sim.ApplyHadamardGate(k);
          sim.ApplyPhaseGate(k);
          sim.ApplyPhaseGate(k);
          sim.ApplyHadamardGate(k);
        }
        sim.ApplyHadamardGate(anc);
        sim.ApplyPhaseGate(anc);
        sim.ApplyPhaseGate(anc);
        sim.ApplyHadamardGate(anc);
      }
    }

    for (int k = 0; k < 3; ++k) {
      auto measurement_id = sim.MeasureQubit(k);
      auto result = sim.GetMeasurement(measurement_id, batch_id);

      REQUIRE(result.value().first == false); // Check the measured value for
      REQUIRE(result.value().second == true); // Check that
    }

    sim.ApplyPhaseGate(3);
    sim.ApplyHadamardGate(3);
    auto measurement_id = sim.MeasureQubit(3);
    auto result = sim.GetMeasurement(measurement_id, batch_id);

    REQUIRE(result.value().first == true);  // Check the measured value for
    REQUIRE(result.value().second == true); // Check that the
  }
}

TEMPLATE_TEST_CASE("BatchCliffordStateKokkos::PhaseKickbackBatched",
                   "[batch_clifford] [hadamard]", int) {

  {
    const std::size_t num_qubits = 2;
    const std::size_t batch_size = 2;

    BatchCliffordStateKokkos<TestType> kokkos_state_1(num_qubits, batch_size);
    kokkos_state_1.ApplyHadamardGate(1);
    kokkos_state_1.ApplyPhaseGate(1);
    kokkos_state_1.ApplyHadamardGate(0);
    kokkos_state_1.ApplyControlNotGate(0, 1);

    auto measurement_id = kokkos_state_1.MeasureQubit(1);
    auto &measurement = kokkos_state_1.GetMeasurement(measurement_id);
    auto result_0_0 = kokkos_state_1.GetMeasurement(measurement_id, 0);
    auto result_0_1 = kokkos_state_1.GetMeasurement(measurement_id, 1);
    REQUIRE(result_0_0.value().second == 0);
    REQUIRE(result_0_1.value().second == 0);

    kokkos_state_1.ApplyPhaseGate(0, *measurement.measurement_results_device);
    kokkos_state_1.ApplyPhaseGate(0, *measurement.measurement_results_device);
    kokkos_state_1.ApplyPhaseGate(0);
    kokkos_state_1.ApplyHadamardGate(0);

    measurement_id = kokkos_state_1.MeasureQubit(0);
    auto result_1_0 = kokkos_state_1.GetMeasurement(measurement_id, 0);
    auto result_1_1 = kokkos_state_1.GetMeasurement(measurement_id, 1);
    REQUIRE(result_1_0.value().second == 1); // Check that the measurement was
    REQUIRE(result_1_0.value().first == 1);  // Check the measured value.
    REQUIRE(result_1_1.value().second == 1); // Check that the measurement was
    REQUIRE(result_1_1.value().first == 1);  // Check the measured value.
  }
}

TEST_CASE("S-State Distillation Low Space Test Batched",
          "[s_state_distillation_low_space]") {

  const std::size_t num_qubits = 5;
  const std::size_t batch_size = 100;
  BatchCliffordStateKokkos<int> sim(num_qubits, batch_size, 1231232132);

  // Define phasors.
  std::vector<std::vector<int>> phasors = {
      {0}, {1}, {2}, {0, 1, 2}, {0, 1, 3}, {0, 2, 3}, {1, 2, 3}};

  int anc = 4;
  for (const auto &phasor : phasors) {
    sim.ApplyHadamardGate(anc);
    for (int k : phasor) {
      sim.ApplyControlNotGate(anc, k);
    }
    sim.ApplyHadamardGate(anc);
    sim.ApplyPhaseGate(anc);
    sim.ApplyHadamardGate(anc);

    auto measurement_id = sim.MeasureQubit(anc);
    auto &measurement = sim.GetMeasurement(measurement_id);

    for (size_t batch_id = 0; batch_id < batch_size; batch_id++) {
      auto v = sim.GetMeasurement(measurement_id, batch_id);
      REQUIRE(v.value().second == false);
    }

    for (int k : phasor) {
      sim.ApplyHadamardGate(k, *measurement.measurement_results_device);
      sim.ApplyPhaseGate(k, *measurement.measurement_results_device);
      sim.ApplyPhaseGate(k, *measurement.measurement_results_device);
      sim.ApplyHadamardGate(k, *measurement.measurement_results_device);
    }
    sim.ApplyHadamardGate(anc, *measurement.measurement_results_device);
    sim.ApplyPhaseGate(anc, *measurement.measurement_results_device);
    sim.ApplyPhaseGate(anc, *measurement.measurement_results_device);
    sim.ApplyHadamardGate(anc, *measurement.measurement_results_device);
  }

  for (int k = 0; k < 3; ++k) {
    auto measurement_id = sim.MeasureQubit(k);
    for (size_t batch_id = 0; batch_id < batch_size; batch_id++) {
      auto result = sim.GetMeasurement(measurement_id, batch_id);
      REQUIRE(result.value().first == false); // Check the measured value for
      REQUIRE(result.value().second == true); // Check that
    }
  }

  sim.ApplyPhaseGate(3);
  sim.ApplyHadamardGate(3);
  auto measurement_id = sim.MeasureQubit(3);
  for (size_t batch_id = 0; batch_id < batch_size; batch_id++) {
    auto result = sim.GetMeasurement(measurement_id, batch_id);
    REQUIRE(result.value().first == true);  // Check the measured value for
    REQUIRE(result.value().second == true); // Check that
  }

}

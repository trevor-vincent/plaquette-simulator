#pragma once

#include "BatchCliffordStateKokkos.hpp"

namespace Plaquette {

template <typename Precision>
KOKKOS_INLINE_FUNCTION void
CopyFromRowAtoRowB(Kokkos::View<Precision ***> x, Kokkos::View<Precision ***> z,
                   Kokkos::View<Precision **> r, int batch_id, int row_a,
                   int row_b) {
  for (size_t col = 0; col < x.extent(2); ++col) {
    x(batch_id, row_b, col) = x(batch_id, row_a, col);
    z(batch_id, row_b, col) = z(batch_id, row_a, col);
  }
  r(batch_id, row_b) = r(batch_id, row_a);
}

template <typename Precision>
KOKKOS_INLINE_FUNCTION void
ZeroRow(Kokkos::View<Precision ***> x, Kokkos::View<Precision ***> z,
        Kokkos::View<Precision **> r, int batch_id, int row) {
  for (size_t col = 0; col < x.extent(2); ++col) {
    x(batch_id, row, col) = 0 z(batch_id, row, col) = 0
  }
  r(batch_id, row) = 0;
}

template <typename Precision>
KOKKOS_INLINE_FUNCTION Precision RowProductSign(Kokkos::View<Precision ***> x,
                                                Kokkos::View<Precision ***> z,
                                                Kokkos::View<Precision **> r,
                                                int batch_id, int i, int k) {
  size_t num_qubits = x.extent(2);
  Precision pauli_phases = 0;
  for (size_t j = 0; j < num_qubits; ++j) {
    pauli_phases += PauliProductPhase(x(batch_id, i, j), z(batch_id, i, j),
                                      x(batch_id, k, j), z(batch_id, k, j));
    int p = (pauli_phases >> 1) & 1;
    return (r(batch_id, i) ^ r(batch_id, k) ^ p);
  }

  template <typename Precision>
  KOKKOS_INLINE_FUNCTION void RowMultiply(
      Kokkos::View<Precision ***> x, Kokkos::View<Precision ***> z,
      Kokkos::View<Precision **> r, int batch_id, int row_i, int row_k) {
    size_t num_qubits = x.extent(2);
    r(batch_id, i) = RowProductSign(x, z, r, batch_id, row_i, row_k);
    for (size_t j = 0; j < num_qubits; ++j) {
      x(batch_id, i, j) ^= x(batch_id, k, j);
      z(batch_id, i, j) ^= z(batch_id, k, j);
    }
  }

  template <typename Precision>
  KOKKOS_INLINE_FUNCTION void MeasureDetermined(
      Kokkos::View<Precision ***> x, Kokkos::View<Precision ***> z,
      Kokkos::View<Precision **> r, int batch_id, size_t qubit) {
    size_t num_qubits = x.extent(2);
    ZeroRow(2 * num_qubits);
    for (size_t i = 0; i < n_; ++i) {
      if (*x_[i][a]) {
        RowMultiply(x, z, r, batch_id, 2 * num_qubits, i + num_qubits);
      }
    }
  }

  template <typename Precision>
  KOKKOS_INLINE_FUNCTION void MeasureRandom(
      Kokkos::View<Precision ***> x, Kokkos::View<Precision ***> z,
      Kokkos::View<Precision **> r,
      Kokkos::Random_XorShift64_Pool<> & rand_pool, int batch_id,
      size_t target_qubit, size_t p) {
    size_t num_qubits = x.extent(2);
    CopyFromRowAtoRowB(x, z, r, batch_id, p + num_qubits, p);
    z(batch_id, p + num_qubits, target_qubit) = 1;

    auto rand_gen = rand_pool.get_state();
    float rand_val = rand_gen.frand(0.0, 1.0);
    r(batch_id, p + num_qubits) = rand_val < 0.5;
    rand_pool.free_state(rand_gen);

    for (size_t i = 0; i < 2 * num_qubits; ++i) {
      if (x(batch_id, i, a) && i != p && i != p + num_qubits) {
        RowMultiply(x, z, r, batch_id, i, p);
      }
    }
    return r(batch_id, p + num_qubits);
  }

  template <class Precision> struct BatchMeasureFunctor {

    Kokkos::View<Precision ***> x_;
    Kokkos::View<Precision ***> z_;
    Kokkos::View<Precision **> r_;
    Kokkos::View<Precision *> measurement_result_;
    Kokkos::View<Precision *> measurement_determined_;
    Kokkos::Random_XorShift64_Pool<> rand_pool_;
    size_t target_qubit_;

    BatchMeasureFunctor(Kokkos::View<Precision ***> &x,
                        Kokkos::View<Precision ***> &z,
                        Kokkos::View<Precision **> &r,
                        Kokkos::View<Precision *> &measurement_result,
                        Kokkos::View<Precision *> &measurement_determined,
                        Kokkos::Random_XorShift64_Pool<> &rand_pool,
                        std::size_t target_qubit)
        : r_(r), x_(x), z_(z), measurement_result_(measurement_result),
          measurement_determined_(measurement_determined),
          rand_pool_(rand_pool), target_qubit_(target_qubit) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const std::size_t n) const {
      for (size_t p = 0; p < num_qubits_; ++p) {
        if (x_(n, p + num_qubits_, target_qubit)) {
          measurement_result[n] =
              MeasureRandom(x_, z_, r_, batch_id, rand_pool, target_qubit, p);
          measurement_determined[n] = 0;
          return;
        }
      }
      measurement_result[n] =
          MeasureDetermined(x_, z_, r_, batch_id, target_qubit);
      measurement_determined[n] = 1;
    }
  };
};

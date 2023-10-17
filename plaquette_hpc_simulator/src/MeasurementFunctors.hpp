#pragma once

#include "BatchCliffordStateKokkos.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

namespace Plaquette {

template <typename Precision>
KOKKOS_INLINE_FUNCTION Precision PauliProductPhase(Precision x1, Precision z1,
                                                   Precision x2, Precision z2) {
  // Y
  if (x1 == 1 and z1 == 1) {
    // YI
    if (x2 == 0 and z2 == 0) {
      return 0;
    }
    // YX
    else if (x2 == 1 and z2 == 0) {
      return -1;
    }
    // YY
    else if (x2 == 1 and z2 == 1) {
      return 0;
    }
    // YZ
    else {
      return 1;
    }
  }
  // X
  else if (x1 == 1 and z1 == 0) {
    // XI
    if (x2 == 0 and z2 == 0) {
      return 0;
    }
    // XX
    else if (x2 == 1 and z2 == 0) {
      return 0;
    }
    // XY
    else if (x2 == 1 and z2 == 1) {
      return +1;
    }
    // XZ
    else {
      return -1;
    }
  }
  // Z
  else if (z1 == 1 and x1 == 0) {
    // ZI
    if (x2 == 0 and z2 == 0) {
      return 0;
    }
    // ZX
    else if (x2 == 1 and z2 == 0) {
      return +1;
    }
    // ZY
    else if (x2 == 1 and z2 == 1) {
      return -1;
    }
    // ZZ
    else {
      return 0;
    }
  } else {
    return 0;
  }
}

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
    x(batch_id, row, col) = 0;
    z(batch_id, row, col) = 0;
  }
  r(batch_id, row) = 0;
}

template <typename Precision>
KOKKOS_INLINE_FUNCTION Precision RowProductSign(Kokkos::View<Precision ***> x,
                                                Kokkos::View<Precision ***> z,
                                                Kokkos::View<Precision **> r,
                                                int batch_id, int i, int k) {

  // std::cout << std::endl << std::endl;
  // std::cout << "==============================" << std::endl;
  // std::cout << "Kokkos Row Product Sign" << std::endl;
  // std::cout << "i = " << i << std::endl;
  // std::cout << "k = " << k << std::endl;
  size_t num_qubits = x.extent(2);
  // std::cout << "num_qubits = " << num_qubits << std::endl;
  Precision pauli_phases = 0;
  for (size_t j = 0; j < num_qubits; ++j) {
    pauli_phases += PauliProductPhase(x(batch_id, i, j), z(batch_id, i, j),
                                      x(batch_id, k, j), z(batch_id, k, j));
    // std::cout << j << " pauli_phases = " << pauli_phases << std::endl;
    // std::cout << "x(batch_id, i, j) = " << x(batch_id, i, j) << std::endl;
    // std::cout << "z(batch_id, i, j) = " << z(batch_id, i, j) << std::endl;
    // std::cout << "x(batch_id, k, j) = " << x(batch_id, k, j) << std::endl;
    // std::cout << "z(batch_id, k, j) = " << z(batch_id, k, j) << std::endl;
  }
  int p = (pauli_phases >> 1) & 1;
  // std::cout << "p = " << p << std::endl;
  // std::cout << "(r(batch_id, i) ^ r(batch_id, k) ^ p) = "
  // << (r(batch_id, i) ^ r(batch_id, k) ^ p) << std::endl;
  // std::cout << "==============================" << std::endl;
  // std::cout << std::endl << std::endl;

  return (r(batch_id, i) ^ r(batch_id, k) ^ p);
}

// mainly for testing
template <class Precision> struct BatchRowProductSignFunctor {

  Kokkos::View<Precision ***> x_;
  Kokkos::View<Precision ***> z_;
  Kokkos::View<Precision **> r_;
  Kokkos::View<Precision *> results_vector_;
  size_t row_i_;
  size_t row_k_;

  BatchRowProductSignFunctor(Kokkos::View<Precision ***> x,
                             Kokkos::View<Precision ***> z,
                             Kokkos::View<Precision **> r,
                             Kokkos::View<Precision *> results_vector,
                             std::size_t row_i, std::size_t row_k)
      : x_(x), z_(z), r_(r), row_i_(row_i), row_k_(row_k),
        results_vector_(results_vector) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t batch_id) const {
    results_vector_[batch_id] =
        RowProductSign(x_, z_, r_, batch_id, row_i_, row_k_);
  }
};

template <typename Precision>
KOKKOS_INLINE_FUNCTION void
RowMultiply(Kokkos::View<Precision ***> x, Kokkos::View<Precision ***> z,
            Kokkos::View<Precision **> r, int batch_id, int row_i, int row_k) {
  size_t num_qubits = x.extent(2);
  r(batch_id, row_i) = RowProductSign(x, z, r, batch_id, row_i, row_k);
  for (size_t col = 0; col < num_qubits; ++col) {
    x(batch_id, row_i, col) ^= x(batch_id, row_k, col);
    z(batch_id, row_i, col) ^= z(batch_id, row_k, col);
  }
}

// mainly for testing
template <class Precision> struct BatchRowMultiplyFunctor {

  Kokkos::View<Precision ***> x_;
  Kokkos::View<Precision ***> z_;
  Kokkos::View<Precision **> r_;
  size_t row_i_;
  size_t row_k_;

  BatchRowMultiplyFunctor(Kokkos::View<Precision ***> x,
                          Kokkos::View<Precision ***> z,
                          Kokkos::View<Precision **> r, std::size_t row_i,
                          std::size_t row_k)
      : r_(r), x_(x), z_(z), row_i_(row_i), row_k_(row_k) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t batch_id) const {
    RowMultiply(x_, z_, r_, batch_id, row_i_, row_k_);
  }
};

template <typename Precision>
KOKKOS_INLINE_FUNCTION Precision
MeasureDetermined(Kokkos::View<Precision ***> x, Kokkos::View<Precision ***> z,
                  Kokkos::View<Precision **> r, int batch_id, size_t qubit) {
  size_t num_qubits = x.extent(2);
  ZeroRow(x, z, r, batch_id, 2 * num_qubits);
  for (size_t i = 0; i < num_qubits; ++i) {
    if (x(batch_id, i, qubit)) {
      RowMultiply(x, z, r, batch_id, 2 * num_qubits, i + num_qubits);
    }
  }
  return r(batch_id, 2 * num_qubits);
}

template <typename Precision>
KOKKOS_INLINE_FUNCTION Precision CheckSumInline(Kokkos::View<Precision ***> x,
                                                Kokkos::View<Precision ***> z,
                                                Kokkos::View<Precision **> r) {
  
  size_t num_qubits = x.extent(2);
  size_t tableau_width = x.extent(1);
  size_t batch_size = x.extent(0);

  Precision sum = 0;
  for (size_t b = 0; b < batch_size; b++){
    for (size_t t = 0; t < tableau_width; t++){
      sum += r(b,t);
      for (size_t q = 0; q < num_qubits; q++){
	sum += x(b,t,q);
	sum += z(b,t,q);
      }
    }
  }
  return sum;
}

template <typename Precision>
KOKKOS_INLINE_FUNCTION Precision MeasureRandom(
    Kokkos::View<Precision ***> x, Kokkos::View<Precision ***> z,
    Kokkos::View<Precision **> r, Kokkos::Random_XorShift64_Pool<> rand_pool,
    size_t batch_id, size_t target_qubit, size_t p, float bias) {
  size_t num_qubits = x.extent(2);

  // std::cout << "CheckSumInline(x,z,r) 0 = " << CheckSumInline(x,z,r) << std::endl;
  
  CopyFromRowAtoRowB(x, z, r, batch_id, p + num_qubits, p);
  // std::cout << "CheckSumInline(x,z,r) 1 = " << CheckSumInline(x,z,r) << std::endl;
  ZeroRow(x, z, r, batch_id, p + num_qubits);
  
  z(batch_id, p + num_qubits, target_qubit) = 1;

  // std::cout << "CheckSumInline(x,z,r) 2 = " << CheckSumInline(x,z,r) << std::endl;
  
  auto rand_gen = rand_pool.get_state();
  float rand_val = rand_gen.frand(0.0, 1.0);
  r(batch_id, p + num_qubits) = rand_val < bias;
  // std::cout << "r(batch_id, p + num_qubits) = " << r(batch_id, p + num_qubits)
            // << std::endl;
  rand_pool.free_state(rand_gen);

  // std::cout << "CheckSumInline(x,z,r) 3 = " << CheckSumInline(x,z,r) << std::endl;
  
  for (size_t i = 0; i < 2 * num_qubits; ++i) {
    if (x(batch_id, i, target_qubit) && i != p && i != p + num_qubits) {
      RowMultiply(x, z, r, batch_id, i, p);
    }
  }
  // std::cout << "CheckSumInline(x,z,r) 4 = " << CheckSumInline(x,z,r) << std::endl;
  
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
  size_t num_qubits_;
  float bias_;

  BatchMeasureFunctor(Kokkos::View<Precision ***> x,
                      Kokkos::View<Precision ***> z,
                      Kokkos::View<Precision **> r,
                      Kokkos::View<Precision *> measurement_result,
                      Kokkos::View<Precision *> measurement_determined,
                      Kokkos::Random_XorShift64_Pool<> rand_pool,
                      size_t target_qubit, float bias)
      : r_(r), x_(x), z_(z), measurement_result_(measurement_result),
        measurement_determined_(measurement_determined), rand_pool_(rand_pool),
        target_qubit_(target_qubit), bias_(bias) {
    num_qubits_ = x.extent(2);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_t batch_id) const {
    // std::cout << std::endl << std::endl;
    // std::cout << "==============================" << std::endl;
    // std::cout << "Kokkos Batch Measure" << std::endl;
    // std::cout << "num_qubits_ = " << num_qubits_ << std::endl;
    for (size_t p = 0; p < num_qubits_; ++p) {
      // std::cout << "p = " << p << std::endl;
      // std::cout << "x_(batch_id, p + num_qubits_, target_qubit_) = "
                // << x_(batch_id, p + num_qubits_, target_qubit_) << std::endl;
      if (x_(batch_id, p + num_qubits_, target_qubit_)) {
        // std::cout << "bias = " << bias_ << std::endl;
        // std::cout << "target_qubit_ = " << target_qubit_ << std::endl;
        measurement_result_[batch_id] = MeasureRandom<Precision>(
            x_, z_, r_, rand_pool_, batch_id, target_qubit_, p, bias_);
        measurement_determined_[batch_id] = 0;
        return;
      }
    }
    measurement_result_[batch_id] =
        MeasureDetermined(x_, z_, r_, batch_id, target_qubit_);
    measurement_determined_[batch_id] = 1;

    // std::cout << "==============================" << std::endl;
    // std::cout << std::endl << std::endl;
  }
};

// // mainly for testing
// template <class Precision> struct BatchMeasureDeterminedFunctor {

//   Kokkos::View<Precision ***> x_;
//   Kokkos::View<Precision ***> z_;
//   Kokkos::View<Precision **> r_;
//   Kokkos::View<Precision *> measurement_result_;
//   size_t target_qubit_;

//   BatchMeasureDeterminedFunctor(Kokkos::View<Precision ***> x,
//                       Kokkos::View<Precision ***> z,
//                       Kokkos::View<Precision **> r,
//                       Kokkos::View<Precision *> measurement_result,
//                       std::size_t target_qubit)
//       : r_(r), x_(x), z_(z), measurement_result_(measurement_result),
//         target_qubit_(target_qubit) {}

//   KOKKOS_INLINE_FUNCTION
//   void operator()(const std::size_t batch_id) const {
//     measurement_result_[batch_id] =
//         MeasureDetermined(x_, z_, r_, batch_id, target_qubit_);
//   }
// };

template <class Precision> struct BatchMeasureDeterminedFunctor {
  Kokkos::View<Precision ***> x_;
  Kokkos::View<Precision ***> z_;
  Kokkos::View<Precision **> r_;
  Kokkos::View<Precision *> measurement_result_;
  size_t target_qubit_;

  BatchMeasureDeterminedFunctor(Kokkos::View<Precision ***> x,
                                Kokkos::View<Precision ***> z,
                                Kokkos::View<Precision **> r,
                                Kokkos::View<Precision *> measurement_result,
                                std::size_t target_qubit)
      : r_(r), x_(x), z_(z), measurement_result_(measurement_result),
        target_qubit_(target_qubit) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t batch_id) const {
    measurement_result_[batch_id] =
        MeasureDetermined(x_, z_, r_, batch_id, target_qubit_);
  }
};

}; // namespace Plaquette

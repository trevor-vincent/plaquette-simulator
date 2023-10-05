#pragma once
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

namespace {
namespace KE = Kokkos::Experimental;
}

namespace Plaquette {

template <class Precision> struct BatchControlPhaseGateFunctor {

  Kokkos::View<Precision ***> x_;
  Kokkos::View<Precision ***> z_;
  Kokkos::View<Precision **> r_;
  size_t target_qubit_;
  size_t control_qubit_;

  BatchControlPhaseGateFunctor(Kokkos::View<Precision ***> &x,
                               Kokkos::View<Precision ***> &z,
                               Kokkos::View<Precision **> &r,
                               std::size_t target_qubit,
                               std::size_t control_qubit)
      : r_(r), x_(x), z_(z), target_qubit_(target_qubit),
        control_qubit_(control_qubit) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t n, const std::size_t bit) const {
    // r = r ^ (x_control * x_target_qubit_ *(z_target_qubit_ ^ z_control))
    r_(n,bit) ^= x_(n,bit,control_qubit_) * x_(n,bit,target_qubit_) *
      (z_(n,bit,target_qubit_) ^ z_(n,bit,control_qubit_));
    // z_target_qubit_ = z_target_qubit_ ^ x_control_qubit_
    z_(n,bit,target_qubit_) ^= x_(n,bit,control_qubit_);
    // z_control_qubit_ = z_control_qubit_ ^ x_target_qubit_
    z_(n,bit,control_qubit_) ^= x_(n,bit,target_qubit_);
  }
};

template <class Precision> struct BatchHadamardGateFunctor {

  Kokkos::View<Precision ***> x_;
  Kokkos::View<Precision ***> z_;
  Kokkos::View<Precision **> r_;
  size_t target_qubit_;

  BatchHadamardGateFunctor(Kokkos::View<Precision ***> &x,
                           Kokkos::View<Precision ***> &z,
                           Kokkos::View<Precision **> &r,
                           std::size_t target_qubit)
      : r_(r), x_(x), z_(z), target_qubit_(target_qubit) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t n, const std::size_t i) const {
    r_(n, i) ^= x_(n, i, target_qubit_) & z_(n, i, target_qubit_);
    KE::swap(x_(n, i, target_qubit_), z_(n, i, target_qubit_));
  }
};

template <class Precision> struct BatchPhaseGateFunctor {

  Kokkos::View<Precision ***> x_;
  Kokkos::View<Precision ***> z_;
  Kokkos::View<Precision **> r_;
  size_t target_qubit_;

  BatchPhaseGateFunctor(Kokkos::View<Precision ***> &x,
                        Kokkos::View<Precision ***> &z,
                        Kokkos::View<Precision **> &r, std::size_t target_qubit)
      : r_(r), x_(x), z_(z), target_qubit_(target_qubit) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t n, const std::size_t i) const {
    r_(n, i) = r_(n, i) ^ x_(n, i, target_qubit_) & z_(n, i, target_qubit_);
    z_(n, i, target_qubit_) = z_(n, i, target_qubit_) ^ x_(n, i, target_qubit_);
  }
};

template <class Precision> struct BatchControlNotGateFunctor {

  Kokkos::View<Precision ***> x_;
  Kokkos::View<Precision ***> z_;
  Kokkos::View<Precision **> r_;
  size_t target_qubit_;
  size_t control_qubit_;

  BatchControlNotGateFunctor(Kokkos::View<Precision ***> &x,
                             Kokkos::View<Precision ***> &z,
                             Kokkos::View<Precision **> &r,
                             std::size_t target_qubit,
                             std::size_t control_qubit)
      : r_(r), x_(x), z_(z), target_qubit_(target_qubit),
        control_qubit_(control_qubit) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t n, const std::size_t i) const {
    r_(n, i) =
        r_(n, i) ^ (x_(n, i, control_qubit_) & z_(n, i, target_qubit_) &
                    (x_(n, i, target_qubit_) ^ z_(n, i, control_qubit_) ^ 1));
    x_(n, i, target_qubit_) ^= x_(n, i, control_qubit_);
    z_(n, i, control_qubit_) ^= z_(n, i, target_qubit_);
  }
};

template <class Precision> struct BatchPauliXGateFunctor {

  Kokkos::View<Precision ***> x_;
  Kokkos::View<Precision ***> z_;
  Kokkos::View<Precision **> r_;
  size_t target_qubit_;

  BatchPauliXGateFunctor(Kokkos::View<Precision ***> &x,
                         Kokkos::View<Precision ***> &z,
                         Kokkos::View<Precision *&> &r, std::size_t num_qubits,
                         std::size_t target_qubit)
      : r_(r), x_(x), z_(z), target_qubit_(target_qubit) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t n, const std::size_t bit) const {
    // Set r = r ^ z_target
    r_(n, bit) = r_(n, bit) ^ z_(n, bit, target_qubit_);
  }
};

template <class Precision> struct BatchPauliZGateFunctor {

  Kokkos::View<Precision ***> x_;
  Kokkos::View<Precision ***> r_;
  Kokkos::View<Precision **> z_;
  size_t target_qubit_;

  BatchPauliZGateFunctor(Kokkos::View<Precision ***> &x,
                         Kokkos::View<Precision ***> &z,
                         Kokkos::View<Precision **> &r,
                         std::size_t target_qubit)
      : r_(r), x_(x), z_(z), target_qubit_(target_qubit) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t n, const std::size_t bit) const {
    // Set r = r ^ x_target
    r_(n, bit) = r_(n, bit) ^ x_(n, bit, target_qubit_);
  }
};

}; // namespace Plaquette

#pragma once
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>

namespace {
namespace KE = Kokkos::Experimental;
}

namespace Plaquette {

// template <class Precision> struct CNotFunctor {

//   Kokkos::View<Precision *> r_;
//   Kokkos::View<Precision *> x_;
//   Kokkos::View<Precision *> z_;

//   size_t target_qubit_;
//   size_t control_qubit_;
//   size_t num_x_or_z_rows_;

//   CNotFunctor(Kokkos::View<Precision *> &r, Kokkos::View<Precision *> &x,
//               Kokkos::View<Precision *> &z, std::size_t num_qubits,
//               std::size_t target_qubit, std::size_t control_qubit)
//       : r_(r), x_(x), z_(z), target_qubit_(target_qubit),
//         control_qubit_(control_qubit) {
//     num_x_or_z_rows_ = 2 * num_qubits + 1;
//   }

//   KOKKOS_INLINE_FUNCTION
//   void operator()(const std::size_t bit) const {
//     // r = r ^ (x_control * z_target *(x_target ^ z_control ^1))
//     r[bit] = r[bit] ^ x[bit * x_or_z_rows_ + control] *
//                           z[bit * x_or_z_rows_ + target] *
//                           (x[bit * x_or_z_rows_ + target] ^
//                            z[bit * x_or_z_rows_ + control] ^ 1);
//     // x_target = x_target ^ x_control
//     x[bit * x_or_z_rows_ + target] ^= x[bit * x_or_z_rows_ + control];
//     // z_control = z_target ^ z_control
//     z[bit * x_or_z_rows_ + control] ^= z[bit * x_or_z_rows_ + target];
//   }
// };

// template <class Precision> struct PhaseGateFunctor {

//   Kokkos::View<Precision *> r_;
//   Kokkos::View<Precision *> x_;
//   Kokkos::View<Precision *> z_;

//   size_t target_qubit_;
//   size_t num_x_or_z_rows_;

//   PhaseGateFunctor(Kokkos::View<Precision *> &r, Kokkos::View<Precision *> &x,
//                    Kokkos::View<Precision *> &z, std::size_t num_qubits,
//                    std::size_t target_qubit)
//       : r_(r), x_(x), z_(z), target_qubit_(target_qubit) {
//     num_x_or_z_rows_ = 2 * num_qubits + 1;
//   }

//   KOKKOS_INLINE_FUNCTION
//   void operator()(const std::size_t bit) const {
//     // Set r = r ^ (x_target*z_target)
//     r[bit] ^= x[bit * x_or_z_rows_ + target] * z[bit * x_or_z_rows_ + target];
//     //  z_target = z_target ^ x_target
//     z[bit * x_or_z_rows_ + target] ^= x[bit * x_or_z_rows_ + target];
//   }
// };

// template <class Precision> struct CPhaseGateFunctor {

//   Kokkos::View<Precision *> r_;
//   Kokkos::View<Precision *> x_;
//   Kokkos::View<Precision *> z_;

//   size_t target_qubit_;
//   size_t control_qubit_;
//   size_t num_x_or_z_rows_;

//   CPhaseGateFunctor(Kokkos::View<Precision *> &r, Kokkos::View<Precision *> &x,
//                     Kokkos::View<Precision *> &z, std::size_t num_qubits,
//                     std::size_t target_qubit, std::size_t control_qubit)
//       : r_(r), x_(x), z_(z), target_qubit_(target_qubit),
//         control_qubit_(control_qubit) {
//     num_x_or_z_rows_ = 2 * num_qubits + 1;
//   }

//   KOKKOS_INLINE_FUNCTION
//   void operator()(const std::size_t bit) const {
//     // r = r ^ (x_control * x_target *(z_target ^ z_control))
//     r ^= x[bit * x_or_z_rows_ + control] * x[bit * x_or_z_rows_ + target] *
//          (z[bit * x_or_z_rows_ + target] ^ z[bit * x_or_z_rows_ + control]);
//     // z_target = z_target ^ x_control
//     z[bit * x_or_z_rows_ + target] ^= x[bit * x_or_z_rows_ + control];
//     // z_control = z_control ^ x_target
//     z[bit * x_or_z_rows_ + control] ^= x[bit * x_or_z_rows_ + target];
//   }
// };

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
    r_(n,i) ^= x_(n,i,target_qubit_) & z_(n,i,target_qubit_);
    KE::swap(x_(n,i,target_qubit_),
             z_(n,i,target_qubit_));
  }
  
};

// template <class Precision> struct PauliXGateFunctor {

//   Kokkos::View<Precision *> r_;
//   Kokkos::View<Precision *> x_;
//   Kokkos::View<Precision *> z_;

//   size_t target_qubit_;
//   size_t num_x_or_z_rows_;

//   PauliXGateFunctor(Kokkos::View<Precision *> &r, Kokkos::View<Precision *> &x,
//                Kokkos::View<Precision *> &z, std::size_t num_qubits,
//                std::size_t target_qubit)
//       : r_(r), x_(x), z_(z), target_qubit_(target_qubit) {
//     num_x_or_z_rows_ = 2 * num_qubits + 1;
//   }

//   KOKKOS_INLINE_FUNCTION
//   void operator()(const std::size_t bit) const {
//     // Set r = r ^ z_target
//     r[bit] = r[bit] ^ z[bit * z_rows_ + target_qubit_];
//   }
// };

// template <class Precision> struct PauliZGateFunctor {

//   Kokkos::View<Precision *> r_;
//   Kokkos::View<Precision *> x_;
//   Kokkos::View<Precision *> z_;

//   size_t target_qubit_;
//   size_t num_x_or_z_rows_;

//   PauliZGateFunctor(Kokkos::View<Precision *> &r, Kokkos::View<Precision *> &x,
//                Kokkos::View<Precision *> &z, std::size_t num_qubits,
//                std::size_t target_qubit)
//       : r_(r), x_(x), z_(z), target_qubit_(target_qubit) {
//     num_x_or_z_rows_ = 2 * num_qubits + 1;
//   }

//   KOKKOS_INLINE_FUNCTION
//   void operator()(const std::size_t bit) const {
//     // Set r = r ^ x_target
//     r[bit] = r[bit] ^ x[bit * x_rows_ + target_qubit_];
//   }
// };
}; // namespace Plaquette

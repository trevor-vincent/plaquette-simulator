// #pragma once
// #include <Kokkos_Core.hpp>
// #include <Kokkos_StdAlgorithms.hpp>

// namespace Plaquette {

  //commutator_sign
  //first_nc_stab
  //other rows
  //multiply
  //deterministic

// };




namespace Plaquette {

template <class Precision> struct RowCopy {

  Kokkos::View<Precision *> x_;
  Kokkos::View<Precision *> z_;
  Kokkos::View<Precision *> r_;

  //Sets row i equal to row k
  size_t i_;
  size_t k_;

  RowCopy(Kokkos::View<Precision *> &x,
	  Kokkos::View<Precision *> &z,
	  Kokkos::View<Precision *> &r,
	  size_t i,
	  size_t k)
    : x_(x), z_(z), r_(r), i_(i), k_(k) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t j) const {
    x_(i_,j) = x_(k_,j);
    z_(i_,j) = z_(k_,j);
    r_(i_) = r_(k_)
  }
};

template <class Precision> struct RowMult {

  Kokkos::View<Precision *> x_;
  Kokkos::View<Precision *> z_;
  Kokkos::View<Precision *> r_;

  //Sets row i equal to row k
  size_t i_;
  size_t k_;

  RowCopy(Kokkos::View<Precision *> &x,
	  Kokkos::View<Precision *> &z,
	  Kokkos::View<Precision *> &r,
	  size_t i,
	  size_t k)
    : x_(x), z_(z), r_(r), i_(i), k_(k) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t j) const {
    x_(i_,j) = x_(k_,j);
    z_(i_,j) = z_(k_,j);
    r_(i_) = r_(k_)
  }
};
  



  
};


#pragma once

#include "CliffordState.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

template <class Precision> class InitTableauToZeroStateFunctor {
  private:
    Kokkos::View<Precision ***> x_;
    Kokkos::View<Precision ***> z_;
    Kokkos::View<Precision **> r_;
    size_t num_qubits_;

  public:
    InitTableauToZeroStateFunctor(Kokkos::View<Precision ***> x_,
                                  Kokkos::View<Precision ***> z_,
                                  Kokkos::View<Precision **> r_,
                                  size_t num_qubits)
        : x_(x_), z_(z_), r_(r_), num_qubits_(num_qubits) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int n, const int i, const int j) const {
        x_(n, i, j) = (i == j) ? 1 : 0;
        z_(n, i, j) = (i - num_qubits_ == j) ? 1 : 0;
        r_(n, i) = (i == 2 * num_qubits_) ? 1 : 0;
    }
};

template <class Precision> class CheckSumFunctor {
  private:
    Kokkos::View<Precision ***> x_;
    Kokkos::View<Precision ***> z_;
    Kokkos::View<Precision **> r_;
    size_t num_qubits_;

  public:
    CheckSumFunctor(Kokkos::View<Precision ***> x_,
                    Kokkos::View<Precision ***> z_,
                    Kokkos::View<Precision **> r_, size_t num_qubits)
        : x_(x_), z_(z_), r_(r_), num_qubits_(num_qubits) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int n, const int i, const int j,
                    Precision &lsum) const {
        lsum += x_(n, i, j);
        lsum += z_(n, i, j);
        lsum += (j == 0) ? r_(n, i) : 0;
    }
};

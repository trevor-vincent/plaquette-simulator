#pragma once
#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "GateFunctors.hpp"

#include <Kokkos_Core.hpp>

namespace Plaquette {

template <class Precision> class InitTableauToZeroState {
private:
  Kokkos::View<Precision ***> x_;
  Kokkos::View<Precision ***> z_;
  Kokkos::View<Precision **> r_;
  size_t num_qubits_;

public:
  InitTableauToZeroState(Kokkos::View<Precision ***> x_,
                         Kokkos::View<Precision ***> z_,
                         Kokkos::View<Precision **> r_, size_t num_qubits)
      : x_(x_), z_(z_), r_(r_), num_qubits_(num_qubits) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int n, const int i, const int j) const {
    x_(n, i, j) = (i == j) ? 1 : 0;
    z_(n, i, j) = (i == j + num_qubits_) ? 1 : 0;
    r_(n, i) = (j == 2 * num_qubits_) ? 1 : 0;
  }
};

template <class Precision> class BatchCliffordStateKokkos {

public:
  using KokkosExecSpace = Kokkos::DefaultExecutionSpace;
  using KokkosVector = Kokkos::View<Precision *>;
  using KokkosMat2D = Kokkos::View<Precision **>;
  using KokkosMat3D = Kokkos::View<Precision ***>;

  using UnmanagedHostVectorView =
      Kokkos::View<Precision *, Kokkos::HostSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using UnmanagedHostMat2DView =
      Kokkos::View<Precision **, Kokkos::HostSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using UnmanagedHostMat3DView =
      Kokkos::View<Precision **, Kokkos::HostSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  BatchCliffordStateKokkos() = delete;
  BatchCliffordStateKokkos(
      size_t num_qubits, size_t batch_size,
      const Kokkos::InitializationSettings &kokkos_args = {}) {

    num_qubits_ = num_qubits;
    batch_size_ = batch_size;
    tableau_width_ = 2 * num_qubits_ + 1;

    {
      const std::lock_guard<std::mutex> lock(init_mutex_);
      if (!Kokkos::is_initialized()) {
        Kokkos::initialize(kokkos_args);
      }
    }

    if (num_qubits_ > 0 and batch_size_ > 0) {
      x_ = std::make_unique<KokkosMat3D>("x_", batch_size_, tableau_width_,
                                         tableau_width_);

      z_ = std::make_unique<KokkosMat3D>("z_", batch_size_, tableau_width_,
                                         tableau_width_);

      r_ = std::make_unique<KokkosMat2D>("r_", batch_size_, tableau_width_);

      Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy(
          {0, 0, 0}, {batch_size_, tableau_width_, tableau_width_});
      Kokkos::parallel_for(policy, InitTableauToZeroState<Precision>(
                                       *x_, *z_, *r_, num_qubits_));
    }
  }

  inline void ApplyHadamardGate(size_t target_qubit) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(policy, BatchHadamardGateFunctor<Precision>(
                                     *x_, *z_, *r_, target_qubit));
  }

  inline void ApplyControlNotGate(size_t target_qubit, size_t control_qubit) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(
        policy, BatchControlNotGateFunctor<Precision>(*x_, *z_, *r_, target_qubit,
                                                   control_qubit));
  }

  inline void ApplyPhaseGate(size_t target_qubit) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(
        policy, BatchPhaseGateFunctor<Precision>(*x_, *z_, *r_, target_qubit));
  }

  ~BatchCliffordStateKokkos() {
    x_.reset();
    z_.reset();
    r_.reset();

    {
      const std::lock_guard<std::mutex> lock(init_mutex_);
      if (!is_exit_reg_) {
        is_exit_reg_ = true;
        std::atexit([]() {
          if (!Kokkos::is_finalized()) {
            Kokkos::finalize();
          }
        });
      }
    }
  }

private:
  size_t num_qubits_;
  size_t batch_size_;
  size_t tableau_width_;
  std::mutex init_mutex_;
  std::unique_ptr<KokkosMat3D> x_;
  std::unique_ptr<KokkosMat3D> z_;
  std::unique_ptr<KokkosMat2D> r_;
  inline static bool is_exit_reg_ = false;
};

}; // namespace Plaquette

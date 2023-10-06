#pragma once
#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "GateFunctors.hpp"
#include "MeasurementFunctors.hpp"

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
  // using UnmanagedHostMat2DView =
      // Kokkos::View<Precision **, Kokkos::HostSpace,
                   // Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  // using UnmanagedHostMat3DView =
      // Kokkos::View<Precision ***, Kokkos::HostSpace,
                   // Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

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



  /**
   * @brief Create a new state vector from data on the host.
   *
   * @param num_qubits Number of qubits
   */
  BatchCliffordStateKokkos(Precision * x, Precision * z,
			   Precision * r, size_t num_qubits,
                      const Kokkos::InitializationSettings &kokkos_args = {})
      : BatchCliffordStateKokkos(num_qubits, kokkos_args) {
    HostToDevice(x, z, r, num_qubits);
  }

  /**
   * @brief Copy constructor
   *
   * @param other Another state vector
   */
  BatchCliffordStateKokkos(const BatchCliffordStateKokkos &other,
                      const Kokkos::InitializationSettings &kokkos_args = {})
      : BatchCliffordStateKokkos(other.GetNumQubits(), kokkos_args) {

    this->DeviceToDevice(other.GetX(), other.GetZ(),
                         other.GetR());
  }

  [[nodiscard]] auto GetX() const -> KokkosMat3D & {
    return *x_;
  }
  [[nodiscard]] auto GetZ() const -> KokkosMat3D & {
    return *z_;
  }
  [[nodiscard]] auto GetR() const -> KokkosMat2D & {
    return *r_;
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
    Kokkos::parallel_for(policy,
                         BatchControlNotGateFunctor<Precision>(
                             *x_, *z_, *r_, target_qubit, control_qubit));
  }

  inline void ApplyControlPhaseGate(size_t target_qubit, size_t control_qubit) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(policy,
                         BatchControlPhaseGateFunctor<Precision>(
                             *x_, *z_, *r_, target_qubit, control_qubit));
  }

  inline void ApplyPhaseGate(size_t target_qubit) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(
        policy, BatchPhaseGateFunctor<Precision>(*x_, *z_, *r_, target_qubit));
  }

  inline void ApplyPauliXGate(size_t target_qubit) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(
        policy, BatchPauliXGateFunctor<Precision>(*x_, *z_, *r_, target_qubit));
  }

  inline void ApplyPauliZGate(size_t target_qubit) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(
        policy, BatchPauliZGateFunctor<Precision>(*x_, *z_, *r_, target_qubit));
  }

  /**
   * @brief Copy data from the host space to the device space.
   *
   */
  inline void HostToDevice(Precision *x, Precision *z, Precision *r,
                           size_t num_qubits) {
    Kokkos::deep_copy(*x_, UnmanagedHostMat3DView(x, x_->extent(0), x_->extent(1),
                                                  x_->extent(2)));

    Kokkos::deep_copy(*z_, UnmanagedHostMat3DView(z, z_->extent(0), z_->extent(1),
                                                  z_->extent(2)));

    Kokkos::deep_copy(*r_,
                      UnmanagedHostMat2DView(r, r_->extent(0), r_->extent(1)));
  }

  /**
   * @brief Copy data from the device space to the host space.
   *
   */
  inline void DeviceToHost(Precision *x, Precision *z, Precision *r) {
    Kokkos::deep_copy(
		      UnmanagedHostVectorView(x, x_->extent(0)*x_->extent(1)*x_->extent(2)),
        *x_);
    // Kokkos::deep_copy(
        // UnmanagedHostMat3DView(z, z_->extent(0), z_->extent(1), z_->extent(2)),
        // *z_);
    // Kokkos::deep_copy(UnmanagedHostMat2DView(r, r_->extent(0), r_->extent(1)),
                      // *r_);
  }

  inline void DeviceToDevice(KokkosMat3D x, KokkosMat3D z, KokkosMat2D r) {
    Kokkos::deep_copy(*x_, x);
    Kokkos::deep_copy(*z_, z);
    Kokkos::deep_copy(*r_, r);
  }

  size_t GetNumQubits() const { return num_qubits_; }

  void ResetCliffordState() {
    if (num_qubits_ > 0 and batch_size_ > 0) {
      Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy(
          {0, 0, 0}, {batch_size_, tableau_width_, tableau_width_});
      Kokkos::parallel_for(policy, InitTableauToZeroState<Precision>(
                                       *x_, *z_, *r_, num_qubits_));
    }
  }

  inline void MeasureQubit(size_t target_qubit, int seed = 1234567) {
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
    Kokkos::RangePolicy<KokkosExecSpace> policy(0, 2 * num_qubits_);
    KokkosVector measurement_results("measurement_result", batch_size_);
    KokkosVector measurement_determined("measurement_determined", batch_size_);

    Kokkos::parallel_for(policy, BatchMeasureQubitFunctor<Precision>(
                                     *x_, *z_, *r_, target_qubit));
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

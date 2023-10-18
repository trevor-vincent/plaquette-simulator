#pragma once
#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Error.hpp"
#include "GateFunctors.hpp"
#include "MeasurementFunctors.hpp"

#include <Kokkos_Core.hpp>

namespace Plaquette {

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
                  Kokkos::View<Precision ***> z_, Kokkos::View<Precision **> r_,
                  size_t num_qubits)
      : x_(x_), z_(z_), r_(r_), num_qubits_(num_qubits) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int n, const int i, const int j,
                  Precision &lsum) const {
    lsum += x_(n, i, j);
    lsum += z_(n, i, j);
    lsum += (j == 0) ? r_(n, i) : 0;
  }
};

template <class Precision> class BatchCliffordStateKokkos {

private:
  struct Measurement {
    size_t qubit_measured;
    std::vector<Precision> measurement_results_host;
    std::vector<Precision> measurement_determined_host;
    std::unique_ptr<Kokkos::View<Precision *>> measurement_results_device;
    std::unique_ptr<Kokkos::View<Precision *>> measurement_determined_device;

    void CopyToHost() {
      if (!measurement_results_host.empty()) {
        return;
      }
      auto batch_size = (*measurement_results_device).extent(0);

      measurement_results_host.resize(batch_size);
      measurement_determined_host.resize(batch_size);

      using UnmanagedHostVectorView =
          Kokkos::View<Precision *, Kokkos::HostSpace,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

      Kokkos::deep_copy(
          UnmanagedHostVectorView(measurement_results_host.data(), batch_size),
          *measurement_results_device);

      Kokkos::deep_copy(UnmanagedHostVectorView(
                            measurement_determined_host.data(), batch_size),
                        *measurement_determined_device);
    }
  };

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
      Kokkos::View<Precision ***, Kokkos::HostSpace,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  BatchCliffordStateKokkos() = delete;
  BatchCliffordStateKokkos(
      size_t num_qubits, size_t batch_size, int seed = -1,
      bool init_to_zero_state = true,
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
                                         num_qubits_);

      z_ = std::make_unique<KokkosMat3D>("z_", batch_size_, tableau_width_,
                                         num_qubits_);

      r_ = std::make_unique<KokkosMat2D>("r_", batch_size_, tableau_width_);

      if (init_to_zero_state) {
        Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy(
            {0, 0, 0}, {batch_size_, tableau_width_, num_qubits_});
        Kokkos::parallel_for(policy, InitTableauToZeroStateFunctor<Precision>(
                                         *x_, *z_, *r_, num_qubits_));
      }
    }

    if (seed >= 0) {
      seed_ = seed;
      rand_pool_ =
          std::make_unique<Kokkos::Random_XorShift64_Pool<KokkosExecSpace>>(
              seed);
    } else {
      seed_ = 213434232223;
      rand_pool_ =
          std::make_unique<Kokkos::Random_XorShift64_Pool<KokkosExecSpace>>(
              seed_);
    }
  }

  /**
   * @brief Create a new state vector from data on the host.
   *
   * @param num_qubits Number of qubits
   */
  BatchCliffordStateKokkos(
      std::vector<Precision> &x, std::vector<Precision> &z,
      std::vector<Precision> &r, size_t num_qubits, size_t batch_size,
      long int seed = -1,
      const Kokkos::InitializationSettings &kokkos_args = {})
      : BatchCliffordStateKokkos(num_qubits, batch_size, seed, false,
                                 kokkos_args) {
    HostToDevice(x, z, r);
  }

  /**
   * @brief Copy constructor
   *
   * @param other Another state vector
   */
  BatchCliffordStateKokkos(
      const BatchCliffordStateKokkos &other,
      const Kokkos::InitializationSettings &kokkos_args = {})
      : BatchCliffordStateKokkos(other.GetNumQubits(), other.GetBatchSize(),
                                 other.GetSeed(), false, kokkos_args) {

    this->DeviceToDevice(other.GetX(), other.GetZ(), other.GetR());
  }

  [[nodiscard]] auto GetX() const -> KokkosMat3D & { return *x_; }
  [[nodiscard]] auto GetZ() const -> KokkosMat3D & { return *z_; }
  [[nodiscard]] auto GetR() const -> KokkosMat2D & { return *r_; }

  inline auto CheckSum(int starting_batch_id = 0, int ending_batch_id = -1)
      -> Precision {
    Precision check_sum = 0;
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy(
        {starting_batch_id, 0, 0},
        {ending_batch_id < 0 ? batch_size_ : ending_batch_id, tableau_width_,
         num_qubits_});
    Kokkos::parallel_reduce(
        policy, CheckSumFunctor<Precision>(*x_, *z_, *r_, num_qubits_),
        check_sum);
    return check_sum;
  }

  inline void ApplyHadamardGate(size_t target_qubit) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(policy, BatchHadamardGateFunctor<Precision>(
                                     *x_, *z_, *r_, target_qubit));
  }

  inline void ApplyHadamardGate(size_t target_qubit,
                                float prob) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(policy,
                         BatchHadamardGateWithProbFunctor<Precision>(
								     *x_, *z_, *r_, target_qubit, rand_pool_, prob));
  }

  
  inline void ApplyHadamardGate(size_t target_qubit,
                                Kokkos::View<Precision *> apply_flag) {
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(policy,
                         BatchHadamardGateWithApplyFlagFunctor<Precision>(
                             *x_, *z_, *r_, target_qubit, apply_flag));
  }

  inline void ApplyControlNotGate(size_t control_qubit, size_t target_qubit) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(policy,
                         BatchControlNotGateFunctor<Precision>(
                             *x_, *z_, *r_, control_qubit, target_qubit));
  }

  inline void ApplyControlNotGate(size_t control_qubit, size_t target_qubit, float prob) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(policy,
                         BatchControlNotGateWithProbFunctor<Precision>(
								       *x_, *z_, *r_, control_qubit, target_qubit, rand_pool_, prob));
  }

  

  inline void ApplyControlPhaseGate(size_t control_qubit, size_t target_qubit) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(policy,
                         BatchControlPhaseGateFunctor<Precision>(
                             *x_, *z_, *r_, control_qubit, target_qubit));
  }


  inline void ApplyControlPhaseGate(size_t control_qubit, size_t target_qubit, float prob) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(policy,
                         BatchControlPhaseGateWithProbFunctor<Precision>(
									 *x_, *z_, *r_, control_qubit, target_qubit, rand_pool_, prob));
  }

  

  inline void ApplyPhaseGate(size_t target_qubit) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(
        policy, BatchPhaseGateFunctor<Precision>(*x_, *z_, *r_, target_qubit));
  }


  inline void ApplyPhaseGate(size_t target_qubit, float prob) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(
			 policy, BatchPhaseGateWithProbFunctor<Precision>(*x_, *z_, *r_, target_qubit, rand_pool_, prob));
  }
  
  inline void ApplyPhaseGate(size_t target_qubit,
                             Kokkos::View<Precision *> apply_flag) {
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(policy, BatchPhaseGateWithApplyFlagFunctor<Precision>(
                                     *x_, *z_, *r_, target_qubit, apply_flag));
  }

  inline void ApplyPauliXGate(size_t target_qubit) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(
        policy, BatchPauliXGateFunctor<Precision>(*x_, *z_, *r_, target_qubit));
  }

  inline void ApplyPauliXGate(size_t target_qubit, float prob) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(
        policy, BatchPauliXGateWithProbFunctor<Precision>(*x_, *z_, *r_, target_qubit));
  }
  
  inline void ApplyPauliZGate(size_t target_qubit) {
    
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(
        policy, BatchPauliZGateFunctor<Precision>(*x_, *z_, *r_, target_qubit));
  }

  inline void ApplyPauliZGate(size_t target_qubit, float prob) {
    
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(
			 policy, BatchPauliZGateFunctor<Precision>(*x_, *z_, *r_, target_qubit, rand_pool_, prob));
  }
  

  /**
   * @brief Copy data from the host space to the device space.
   *
   */
  inline void HostToDevice(std::vector<Precision> &x, std::vector<Precision> &z,
                           std::vector<Precision> &r) {

    Kokkos::View<Precision ***, Kokkos::HostSpace> x_host(
        "x_host", batch_size_, tableau_width_, num_qubits_);
    Kokkos::View<Precision ***, Kokkos::HostSpace> z_host(
        "z_host", batch_size_, tableau_width_, num_qubits_);
    Kokkos::View<Precision **, Kokkos::HostSpace> r_host("r_host", batch_size_,
                                                         tableau_width_);

    // for (size_t batch_id = 0; batch_id < batch_size_; ++batch_id) {
    //   for (size_t i = 0; i < tableau_width_; ++i) {
    //     r_host(batch_id, i) = r[i];
    //     for (size_t j = 0; j < num_qubits_; ++j) {
    //       x_host(batch_id, i, j) =
    //           x[batch_id * tableau_width_ * num_qubits_ + i * num_qubits_ + j];
    //       z_host(batch_id, i, j) =
    //           z[batch_id * tableau_width_ * num_qubits_ + i * num_qubits_ + j];
    //     }
    //   }
    // }

    // Kokkos::deep_copy(*x_, x_host);
    // Kokkos::deep_copy(*z_, z_host);
    // Kokkos::deep_copy(*r_, r_host);

    Kokkos::deep_copy(*x_,
                      UnmanagedHostMat3DView(x.data(), x_->extent(0),
                                             x_->extent(1), x_->extent(2)));

    Kokkos::deep_copy(*z_,
                      UnmanagedHostMat3DView(z.data(), z_->extent(0),
                                             z_->extent(1), z_->extent(2)));

    Kokkos::deep_copy(
        *r_, UnmanagedHostMat2DView(r.data(), r_->extent(0), r_->extent(1)));
  }

  auto DeviceToHost() {
    std::vector<Precision> x(x_->size());
    std::vector<Precision> z(z_->size());
    std::vector<Precision> r(r_->size());

    Kokkos::deep_copy(UnmanagedHostMat3DView(x.data(), x_->extent(0),
                                             x_->extent(1), x_->extent(2)),
                      *x_);
    Kokkos::deep_copy(UnmanagedHostMat3DView(z.data(), z_->extent(0),
                                             z_->extent(1), z_->extent(2)),
                      *z_);
    Kokkos::deep_copy(
        UnmanagedHostMat2DView(r.data(), r_->extent(0), r_->extent(1)), *r_);

    return std::make_tuple(x, z, r);
  }

  inline void DeviceToDevice(KokkosMat3D x, KokkosMat3D z, KokkosMat2D r) {
    Kokkos::deep_copy(*x_, x);
    Kokkos::deep_copy(*z_, z);
    Kokkos::deep_copy(*r_, r);
  }

  size_t GetNumQubits() const { return num_qubits_; }
  size_t GetBatchSize() const { return batch_size_; }
  long int GetSeed() const { return seed_; }

  void ResetCliffordState() {
    if (num_qubits_ > 0 and batch_size_ > 0) {
      Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy(
          {0, 0, 0}, {batch_size_, tableau_width_, tableau_width_});
      Kokkos::parallel_for(policy, InitTableauToZeroStateFunctor<Precision>(
                                       *x_, *z_, *r_, num_qubits_));
    }
  }

  size_t MeasureQubit(size_t target_qubit, float bias = 0.5,
                      std::optional<size_t> measurement_id = std::nullopt) {

    Kokkos::RangePolicy<KokkosExecSpace> policy(0, batch_size_);

    if (!measurement_id.has_value()) {
      measurements_.push_back(Measurement());
      measurement_id = measurements_.size() - 1;
      measurements_.back().qubit_measured = target_qubit;
      std::string qubit_name = "q" + std::to_string(target_qubit);
      std::string measurement_name = "m" + std::to_string(*measurement_id);
      std::string results_name =
          qubit_name + "_" + measurement_name + "_results";
      std::string determined_name =
          qubit_name + "_" + measurement_name + "_determined";
      measurements_.back().measurement_results_device =
          std::make_unique<KokkosVector>(results_name, batch_size_);
      measurements_.back().measurement_determined_device =
          std::make_unique<KokkosVector>(determined_name, batch_size_);
    } else {
      PLAQUETTE_ABORT_IF(measurement_id.value() >= measurements_.size(),
                         "Measurement id is out of range.");
    }

    Kokkos::parallel_for(
        policy,
        BatchMeasureFunctor<Precision>(
            *x_, *z_, *r_,
            *(measurements_[measurement_id.value()].measurement_results_device),
            *(measurements_[measurement_id.value()]
                  .measurement_determined_device),
            *rand_pool_, target_qubit, bias));

    return measurement_id.value();
  }

  auto GetMeasurement(size_t measurement_id, size_t batch_index) {
    std::optional<std::pair<int, int>> result;
    if (measurement_id < measurements_.size() and batch_index < batch_size_) {
      measurements_[measurement_id].CopyToHost();
      // std::cout << "Error after here" << std::endl;
      result = std::make_pair(
          measurements_[measurement_id].measurement_results_host[batch_index],
          measurements_[measurement_id]
              .measurement_determined_host[batch_index]);
    }
    return result;
  }

  auto &GetMeasurement(size_t measurement_id) {
    PLAQUETTE_ABORT_IF_NOT(measurement_id < measurements_.size(),
                           "Measurement id or batch index is out of range.");
    return measurements_[measurement_id];
  }

  ~BatchCliffordStateKokkos() {
    x_.reset();
    z_.reset();
    r_.reset();

    for (auto &measurement : measurements_) {
      measurement.measurement_results_device.reset();
      measurement.measurement_determined_device.reset();
    }

    rand_pool_.reset();

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
  long int seed_;
  std::mutex init_mutex_;
  std::unique_ptr<KokkosMat3D> x_;
  std::unique_ptr<KokkosMat3D> z_;
  std::unique_ptr<KokkosMat2D> r_;

  std::unique_ptr<Kokkos::Random_XorShift64_Pool<>> rand_pool_;
  std::vector<Measurement> measurements_;
  inline static bool is_exit_reg_ = false;
};

}; // namespace Plaquette

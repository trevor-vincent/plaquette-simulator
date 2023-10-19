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

/**
 * @class CliffordState
 * @brief This class represents a Clifford quantum state in the Stabilizer
 * tableau formalism, utilizing the Kokkos library for performance
 * optimizations.
 * @tparam Precision The precision of the data type, typically a int
 */
template <class Precision> class CliffordState {

private:
  /**
   * @struct Measurement
   * @brief Holds quantum measurement results for a specific qubit, both on host
   * and device memory.
   */
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

      Kokkos::deep_copy(Kokkos::View<Precision *, Kokkos::HostSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                            measurement_results_host.data(), batch_size),
                        *measurement_results_device);

      Kokkos::deep_copy(Kokkos::View<Precision *, Kokkos::HostSpace,
                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                            measurement_determined_host.data(), batch_size),
                        *measurement_determined_device);
    }
  };

  // Member variables
  size_t num_qubits_;     ///< Number of qubits in the state.
  size_t batch_size_;     ///< Size of the batch (number of simulation shots).
  size_t tableau_width_;  ///< Width of the tableau, equal to 2*num_qubits + 1.
  long int seed_;         ///< Seed for random number generation.
  std::mutex init_mutex_; ///< Mutex for thread-safe initialization of Kokkos.
  std::unique_ptr<Kokkos::View<Precision ***>>
      x_; ///< Representation of the X matrix.
  std::unique_ptr<Kokkos::View<Precision ***>>
      z_; ///< Representation of the Z matrix.
  std::unique_ptr<Kokkos::View<Precision **>>
      r_; ///< Representation of the R matrix.

  std::unique_ptr<Kokkos::Random_XorShift64_Pool<>>
      rand_pool_;                         ///< Pool of random number generators.
  std::vector<Measurement> measurements_; ///< Collection of measurements.
  inline static bool is_exit_reg_ =
      false; ///< Flag indicating whether to exit the register.

public:
  /**
   * @brief Deleted default constructor. Objects of CliffordState must be
   * constructed with specific parameters.
   */
  CliffordState() = delete;

  /**
   * @brief Constructs a CliffordState object with specified parameters.
   *
   * @param num_qubits Number of qubits in the state.
   * @param batch_size Size of the batch for operations.
   * @param seed Seed for random number generation, defaults to -1.
   * @param init_to_zero_state Whether to initialize the state vector to the
   * zero state, defaults to true.
   * @param kokkos_args Arguments for Kokkos initialization, defaults to empty
   * settings.
   */
  CliffordState(size_t num_qubits, size_t batch_size, int seed = -1,
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
      x_ = std::make_unique<Kokkos::View<Precision ***>>(
          "x_", batch_size_, tableau_width_, num_qubits_);

      z_ = std::make_unique<Kokkos::View<Precision ***>>(
          "z_", batch_size_, tableau_width_, num_qubits_);

      r_ = std::make_unique<Kokkos::View<Precision **>>("r_", batch_size_,
                                                        tableau_width_);

      if (init_to_zero_state) {
        Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy(
            {0, 0, 0}, {batch_size_, tableau_width_, num_qubits_});
        Kokkos::parallel_for(policy, InitTableauToZeroStateFunctor<Precision>(
                                         *x_, *z_, *r_, num_qubits_));
      }
    }

    if (seed >= 0) {
      seed_ = seed;
      rand_pool_ = std::make_unique<
          Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>>(seed);
    } else {
      seed_ = 213434232223;
      rand_pool_ = std::make_unique<
          Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>>(seed_);
    }
  }

  /**
   * @brief Constructs a CliffordState object from existing state data.
   *
   * @param x State data for X matrix.
   * @param z State data for Z matrix.
   * @param r State data for R matrix.
   * @param num_qubits Number of qubits in the state.
   * @param batch_size Size of the batch for operations.
   * @param seed Seed for random number generation, defaults to -1.
   * @param kokkos_args Arguments for Kokkos initialization, defaults to empty
   * settings.
   */
  CliffordState(std::vector<Precision> &x, std::vector<Precision> &z,
                std::vector<Precision> &r, size_t num_qubits, size_t batch_size,
                long int seed = -1,
                const Kokkos::InitializationSettings &kokkos_args = {})
      : CliffordState(num_qubits, batch_size, seed, false, kokkos_args) {
    HostToDevice(x, z, r);
  }

  /**
   * @brief Copy constructor.
   *
   * @param other The other CliffordState object to copy from.
   * @param kokkos_args Arguments for Kokkos initialization, defaults to empty
   * settings.
   */
  CliffordState(const CliffordState &other,
                const Kokkos::InitializationSettings &kokkos_args = {})
      : CliffordState(other.GetNumQubits(), other.GetBatchSize(),
                      other.GetSeed(), false, kokkos_args) {

    this->DeviceToDevice(other.GetX(), other.GetZ(), other.GetR());
  }

  /**
   * @brief Retrieves the X matrix of the state.
   * @return Reference to the X matrix.
   */
  [[nodiscard]] auto GetX() const -> Kokkos::View<Precision ***> & {
    return *x_;
  }

  /**
   * @brief Retrieves the Z matrix of the state.
   * @return Reference to the Z matrix.
   */
  [[nodiscard]] auto GetZ() const -> Kokkos::View<Precision ***> & {
    return *z_;
  }

  /**
   * @brief Retrieves the R matrix of the state.
   * @return Reference to the R matrix.
   */
  [[nodiscard]] auto GetR() const -> Kokkos::View<Precision **> & {
    return *r_;
  }

  /**
   * @brief Calculates a checksum of the state for verification purposes.
   *
   * @param starting_batch_id The starting batch ID for the checksum
   * calculation, defaults to 0.
   * @param ending_batch_id The ending batch ID for the checksum calculation,
   * defaults to batch_size.
   * @return The calculated checksum.
   */
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
  /**
   * @brief Applies the Hadamard gate to a target qubit.
   *
   * @param target_qubit Index of the target qubit.
   */
  inline void ApplyHadamardGate(size_t target_qubit) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(policy, BatchHadamardGateFunctor<Precision>(
                                     *x_, *z_, *r_, target_qubit));
  }
  /**
   * @brief Applies the Hadamard gate to a target qubit with a certain
   * probability.
   *
   * @param target_qubit Index of the target qubit.
   * @param prob Probability with which the gate is applied.
   */
  inline void ApplyHadamardGateWithProb(size_t target_qubit, float prob) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(policy,
                         BatchHadamardGateWithProbFunctor<Precision>(
                             *x_, *z_, *r_, target_qubit, *rand_pool_, prob));
  }

  /**
   * @brief Applies the Hadamard gate to a target qubit if a certain flag is
   * set.
   *
   * @param target_qubit Index of the target qubit.
   * @param apply_flag View containing flags that determine whether the gate
   * should be applied.
   */
  inline void
  ApplyHadamardGateWithApplyFlag(size_t target_qubit,
                                 Kokkos::View<Precision *> apply_flag) {
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(policy,
                         BatchHadamardGateWithApplyFlagFunctor<Precision>(
                             *x_, *z_, *r_, target_qubit, apply_flag));
  }
  /**
   * @brief Applies the Controlled-NOT (CNOT) gate between control and target
   * qubits.
   *
   * @param control_qubit Index of the control qubit.
   * @param target_qubit Index of the target qubit.
   */
  inline void ApplyControlNotGate(size_t control_qubit, size_t target_qubit) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(policy,
                         BatchControlNotGateFunctor<Precision>(
                             *x_, *z_, *r_, control_qubit, target_qubit));
  }

  /**
   * @brief Applies the Controlled-NOT (CNOT) gate between control and target
   * qubits with a certain probability.
   *
   * @param control_qubit Index of the control qubit.
   * @param target_qubit Index of the target qubit.
   * @param prob Probability with which the gate is applied.
   */
  inline void ApplyControlNotGateWithProb(size_t control_qubit,
                                          size_t target_qubit, float prob) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(policy, BatchControlNotGateWithProbFunctor<Precision>(
                                     *x_, *z_, *r_, control_qubit, target_qubit,
                                     *rand_pool_, prob));
  }

  /**
   * @brief Applies the Controlled-Phase (CPhase) gate between control and
   * target qubits.
   *
   * @param control_qubit Index of the control qubit.
   * @param target_qubit Index of the target qubit.
   */
  inline void ApplyControlPhaseGate(size_t control_qubit, size_t target_qubit) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(policy,
                         BatchControlPhaseGateFunctor<Precision>(
                             *x_, *z_, *r_, control_qubit, target_qubit));
  }
  /**
   * @brief Applies the Controlled-Phase (CPhase) gate between control and
   * target qubits with a certain probability.
   *
   * @param control_qubit Index of the control qubit.
   * @param target_qubit Index of the target qubit.
   * @param prob Probability with which the gate is applied.
   */
  inline void ApplyControlPhaseGateWithProb(size_t control_qubit,
                                            size_t target_qubit, float prob) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(
        policy,
        BatchControlPhaseGateWithProbFunctor<Precision>(
            *x_, *z_, *r_, control_qubit, target_qubit, *rand_pool_, prob));
  }
  /**
   * @brief Applies the Phase gate to a target qubit.
   *
   * @param target_qubit Index of the target qubit.
   */
  inline void ApplyPhaseGate(size_t target_qubit) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(
        policy, BatchPhaseGateFunctor<Precision>(*x_, *z_, *r_, target_qubit));
  }

  /**
   * @brief Applies the Phase gate to a target qubit with a certain probability.
   *
   * @param target_qubit Index of the target qubit.
   * @param prob Probability with which the gate is applied.
   */
  inline void ApplyPhaseGateWithProb(size_t target_qubit, float prob) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(policy,
                         BatchPhaseGateWithProbFunctor<Precision>(
                             *x_, *z_, *r_, target_qubit, *rand_pool_, prob));
  }
  /**
   * @brief Applies the Phase gate to a target qubit if a certain flag is set.
   *
   * @param target_qubit Index of the target qubit.
   * @param apply_flag View containing flags that determine whether the gate
   * should be applied.
   */
  inline void
  ApplyPhaseGateWithApplyFlag(size_t target_qubit,
                              Kokkos::View<Precision *> apply_flag) {
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(policy, BatchPhaseGateWithApplyFlagFunctor<Precision>(
                                     *x_, *z_, *r_, target_qubit, apply_flag));
  }
  /**
   * @brief Applies the Pauli-X gate to a target qubit.
   *
   * @param target_qubit Index of the target qubit.
   */
  inline void ApplyPauliXGate(size_t target_qubit) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(
        policy, BatchPauliXGateFunctor<Precision>(*x_, *z_, *r_, target_qubit));
  }
  /**
   * @brief Applies the Pauli-X gate to a target qubit with a certain
   * probability.
   *
   * @param target_qubit Index of the target qubit.
   * @param prob Probability with which the gate is applied.
   */
  inline void ApplyPauliXGateWithProb(size_t target_qubit, float prob) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(policy,
                         BatchPauliXGateWithProbFunctor<Precision>(
                             *x_, *z_, *r_, target_qubit, *rand_pool_, prob));
  }
  /**
   * @brief Applies the Pauli-Z gate to a target qubit.
   *
   * @param target_qubit Index of the target qubit.
   */
  inline void ApplyPauliZGate(size_t target_qubit) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(
        policy, BatchPauliZGateFunctor<Precision>(*x_, *z_, *r_, target_qubit));
  }
  /**
   * @brief Applies the Pauli-Z gate to a target qubit with a certain
   * probability.
   *
   * @param target_qubit Index of the target qubit.
   * @param prob Probability with which the gate is applied.
   */
  inline void ApplyPauliZGateWithProb(size_t target_qubit, float prob) {

    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy(
        {0, 0}, {batch_size_, tableau_width_});
    Kokkos::parallel_for(policy,
                         BatchPauliZGateWithProbFunctor<Precision>(
                             *x_, *z_, *r_, target_qubit, *rand_pool_, prob));
  }

  /**
   * @brief Copies quantum state data from host to device memory.
   *
   * @param x Host-side data representing the X matrix.
   * @param z Host-side data representing the Z matrix.
   * @param r Host-side data representing the R matrix.
   */
  inline void HostToDevice(std::vector<Precision> &x, std::vector<Precision> &z,
                           std::vector<Precision> &r) {

    Kokkos::View<Precision ***, Kokkos::HostSpace> x_host(
        "x_host", batch_size_, tableau_width_, num_qubits_);
    Kokkos::View<Precision ***, Kokkos::HostSpace> z_host(
        "z_host", batch_size_, tableau_width_, num_qubits_);
    Kokkos::View<Precision **, Kokkos::HostSpace> r_host("r_host", batch_size_,
                                                         tableau_width_);

    Kokkos::deep_copy(
        *x_, Kokkos::View<Precision ***, Kokkos::HostSpace,
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                 x.data(), x_->extent(0), x_->extent(1), x_->extent(2)));

    Kokkos::deep_copy(
        *z_, Kokkos::View<Precision ***, Kokkos::HostSpace,
                          Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                 z.data(), z_->extent(0), z_->extent(1), z_->extent(2)));

    Kokkos::deep_copy(*r_,
                      Kokkos::View<Precision **, Kokkos::HostSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                          r.data(), r_->extent(0), r_->extent(1)));
  }

  /**
   * @brief Retrieves quantum state data from device to host memory.
   *
   * @return Tuple containing X, Z, and R matrices data.
   */
  auto DeviceToHost() {
    std::vector<Precision> x(x_->size());
    std::vector<Precision> z(z_->size());
    std::vector<Precision> r(r_->size());

    Kokkos::deep_copy(
        Kokkos::View<Precision ***, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            x.data(), x_->extent(0), x_->extent(1), x_->extent(2)),
        *x_);
    Kokkos::deep_copy(
        Kokkos::View<Precision ***, Kokkos::HostSpace,
                     Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
            z.data(), z_->extent(0), z_->extent(1), z_->extent(2)),
        *z_);
    Kokkos::deep_copy(Kokkos::View<Precision **, Kokkos::HostSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                          r.data(), r_->extent(0), r_->extent(1)),
                      *r_);

    return std::make_tuple(x, z, r);
  }
  /**
   * @brief Copies quantum state data from one device memory location to
   * another.
   *
   * @param x Device-side data representing the X matrix.
   * @param z Device-side data representing the Z matrix.
   * @param r Device-side data representing the R matrix.
   */
  inline void DeviceToDevice(Kokkos::View<Precision ***> x,
                             Kokkos::View<Precision ***> z,
                             Kokkos::View<Precision **> r) {
    Kokkos::deep_copy(*x_, x);
    Kokkos::deep_copy(*z_, z);
    Kokkos::deep_copy(*r_, r);
  }
  /**
   * @brief Retrieves the name of the Kokkos backend being used.
   *
   * @return String representing the name of the Kokkos backend.
   */
  std::string GetKokkosBackend() const {
    return Kokkos::DefaultExecutionSpace::name();
  }

  /**
   * @brief Retrieves the number of qubits represented in the quantum state.
   *
   * @return Number of qubits.
   */
  size_t GetNumQubits() const { return num_qubits_; }

  /**
   * @brief Retrieves the batch size (number of simulation shots).
   *
   * @return Batch size (number of simulation shots).
   */
  size_t GetBatchSize() const { return batch_size_; }

  /**
   * @brief Retrieves the seed used for random number generation.
   *
   * @return Seed value.
   */
  long int GetSeed() const { return seed_; }

  /**
   * @brief Resets the quantum state to its initial state, characterized by all
   * qubits in the |0> state.
   */
  void ResetCliffordState() {
    if (num_qubits_ > 0 and batch_size_ > 0) {
      Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy(
          {0, 0, 0}, {batch_size_, tableau_width_, tableau_width_});
      Kokkos::parallel_for(policy, InitTableauToZeroStateFunctor<Precision>(
                                       *x_, *z_, *r_, num_qubits_));
    }
  }

  /**
   * @brief Measures the state of a target qubit with an optional bias and
   * stores the result with an optional measurement ID.
   *
   * @param target_qubit Index of the qubit to be measured.
   * @param bias Optional bias for the measurement, defaults to 0.5.
   * @param measurement_id Optional identifier for the measurement, if absent a
   * new measurement is created.
   * @return The measurement ID associated with this measurement operation.
   */
  size_t MeasureQubit(size_t target_qubit, float bias = 0.5,
                      std::optional<size_t> measurement_id = std::nullopt) {

    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace> policy(0, batch_size_);

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
          std::make_unique<Kokkos::View<Precision *>>(results_name,
                                                      batch_size_);
      measurements_.back().measurement_determined_device =
          std::make_unique<Kokkos::View<Precision *>>(determined_name,
                                                      batch_size_);
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
  /**
   * @brief Retrieves the result of a specific measurement operation for a
   * particular batch.
   *
   * @param measurement_id ID of the measurement to retrieve.
   * @param batch_index Index within the batch for the desired result.
   * @return A pair containing the measurement result and a flag indicating if
   * the qubit state was determined, or nullopt if the parameters are out of
   * range.
   */
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
  /**
   * @brief Retrieves a reference to a specific measurement operation.
   *
   * @param measurement_id ID of the measurement to retrieve.
   * @return A reference to the specified measurement.
   */
  auto &GetMeasurement(size_t measurement_id) {
    PLAQUETTE_ABORT_IF_NOT(measurement_id < measurements_.size(),
                           "Measurement id or batch index is out of range.");
    return measurements_[measurement_id];
  }

  /**
   * @brief Destructor for the CliffordState class, ensuring proper release of
   * resources.
   */
  ~CliffordState() {
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
};

}; // namespace Plaquette

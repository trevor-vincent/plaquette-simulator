#pragma once
#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Kokkos_Core.hpp>

/**
 * @brief Kokkos functor for initializing the state vector to the \f$\ket{0}\f$
 * state
 *
 * @tparam Precision Floating point precision of underlying statevector data
 */
template <typename Precision> struct InitView {
  Kokkos::View<Precision *> a;
  size_t tableau_size;
  InitView(Kokkos::View<Precision *> a_, size_t tableau_size)
      : a(a_), tableau_size_(tableau_size) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t i) const { a(i * tableau_size_ + i) = 1.0; }
};

template <typename Precision> struct initZerosFunctor {
  Kokkos::View<Kokkos::complex<Precision> *> a;
  size_t tableau_size_;

  initZerosFunctor(Kokkos::View<Precision *> a_, size_t tableau_size)
      : a(a_), tableau_size_(tableau_size) {}
  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t i) const { a(i) = 0; }
};

template <class Precision> class CliffordStateKokkos {

public:
  using KokkosExecSpace = Kokkos::DefaultExecutionSpace;
  using KokkosVector = Kokkos::View<Precision *>;

  CliffordStateKokkos() = delete;
  CliffordStateKokkos(size_t num_qubits,
                      const Kokkos::InitArguments &kokkos_args = {}) {

    num_qubits_ = num_qubits;
    tableau_size_ = (2 * num_qubits_ + 1) * (2 * num_qubits_ + 1);

    {
      const std::lock_guard<std::mutex> lock(init_mutex_);
      if (!Kokkos::is_initialized()) {
        Kokkos::initialize(kokkos_args);
      }
    }

    if (num_qubits > 0) {
      data_ = std::make_unique<KokkosVector>("data_", Util::exp2(num_qubits));
      Kokkos::parallel_for(tableau_size_, InitView(*data_));
    }
  }

  /**
   * @brief Create a new state vector from data on the host.
   *
   * @param num_qubits Number of qubits
   */
  CliffordStateKokkos(Precision *hostdata_, size_t length,
                      const Kokkos::InitArguments &kokkos_args = {})
      : CliffordStateKokkos(Util::log2(length), kokkos_args) {
    HostToDevice(hostdata_, length);
  }

  /**
   * @brief Copy constructor
   *
   * @param other Another state vector
   */
  CliffordStateKokkos(const CliffordStateKokkos &other,
                      const Kokkos::InitArguments &kokkos_args = {})
      : CliffordStateKokkos(other.getNumQubits(), kokkos_args) {
    this->DeviceToDevice(other.getData());
  }

  /**
   * @brief Destructor for CliffordStateKokkos class
   *
   * @param other Another state vector
   */
  ~CliffordStateKokkos() {
    data_.reset();
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

  /**
   * @brief Reset the data back to the \f$\ket{0}\f$ state.
   *
   * @param num_qubits Number of qubits
   */
  void ResetCliffordState() {
    if (tableau_size_ > 0) {
      Kokkos::parallel_for(tableau_size, InitView(*data_, tableau_size_));
    }
  }

  /**
   * @brief Init zeros for the state-vector on device.
   */
  void InitZeros() {
    Kokkos::parallel_for(getLength(), initZerosFunctor(getData()));
  }

  /**
   * @brief Get the number of qubits of the state vector.
   *
   * @return The number of qubits of the state vector
   */
  size_t GetNumQubits() const { return num_qubits_; }

  /**
   * @brief Get the size of the state vector
   *
   * @return The size of the state vector
   */
  size_t GetLength() const { return tableau_size; }

  void UpdateData(const CliffordStateKokkos<Precision> &other) {
    Kokkos::deep_copy(*data_, other.getData());
  }

  /**
   * @brief Get the Kokkos data of the state vector.
   *
   * @return The pointer to the data of state vector
   */
  [[nodiscard]] auto GetData() const -> KokkosVector & { return *data_; }

  /**
   * @brief Get the Kokkos data of the state vector
   *
   * @return The pointer to the data of state vector
   */
  [[nodiscard]] auto GetData() -> KokkosVector & { return *data_; }

  /**
   * @brief Copy data from the host space to the device space.
   *
   */
  inline void HostToDevice(Precision *sv, size_t length) {
    Kokkos::deep_copy(*data_, UnmanagedComplexHostView(sv, length));
  }

  /**
   * @brief Copy data from the device space to the host space.
   *
   */
  inline void DeviceToHost(Precision *sv, size_t length) {
    Kokkos::deep_copy(UnmanagedComplexHostView(sv, length), *data_);
  }

  /**
   * @brief Copy data from the device space to the device space.
   *
   */
  inline void DeviceToDevice(KokkosVector vector_to_copy) {
    Kokkos::deep_copy(*data_, vector_to_copy);
  }

  inline void ApplyHadamardGate(size_t target_qubit) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<KokkosExecSpace>(0, 2 * num_qubits_ + 1),
        HadmardGateFunctor<Precision>(*x_, *z_, *r_, num_qubits_,
                                      target_qubit));
  }

  inline void ApplyPauliXGate(size_t target_qubit) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<KokkosExecSpace>(0, 2 * num_qubits_ + 1),
        PauliXGateFunctor<Precision>(*x_, *z_, *r_, num_qubits_, target_qubit));
  }

  inline void ApplyPauliZGate(size_t target_qubit) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<KokkosExecSpace>(0, 2 * num_qubits_ + 1),
        PauliXGateFunctor<Precision>(*x_, *z_, *r_, num_qubits_, target_qubit));
  }

  inline void ApplyPauliYGate(size_t target_qubit) {
    ApplyPauliXGate(target_qubit);
    ApplyPauliZGate(target_qubit);
  }

  inline void ApplyPhaseGate(size_t target_qubit) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<KokkosExecSpace>(0, 2 * num_qubits_ + 1),
        PhaseGateFunctor<Precision>(*x_, *z_, *r_, num_qubits_, target_qubit));
  }

  inline void ApplyCPhaseGate(size_t target_qubit, size_t control_qubit) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<KokkosExecSpace>(0, 2 * num_qubits_ + 1),
        CPhaseGateFunctor<Precision>(*x_, *z_, *r_, num_qubits_, target_qubit,
                                     control_qubit));
  }

  inline void ApplyCNotGate(size_t target_qubit, size_t control_qubit) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<KokkosExecSpace>(0, 2 * num_qubits_ + 1),
        CNotGateFunctor<Precision>(*x_, *z_, *r_, num_qubits_, target_qubit,
                                   control_qubit));
  }

private:
  size_t num_qubits_;
  size_t tableau_size_;
  std::mutex init_mutex_;
  std::unique_ptr<KokkosVector> data_;
  inline static bool is_exit_reg_ = false;
};

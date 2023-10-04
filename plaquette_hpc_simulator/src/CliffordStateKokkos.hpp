#pragma once
#include <cstdlib>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Kokkos_Core.hpp>

namespace Plaquette {

/**
 * @brief Kokkos functor for initializing the state vector to the \f$\ket{0}\f$
 * state
 *
 * @tparam Precision Floating point precision of underlying statevector data
 */
template <typename Precision> struct InitTableauToZeroState {
  Kokkos::View<Precision *> x_;
  Kokkos::View<Precision *> z_;
  Kokkos::View<Precision *> r_;
  size_t num_qubits_;

  InitTableauToZeroState(Kokkos::View<Precision *> x,
                         Kokkos::View<Precision *> z,
                         Kokkos::View<Precision *> r, size_t num_qubits)
      : x_(x), z_(z), r_(r), num_qubits_(num_qubits) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const std::size_t i) const {
    z_[(num_qubits_ + i) * num_qubits_ + i] = 1.0;
    x_[i * num_qubits_ + i] = 1.0;
  }
};

// template <typename Precision> struct initZerosFunctor {
//   Kokkos::View<Kokkos::complex<Precision> *> a;
//   size_t tableau_size_;

//   initZerosFunctor(Kokkos::View<Precision *> a_, size_t tableau_size)
//       : a(a_), tableau_size_(tableau_size) {}
//   KOKKOS_INLINE_FUNCTION
//   void operator()(const std::size_t i) const { a(i) = 0; }
// };

template <class Precision> class CliffordStateKokkos {

public:
  using KokkosExecSpace = Kokkos::DefaultExecutionSpace;
  using KokkosVector = Kokkos::View<Precision *>;

    using UnmanagedHostView =
      Kokkos::View<Precision *, Kokkos::HostSpace,
		   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  
  CliffordStateKokkos() = delete;
  CliffordStateKokkos(size_t num_qubits,
                      const Kokkos::InitArguments &kokkos_args = {}) {

    num_qubits_ = num_qubits;

    {
      const std::lock_guard<std::mutex> lock(init_mutex_);
      if (!Kokkos::is_initialized()) {
        Kokkos::initialize(kokkos_args);
      }
    }

    if (num_qubits > 0) {
      tableau_x_data_ = std::make_unique<KokkosVector>(
          "tableau_x_", 2 * num_qubits_ * num_qubits_);
      tableau_z_data_ = std::make_unique<KokkosVector>(
          "tableau_z_", 2 * num_qubits_ * num_qubits_);
      tableau_r_data_ =
          std::make_unique<KokkosVector>("tableau_r_", 2 * num_qubits_);

      Kokkos::parallel_for(num_qubits_, InitTableauToZeroState(
                                            *tableau_x_data_, *tableau_z_data_,
                                            *tableau_r_data_, num_qubits_));
    }
  }

  /**
   * @brief Create a new state vector from data on the host.
   *
   * @param num_qubits Number of qubits
   */
  CliffordStateKokkos(Precision *tableau_x_data, Precision *tableau_z_data,
                      Precision *tableau_r_data, size_t num_qubits,
                      const Kokkos::InitArguments &kokkos_args = {})
      : CliffordStateKokkos(num_qubits, kokkos_args) {
    HostToDevice(tableau_x_data, tableau_z_data, tableau_r_data, num_qubits);
  }

  /**
   * @brief Copy constructor
   *
   * @param other Another state vector
   */
  CliffordStateKokkos(const CliffordStateKokkos &other,
                      const Kokkos::InitArguments &kokkos_args = {})
      : CliffordStateKokkos(other.GetNumQubits(), kokkos_args) {

    this->DeviceToDevice(other.GetXTableauData(), other.GetZTableauData(),
                         other.GetSignTableauData());
  }

  /**
   * @brief Destructor for CliffordStateKokkos class
   *
   * @param other Another state vector
   */
  ~CliffordStateKokkos() {
    tableau_x_data_.reset();
    tableau_z_data_.reset();
    tableau_r_data_.reset();

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
    if (num_qubits_ > 0) {
      Kokkos::parallel_for(num_qubits_, InitTableauToZeroState(
                                            *tableau_x_data_, *tableau_z_data_,
                                            *tableau_r_data_, num_qubits_));
    }
  }

  /**
   * @brief Get the number of qubits of the state vector.
   *
   * @return The number of qubits of the state vector
   */
  size_t GetNumQubits() const { return num_qubits_; }

  void UpdateData(const CliffordStateKokkos<Precision> &other) {
    Kokkos::deep_copy(*tableau_x_data_, other.GetXTableauData());
    Kokkos::deep_copy(*tableau_z_data_, other.GetZTableauData());
    Kokkos::deep_copy(*tableau_r_data_, other.GetSignTableauData());
  }

  [[nodiscard]] auto GetXTableauData() const -> KokkosVector & {
    return *tableau_x_data_;
  }
  [[nodiscard]] auto GetZTableauData() const -> KokkosVector & {
    return *tableau_z_data_;
  }
  [[nodiscard]] auto GetSignTableauData() const -> KokkosVector & {
    return *tableau_r_data_;
  }

  /**
   * @brief Copy data from the host space to the device space.
   *
   */
  inline void HostToDevice(Precision *tableau_x_data, Precision *tableau_z_data,
                           Precision *tableau_r_data, size_t num_qubits) {
    Kokkos::deep_copy(
        *tableau_x_data_,
        UnmanagedHostView(tableau_x_data, 2 * num_qubits * num_qubits));
    Kokkos::deep_copy(
        *tableau_z_data_,
        UnmanagedHostView(tableau_z_data, 2 * num_qubits * num_qubits));
    Kokkos::deep_copy(
        *tableau_r_data_,
        UnmanagedHostView(tableau_r_data, 2 * num_qubits));
  }

  /**
   * @brief Copy data from the device space to the host space.
   *
   */
  inline void DeviceToHost(Precision *tableau_x_data, Precision *tableau_z_data,
                           Precision *tableau_r_data) {
    Kokkos::deep_copy(
        UnmanagedHostView(tableau_x_data, 2 * num_qubits_ * num_qubits_),
        *tableau_x_data_);
    Kokkos::deep_copy(
        UnmanagedHostView(tableau_z_data, 2 * num_qubits_ * num_qubits_),
        *tableau_z_data_);
    Kokkos::deep_copy(
        UnmanagedHostView(tableau_r_data, 2 * num_qubits_),
        *tableau_r_data_);
  }

  /**
   * @brief Copy data from the device space to the device space.
   *
   */
  inline void DeviceToDevice(KokkosVector tableau_x_data,
                             KokkosVector tableau_z_data,
                             KokkosVector tableau_r_data) {
    Kokkos::deep_copy(*tableau_x_data_, tableau_x_data);
    Kokkos::deep_copy(*tableau_z_data_, tableau_z_data);
    Kokkos::deep_copy(*tableau_r_data_, tableau_r_data);
  }

  inline void ApplyHadamardGate(size_t target_qubit) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<KokkosExecSpace>(0, 2 * num_qubits_),
        HadmardGateFunctor<Precision>(*tableau_x_data_, *tableau_z_data_,
                                      *tableau_r_data_, num_qubits_,
                                      target_qubit));
  }

  inline void ApplyPauliXGate(size_t target_qubit) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<KokkosExecSpace>(0, 2 * num_qubits_),
        PauliXGateFunctor<Precision>(*tableau_x_data_, *tableau_z_data_,
                                     *tableau_r_data_, num_qubits_,
                                     target_qubit));
  }

  inline void ApplyPauliZGate(size_t target_qubit) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<KokkosExecSpace>(0, 2 * num_qubits_),
        PauliXGateFunctor<Precision>(*tableau_x_data_, *tableau_z_data_,
                                     *tableau_r_data_, num_qubits_,
                                     target_qubit));
  }

  inline void ApplyPauliYGate(size_t target_qubit) {
    ApplyPauliXGate(target_qubit);
    ApplyPauliZGate(target_qubit);
  }

  inline void ApplyPhaseGate(size_t target_qubit) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<KokkosExecSpace>(0, 2 * num_qubits_),
        PhaseGateFunctor<Precision>(*tableau_x_data_, *tableau_z_data_,
                                    *tableau_r_data_, num_qubits_,
                                    target_qubit));
  }

  inline void ApplyCPhaseGate(size_t target_qubit, size_t control_qubit) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<KokkosExecSpace>(0, 2 * num_qubits_),
        CPhaseGateFunctor<Precision>(*tableau_x_data_, *tableau_z_data_,
                                     *tableau_r_data_, num_qubits_,
                                     target_qubit, control_qubit));
  }

  inline void ApplyCNotGate(size_t target_qubit, size_t control_qubit) {
    Kokkos::parallel_for(
        Kokkos::RangePolicy<KokkosExecSpace>(0, 2 * num_qubits_),
        CNotGateFunctor<Precision>(*tableau_x_data_, *tableau_z_data_,
                                   *tableau_r_data_, num_qubits_,
                                   target_qubit, control_qubit));
  }

private:
  size_t num_qubits_;
  std::mutex init_mutex_;
  std::unique_ptr<KokkosVector> tableau_x_data_;
  std::unique_ptr<KokkosVector> tableau_z_data_;
  std::unique_ptr<KokkosVector> tableau_r_data_;
  inline static bool is_exit_reg_ = false;
};

}; // namespace Plaquette

#include <functional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "CliffordState.hpp"

namespace {
using namespace Plaquette;
namespace py = pybind11;

PYBIND11_MODULE(plaquette_simulator_bindings, m) {

    py::class_<CliffordState<int>>(m, "CliffordState")
        .def(py::init<size_t, size_t, int, bool>())
        .def("get_num_qubits", &CliffordState<int>::GetNumQubits)
        .def("get_batch_size", &CliffordState<int>::GetBatchSize)
        .def("get_seed", &CliffordState<int>::GetSeed)
        .def("reset_clifford_state", &CliffordState<int>::ResetCliffordState)
        .def("measure_qubit", &CliffordState<int>::MeasureQubit)
        .def("get_kokkos_backend", &CliffordState<int>::GetKokkosBackend)
        .def("get_measurement", (std::optional<std::pair<int, int>>(
                                    CliffordState<int>::*)(size_t, size_t)) &
                                    CliffordState<int>::GetMeasurement)
        .def("apply_hadamard_gate", &CliffordState<int>::ApplyHadamardGate)
        .def("apply_control_not_gate", &CliffordState<int>::ApplyControlNotGate)
        .def("apply_control_phase_gate",
             &CliffordState<int>::ApplyControlPhaseGate)
        .def("apply_phase_gate", &CliffordState<int>::ApplyPhaseGate)
        .def("apply_pauli_x_gate", &CliffordState<int>::ApplyPauliXGate)
        .def("apply_pauli_z_gate", &CliffordState<int>::ApplyPauliZGate)
        .def("device_to_host", &CliffordState<int>::DeviceToHost);
}
} // namespace

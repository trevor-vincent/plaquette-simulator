Plaquette HPC Simulator Plugin
#################################

.. header-start-inclusion-marker-do-not-remove

The `Plaquette-HPC-Simulator <https://github.com/qc-design/plaquette-hpc-simualtor>`_ plugin extends the `Plaquette <https://github.com/qc-design/plaquette>`_ error correction software, providing a Kokkos-accelerated Clifford simulator.

.. installation-start-inclusion-marker-do-not-remove

Installation
============

The basic dependencies for installation are cmake and ninja.

The C++ tests/examples and python bindings can be built independently by

.. code-block:: console

   cmake -Bbuild -G Ninja
   cmake --build ./build

   
You can run the C++ backend tests with
   
.. code-block:: console

   make test-cpp


You can install just the python interface with (this quietly builds the C++ backend):

.. code-block:: console

   pip install -r requirements.txt
   pip install .


You can run the python frontend tests with
   
.. code-block:: console

   make test-python



Usage
==========

Python Frontend
---------------

.. code-block:: python

    import plaquette_simulator as ps

    init_to_zero = true
    seed = 123423
    batch_size = 10
    num_qubits = 5
    
    clifford_state = ps.CliffordState(num_qubits, batch_size, seed, init_to_zero)
    clifford_state.apply_hadamard_gate(0)
    clifford_state.apply_cnot_gate(0,1)
    clifford_state.measure_qubit(0)

C++ Backend
-----------

.. code-block:: cpp

    #include "CliffordState.hpp"

    int main(int argc, char *argv[]) {

        using namespace Plaquette;
        //a vector storing a flag that is 1 if the vertex is on the boundary
 
	int num_qubits = 5;
	int batch_size = 10;
	CliffordState clifford_state(num_qubits, batch_size);

	clifford_state.ApplyHadamardGate(0);
	clifford_state.ApplyControlNotGate(0,1);
	clifford_state.MeasureQubit(0);
    }

    
Documentation
=============

To generate the documentation you will need to install graphviz and doxygen. Then run

.. code-block:: console

   pip install -r doc/requirements.txt
   make docs
   firefox ./doc/_build/html/index.html

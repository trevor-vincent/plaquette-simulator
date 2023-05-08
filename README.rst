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

   cmake -Bbuild -G Ninja -DPLAQUETTE_HPC_SIMULATOR_BUILD_TESTS=On -DPLAQUETTE_HPC_SIMULATOR_BUILD_BINDINGS=On
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


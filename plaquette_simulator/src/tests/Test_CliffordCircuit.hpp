#pragma once

#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <Kokkos_Core.hpp>
#include <catch2/catch.hpp>

#include "CliffordState.hpp"
#include "CliffordCircuit.hpp"
#include "Test_Helpers.hpp"

using namespace Plaquette;
namespace {} // namespace

TEMPLATE_TEST_CASE("CliffordCircuit::Initialization",
                   "[circuit] [initialization]", int) {

  {
    const std::size_t num_qubits = 3;
    const std::size_t batch_size = 1;

    // Setup
    std::string filename = "testfile.txt";
    {
      std::ofstream outfile(filename);
      // outfile << "Line 1\nLine 2\nLine 3"; // Writing lines to file
    } // outfile destructor will close the file here

    CliffordCircuit<TestType> circuit(filename, num_qubits, batch_size);

    // Teardown
    std::remove(filename.c_str()); // This will delete the file
    REQUIRE(circuit.GetNumQubits() == num_qubits);
    REQUIRE(circuit.GetBatchSize() == batch_size);
    REQUIRE(circuit.GetNumOperations() == 0);
  }
}



TEMPLATE_TEST_CASE("CliffordCircuit::Execute",
                   "[circuit] [initialization]", int) {

  {
    const std::size_t num_qubits = 3;
    const std::size_t batch_size = 1;

    // Setup
    std::string filename = "testfile.txt";
    {
      std::ofstream outfile(filename);
      outfile << "M 0 1 2 3\n"; // Writing lines to file
    } // outfile destructor will close the file here

    CliffordCircuit<TestType> circuit(filename, num_qubits, batch_size);
    circuit.Execute();
    // Teardown
    std::remove(filename.c_str()); // This will delete the file
    REQUIRE(circuit.GetNumQubits() == num_qubits);
    REQUIRE(circuit.GetBatchSize() == batch_size);
    REQUIRE(circuit.GetNumOperations() == 1);
  }
}


TEMPLATE_TEST_CASE("CliffordCircuit::Execute1",
                   "[circuit] [initialization]", int) {

  {
    const std::size_t num_qubits = 3;
    const std::size_t batch_size = 1;

    // Setup
    std::string filename = "testfile.txt";
    {
      std::ofstream outfile(filename);
      outfile << "H 0\n S 0\n S 0\n H 0\n M 0"; // Writing lines to file
    } // outfile destructor will close the file here

    CliffordCircuit<TestType> circuit(filename, num_qubits, batch_size);
    circuit.Execute();
    auto result = circuit.GetCliffordState().GetMeasurement(0,0);
    REQUIRE(result.has_value());
    REQUIRE(result.value().first == 1);  // Check the measured value.
    REQUIRE(result.value().second == 1); // Check that the measurement was
    
    // Teardown
    std::remove(filename.c_str()); // This will delete the file
    REQUIRE(circuit.GetNumQubits() == num_qubits);
    REQUIRE(circuit.GetBatchSize() == batch_size);
    REQUIRE(circuit.GetNumOperations() == 5);
  }
}


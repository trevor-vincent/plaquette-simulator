#include <iostream>
#include <chrono>

#include <Kokkos_Core.hpp>

#include "CliffordState.hpp"
#include "CliffordCircuit.hpp"


int main(int argc, char *argv[])
{
  using namespace Plaquette;

  if (argc != 4) {
    std::cout << "Usage: " << argv[0] << " <file_name> <n_qubits> <batch_size>" << std::endl;
    return 1;
  }

  std::string file_name = argv[1];
  size_t num_qubits = std::stoi(argv[2]);
  size_t batch_size = std::stoi(argv[3]);
  
  CliffordCircuit<int> circuit(file_name, num_qubits, batch_size);

  auto t1 = std::chrono::high_resolution_clock::now();
  circuit.Execute();  
  auto t2 = std::chrono::high_resolution_clock::now();
  auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
  std::cout << file_name << " " << time_span.count() << std::endl;   
  
  return 0;
}

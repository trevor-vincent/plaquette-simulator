#pragma once

#include <sstream>
#include <unordered_set>
#include <vector>
#include <iostream>
#include <fstream>

class CliffordCircuit {
private:
  const std::unordered_set<std::string> supported_operations = {
      "X", "Y", "Z", "H", "M", "CX", "CZ", "E_PAULI", "E_PAULI2"};

  struct CliffordOperation {
    std::string name;
    std::vector<float> parameters;
    std::vector<size_t> qubits;
  };

  std::vector<CliffordOperation> operations;

  void read_operations(std::istream &stream) {
    std::string line;
    while (std::getline(stream, line)) {
      std::istringstream line_stream(line);

      CliffordOperation op;
      line_stream >> op.name;


      if (!supported_operations.count(op.name)) {
	throw std::runtime_error("Unsupported operation: " + op.name);
      }

      if (op.name == "E_PAULI") {
	for (size_t i = 0; i < 3; i++) {
	  float param;
	  line_stream >> param;
	  op.parameters.push_back(param);
	}	
      }
      else if (op.name == "E_PAULI2") {
	for (size_t i = 0; i < 15; i++) {
	  float param;
	  line_stream >> param;
	  op.parameters.push_back(param);
	}
      }

      size_t qubit; 
      while (line_stream >> qubit) {
	op.qubits.push_back(qubit);
      }
      
      operations.push_back(op);
    }
  }

public:
  // Constructor that initializes the circuit from a string or file
  CliffordCircuit(const std::string &input, bool isFile = false) {
    std::istringstream ss(input);
    if (isFile) {
      std::ifstream inFile(input);
      read_operations(inFile);
      inFile.close();
    } else {
      read_operations(ss);
    }
  }

  void Print(){
    std::cout << "Circuit:" << std::endl;
    for (auto op : operations) {
      std::cout << op.name << " ";
      for (auto param : op.parameters) {
	std::cout << param << " ";
      }
      for (auto qubit : op.qubits) {
	std::cout << qubit << " ";
      }
      std::cout << std::endl;
    }
  }

};

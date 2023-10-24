#pragma once

#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <unordered_set>
#include <vector>

#include "CliffordState.hpp"

namespace Plaquette {

template <typename Precision>
class CliffordCircuit {
private:
  const std::unordered_set<std::string> supported_operations = {
    "X", "Y", "Z", "H", "M", "CX", "CZ", "S"};

  struct CliffordOperation {
    std::string name;
    std::vector<float> parameters;
    std::vector<size_t> qubits;
  };

  CliffordState<Precision> state_;
  std::vector<CliffordOperation> operations_;
  std::map<std::string, std::function<void(const std::vector<size_t> &)>>
      no_param_operation_map_;

  void ReadOperations(std::istream &stream) {
    std::string line;
    while (std::getline(stream, line)) {
      std::istringstream line_stream(line);

      CliffordOperation op;
      line_stream >> op.name;

      if (!supported_operations.count(op.name)) {
        throw std::runtime_error("Unsupported operation: " + op.name);
      }

      // if (op.name == "E_PAULI") {
      //   for (size_t i = 0; i < 3; i++) {
      //     float param;
      //     line_stream >> param;
      //     op.parameters.push_back(param);
      //   }
      // } else if (op.name == "E_PAULI2") {
      //   for (size_t i = 0; i < 15; i++) {
      //     float param;
      //     line_stream >> param;
      //     op.parameters.push_back(param);
      //   }
      // }

      size_t qubit;
      while (line_stream >> qubit) {
        op.qubits.push_back(qubit);
      }
      
      operations_.push_back(op);
    }

  }

  void ApplyOperation(const CliffordOperation & op) {
    if (op.parameters.empty()) {
      no_param_operation_map_[op.name](op.qubits);
    }
  }

public:
  // Constructor that initializes the circuit from a string or file
  CliffordCircuit(
		  const std::string &file_name,
		  size_t num_qubits,
		  size_t batch_size
		  ) : state_(num_qubits, batch_size) 
  {

    {
      std::ifstream inFile(file_name);
      ReadOperations(inFile);
      inFile.close();
    }
    
    no_param_operation_map_["H"] = [this](const std::vector<size_t> &target) {
      state_.ApplyHadamardGateToQubits(target);
    };
    no_param_operation_map_["CX"] = [this](const std::vector<size_t> &target) {
      state_.ApplyControlNotGateToQubits(target);
    };
    no_param_operation_map_["CZ"] = [this](const std::vector<size_t> &target) {
      state_.ApplyControlPhaseGateToQubits(target);
    };    
    no_param_operation_map_["X"] = [this](const std::vector<size_t> &target) {
      state_.ApplyPauliXGateToQubits(target);
    };
    no_param_operation_map_["S"] = [this](const std::vector<size_t> &target) {
      state_.ApplyPhaseGateToQubits(target);
    };
    no_param_operation_map_["Y"] = [this](const std::vector<size_t> &target) {
      state_.ApplyPauliYGateToQubits(target);
    };
    no_param_operation_map_["Z"] = [this](const std::vector<size_t> &target) {
      state_.ApplyPauliZGateToQubits(target);
    };
    no_param_operation_map_["M"] = [this](const std::vector<size_t> &target) {
      state_.MeasureQubits(target);
    };
  }

  auto & GetCliffordState() {
    return state_;
  }

  auto GetNumQubits() const {
    return state_.GetNumQubits();
  }

  auto GetBatchSize() const {
    return state_.GetBatchSize();
  }

  auto GetNumOperations() const {
    return operations_.size();
  }
  
  void Execute() {
    for (const auto &op : operations_) {
      ApplyOperation(op);
    }
  }

  void Print() {
    std::cout << "Circuit:" << std::endl;
    for (auto op : operations_) {
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
};

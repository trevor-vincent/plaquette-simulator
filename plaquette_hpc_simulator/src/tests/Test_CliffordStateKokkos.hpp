#include <algorithm>
#include <complex>
#include <iostream>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch.hpp>

#include "CliffordStateKokkos.hpp"

using namespace Plaquette;
namespace {} // namespace

#define PRINT_MATRIX(matrix, numRows, numColumns) print_matrix(matrix, numRows, numColumns, #matrix)

void print_matrix(const std::vector<int>& matrix, int numRows, int numColumns, const char* varName) {
    int size = matrix.size();
    if (size != numRows * numColumns) {
        std::cout << "Invalid matrix dimensions for variable '" << varName << "'." << std::endl;
        return;
    }
    std::cout << "Matrix for variable '" << varName << "':" << std::endl;
    
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numColumns; ++j) {
            int index = i * numColumns + j;
            std::cout << matrix[index] << " ";
        }
        std::cout << std::endl;
    }
}

TEMPLATE_TEST_CASE("CliffordStateKokkos::CheckInitialization",
                   "[CliffordStateKokkos]", int) {

  {
    const std::size_t num_qubits = 3;
    CliffordStateKokkos<TestType> kokkos_state_1{num_qubits};
    std::vector<TestType> tableau_x_host(2*num_qubits*num_qubits);
    std::vector<TestType> tableau_z_host(2*num_qubits*num_qubits);
    std::vector<TestType> tableau_sign_host(2*num_qubits);
    
	
    kokkos_state_1.DeviceToHost(tableau_x_host.data(),
				tableau_z_host.data(),
				tableau_sign_host.data());


    std::vector<std::vector<int>> tableau_x_check = {
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
        {0, 0, 0},
        {0, 0, 0},
        {0, 0, 0}
    };

   std::vector<std::vector<int>> tableau_z_check = {
     {0, 0, 0},
     {0, 0, 0},
     {0, 0, 0},
     {1, 0, 0},
     {0, 1, 0},
     {0, 0, 1}
   };
   
    
   for (int i = 0; i < 2*num_qubits*num_qubits; ++i) {
     REQUIRE(tableau_x_host[i] == tableau_x_check[i/num_qubits][i%num_qubits]);
     REQUIRE(tableau_z_host[i] == tableau_z_check[i/num_qubits][i%num_qubits]);
   }

   for (int i = 0; i < 2*num_qubits; ++i) {
     REQUIRE(tableau_sign_host[i] == 0);
   }

   // PRINT_MATRIX(tableau_x_host, 2*num_qubits, num_qubits);
    // PRINT_MATRIX(tableau_z_host, 2*num_qubits, num_qubits);
    // PRINT_MATRIX(tableau_sign_host, 2*num_qubits, 1);
    // REQUIRE(1==0); 
  }
}


// TEMPLATE_TEST_CASE("CliffordStateKokkos::CopyConstructor",
//                    "[CliffordStateKokkos]", int) {

//   {
//     const std::size_t num_qubits = 3;
//     CliffordStateKokkos<TestType> kokkos_state_1{num_qubits};
//     CliffordStateKokkos<TestType> kokkos_state_2{kokkos_state_1};

//     CHECK(kokkos_state_1.GetNumQubits() == kokkos_state_2.GetNumQubits());

//     std::vector<Kokkos::complex<TestType>> kokkos_state_1_host(kokkos_state_1.getLength());

//     std::vector<Kokkos::complex<TestType>> kokkos_state_2_host(kokkos_state_2.getLength());
	
//     kokkos_state_1.DeviceToHost(kokkos_state_1_host.data(),
// 				kokkos_state_1.getLength());

//     kokkos_state_2.DeviceToHost(kokkos_state_2_host.data(),
// 				kokkos_state_2.getLength());

//     // for (size_t i = 0; i < kokkos_state_1_host.size(); i++) {
//     //   CHECK(kokkos_state_1_host[i] == kokkos_state_2_host[i]);
//     // }
//   }
// }

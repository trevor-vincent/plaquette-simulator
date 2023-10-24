#pragma once
#include <iostream>
#include <vector>

#define PRINT_MATRIX_BATCH(variable, batch_size, tableau_width, num_qubits)    \
    do {                                                                       \
        std::cout << #variable << ":\n";                                       \
        for (size_t batch_id = 0; batch_id < batch_size; ++batch_id) {         \
            std::cout << "Batch " << batch_id << ":\n";                        \
            for (size_t i = 0; i < tableau_width; ++i) {                       \
                for (size_t j = 0; j < num_qubits; ++j) {                      \
                    std::cout                                                  \
                        << variable[batch_id * tableau_width * num_qubits +    \
                                    i * num_qubits + j]                        \
                        << " ";                                                \
                }                                                              \
                std::cout << "\n";                                             \
            }                                                                  \
            std::cout << "\n";                                                 \
        }                                                                      \
    } while (0)

#define PRINT_VECTOR_BATCH(variable, batch_size, tableau_width)                \
    do {                                                                       \
        std::cout << #variable << ":\n";                                       \
        for (size_t batch_id = 0; batch_id < batch_size; ++batch_id) {         \
            std::cout << "Batch " << batch_id << ": ";                         \
            for (size_t i = 0; i < tableau_width; ++i) {                       \
                std::cout << variable[batch_id * tableau_width + i] << " ";    \
            }                                                                  \
            std::cout << "\n";                                                 \
        }                                                                      \
    } while (0)

#define PRINT_VECTOR_3D(v)                                                     \
    std::cout << #v << ":\n";                                                  \
    for (size_t i = 0; i < v.size(); ++i) {                                    \
        std::cout << "Batch " << i << ":\n";                                   \
        for (size_t j = 0; j < v[i].size(); ++j) {                             \
            for (size_t k = 0; k < v[i][j].size(); ++k) {                      \
                std::cout << v[i][j][k] << " ";                                \
            }                                                                  \
            std::cout << "\n";                                                 \
        }                                                                      \
        std::cout << "\n";                                                     \
    }

#define PRINT_VECTOR_2D(v)                                                     \
    std::cout << #v << ":\n";                                                  \
    for (size_t i = 0; i < v.size(); ++i) {                                    \
        std::cout << "Batch " << i << ":\n";                                   \
        for (size_t j = 0; j < v[i].size(); ++j) {                             \
            std::cout << v[i][j] << " ";                                       \
        }                                                                      \
        std::cout << "\n\n";                                                   \
    }



#include <vector>
#include <iostream>

template <typename T>
class RawPointer3D {
    T*** ptr;
    int dim1, dim2, dim3;

public:
    // Constructor using dimensions
    RawPointer3D(int d1, int d2, int d3) : dim1(d1), dim2(d2), dim3(d3) {
        allocate();
        initialize();
    }

    // Constructor using a 3D vector
    RawPointer3D(const std::vector<std::vector<std::vector<T>>>& vec) {
        dim1 = vec.size();
        dim2 = (dim1 > 0) ? vec[0].size() : 0;
        dim3 = (dim2 > 0) ? vec[0][0].size() : 0;
        allocate();
        CopyVectorToPointer(vec);
    }

    // Destructor
    ~RawPointer3D() {
        for (int i = 0; i < dim1; ++i) {
            for (int j = 0; j < dim2; ++j) {
                delete[] ptr[i][j];
            }
            delete[] ptr[i];
        }
        delete[] ptr;
    }

  T*** data(){
    return ptr;
  }
  
    // Allocate memory for the 3D pointer
    void allocate() {
        ptr = new T**[dim1];
        for (int i = 0; i < dim1; ++i) {
            ptr[i] = new T*[dim2];
            for (int j = 0; j < dim2; ++j) {
                ptr[i][j] = new T[dim3];
            }
        }
    }

    // Initialize the 3D pointer with default values
    void initialize() {
        for (int i = 0; i < dim1; ++i) {
            for (int j = 0; j < dim2; ++j) {
                for (int k = 0; k < dim3; ++k) {
                    ptr[i][j][k] = T();
                }
            }
        }
    }

    // Copy data from pointer to vector
    void CopyPointerToVector(std::vector<std::vector<std::vector<T>>>& vec) {
        vec.resize(dim1);
        for (int i = 0; i < dim1; ++i) {
            vec[i].resize(dim2);
            for (int j = 0; j < dim2; ++j) {
                vec[i][j].resize(dim3);
                for (int k = 0; k < dim3; ++k) {
                    vec[i][j][k] = ptr[i][j][k];
                }
            }
        }
    }

    // Copy data from vector to pointer
    void CopyVectorToPointer(const std::vector<std::vector<std::vector<T>>>& vec) {
        for (int i = 0; i < dim1; ++i) {
            for (int j = 0; j < dim2; ++j) {
                for (int k = 0; k < dim3; ++k) {
                    ptr[i][j][k] = vec[i][j][k];
                }
            }
        }
    }

    void print() {
        for (int i = 0; i < dim1; ++i) {
            for (int j = 0; j < dim2; ++j) {
                for (int k = 0; k < dim3; ++k) {
                    std::cout << ptr[i][j][k] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "----" << std::endl;
        }
    }
  
};

template <typename T>
class RawPointer2D {
    T** ptr;
    int dim1, dim2;

public:
    // Constructor using dimensions
    RawPointer2D(int d1, int d2) : dim1(d1), dim2(d2) {
        allocate();
        initialize();
    }

    // Constructor using a 2D vector
    RawPointer2D(const std::vector<std::vector<T>>& vec) {
        dim1 = vec.size();
        dim2 = (dim1 > 0) ? vec[0].size() : 0;
        allocate();
        CopyVectorToPointer(vec);
    }

    // Destructor
    ~RawPointer2D() {
        for (int i = 0; i < dim1; ++i) {
            delete[] ptr[i];
        }
        delete[] ptr;
    }

    // Allocate memory for the 2D pointer
    void allocate() {
        ptr = new T*[dim1];
        for (int i = 0; i < dim1; ++i) {
            ptr[i] = new T[dim2];
        }
    }

    // Initialize the 2D pointer with default values
    void initialize() {
        for (int i = 0; i < dim1; ++i) {
            for (int j = 0; j < dim2; ++j) {
                ptr[i][j] = T();
            }
        }
    }

    // Copy data from pointer to vector
    void CopyPointerToVector(std::vector<std::vector<T>>& vec) {
        vec.resize(dim1);
        for (int i = 0; i < dim1; ++i) {
            vec[i].resize(dim2);
            for (int j = 0; j < dim2; ++j) {
                vec[i][j] = ptr[i][j];
            }
        }
    }

    // Copy data from vector to pointer
    void CopyVectorToPointer(const std::vector<std::vector<T>>& vec) {
        for (int i = 0; i < dim1; ++i) {
            for (int j = 0; j < dim2; ++j) {
                ptr[i][j] = vec[i][j];
            }
        }
    }

    T** data(){
    return ptr;
    }


    void print() {
        for (int i = 0; i < dim1; ++i) {
            for (int j = 0; j < dim2; ++j) {
                std::cout << ptr[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
  
};
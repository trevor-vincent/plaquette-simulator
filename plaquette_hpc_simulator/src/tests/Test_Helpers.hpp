#include <vector>

template <typename T>
struct RawPtr3D {
    T*** ptr;

    RawPtr3D(const std::vector<std::vector<std::vector<T>>>& vec) {
        int outerSize = vec.size();
        ptr = new T**[outerSize];

        for (int i = 0; i < outerSize; ++i) {
            int middleSize = vec[i].size();
            ptr[i] = new T*[middleSize];

            for (int j = 0; j < middleSize; ++j) {
                int innerSize = vec[i][j].size();
                ptr[i][j] = new T[innerSize];

                for (int k = 0; k < innerSize; ++k) {
                    ptr[i][j][k] = vec[i][j][k];
                }
            }
        }
    }

    ~RawPtr3D() {
        for (int i = 0; i < outerSize; ++i) {
            for (int j = 0; j < middleSize; ++j) {
                delete[] ptr[i][j];
            }
            delete[] ptr[i];
        }
        delete[] ptr;
    }
};


template <typename T>
struct RawPtr2D {
    T** ptr;
    int outerSize;

    RawPtr2D(const std::vector<std::vector<T>>& vec) {
        outerSize = vec.size();
        ptr = new T*[outerSize];

        for (int i = 0; i < outerSize; ++i) {
            int innerSize = vec[i].size();
            ptr[i] = new T[innerSize];

            for (int j = 0; j < innerSize; ++j) {
                ptr[i][j] = vec[i][j];
            }
        }
    }

    ~RawPtr2D() {
        for (int i = 0; i < outerSize; ++i) {
            delete[] ptr[i];
        }
        delete[] ptr;
    }
};

#pragma once

#include <fstream>
#include <Eigen/Dense>
#include <iomanip>

template<typename T>
struct Pixel {
    T channel1, channel2, channel3;
};

template<typename T>
class Image2D {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> channel1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> channel2;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> channel3;

public:
    // !!! width & height not necessarily correct for each channel after sampling
    int width;
    int height;
    int sx, sy;
    int maxVal;
    bool isSampled;

    // constructor
    // explicit keyword prevents the compiler from using that constructor for implicit conversions
    // implicit conversion: void foo(Image2D image) {}
    // I could now pass an int to foo --> foo(42) because 42 gets passed to the Image2D constructor and will be
    // implicitly converted
    explicit Image2D(int w = 0, int h = 0, const int mV = -1) : width(w), height(h),
                                                                sx(1), sy(1),
                                                                maxVal(mV), isSampled(false) {
        if (w > 0 && h > 0) {
            channel1.resize(h, w);
            channel2.resize(h, w);
            channel3.resize(h, w);
        }
    }

    // getters
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &getChannel1() { return channel1; }
    [[nodiscard]] const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &getChannel1() const { return channel1; }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &getChannel2() { return channel2; }
    [[nodiscard]] const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &getChannel2() const { return channel2; }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &getChannel3() { return channel3; }
    [[nodiscard]] const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &getChannel3() const { return channel3; }


    // setters
    void setChannel1(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &matrix) { channel1 = matrix; }
    void setChannel2(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &matrix) { channel2 = matrix; }
    void setChannel3(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &matrix) { channel3 = matrix; }


    // get/set pixel
    [[nodiscard]] Pixel<T> getPixel(int x, int y) const {
        T c1 = channel1(y, x);
        if (isSampled) {
            int y2 = y / sy;
            int x2 = x / sx;
            return {c1, channel2(y2, x2), channel3(y2, x2)};
        }
        return {c1, channel2(y, x), channel3(y, x)};
    }

    void setPixel(int x, int y, T c1, T c2, T c3) {
        channel1(y, x) = c1;

        if (isSampled) {
            int y2 = y / sy;
            int x2 = x / sx;

            channel2(y2, x2) = c2;
            channel3(y2, x2) = c3;
        } else {
            channel2(y, x) = c2;
            channel3(y, x) = c3;
        }
    }

    // Sets chrominance channels (Cb/Cr) for the pixel at (x, y).
    // Note: Eigen matrices are indexed as (row=y, col=x).
    void setColourPixel(int x, int y, T c2, T c3) {
        channel2(y, x) = c2;
        channel3(y, x) = c3;
    }


    // Sets the Steps for sampling and expands the existing matrix to the furthest over-step possible
    void setSteps(const int sxI, const int syI) {
        sx = sxI;
        sy = syI;

        // Helper for ceiling to the next multiple
        auto ceil_mul = [](const int size, const int step) -> int {
            return ((size + step - 1) / step) * step; // == ceil(size / step) * step
        };

        int newWidth = ceil_mul(width, sx);
        int newHeight = ceil_mul(height, sy);

        channel1.conservativeResize(newHeight, newWidth);
        channel2.conservativeResize(newHeight, newWidth);
        channel3.conservativeResize(newHeight, newWidth);

        // Fill newly added pixels (bottom and right edges)
        for (int y = height; y < newHeight; ++y) {
            channel1.row(y) = channel1.row(height - 1);
            channel2.row(y) = channel2.row(height - 1);
            channel3.row(y) = channel3.row(height - 1);
        }

        for (int y = 0; y < newHeight; ++y) // all rows
            for (int x = width; x < newWidth; ++x) {
                channel1(y, x) = channel1(y, width - 1);
                channel2(y, x) = channel2(y, width - 1);
                channel3(y, x) = channel3(y, width - 1);
            }

        width = newWidth;
        height = newHeight;
    }

    void printData() const {
        std::cout << "Image2D<" << typeid(T).name() << ">\n";
        std::cout << "Channel1 size: " << channel1.cols() << "x" << channel1.rows() << '\n';
        std::cout << "Channel2 size: " << channel2.cols() << "x" << channel2.rows() << '\n';
        std::cout << "Channel3 size: " << channel3.cols() << "x" << channel3.rows() << '\n';
        std::cout << "sx: " << sx << ", sy: " << sy << ", isSampled: " << std::boolalpha << isSampled << "\n\n";

        if constexpr (std::is_same_v<T, uint8_t>) {
            auto printMatrix = [](const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &m, const std::string &name) {
                std::cout << name << " (" << m.cols() << "x" << m.rows() << "):\n";
                for (int y = 0; y < m.rows(); ++y) {
                    for (int x = 0; x < m.cols(); ++x)
                        std::cout << std::setw(3) << static_cast<int>(m(y, x)) << ' ';
                    std::cout << '\n';
                }
                std::cout << '\n';
            };

            printMatrix(channel1, "Channel1 (Y)");
            printMatrix(channel2, "Channel2 (Cb)");
            printMatrix(channel3, "Channel3 (Cr)");
        } else {
            auto printMatrix = [](const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &m, const std::string &name) {
                std::cout << name << " (" << m.cols() << "x" << m.rows() << "):\n";
                std::cout << std::fixed << std::setprecision(3);
                for (int y = 0; y < m.rows(); ++y) {
                    for (int x = 0; x < m.cols(); ++x)
                        std::cout << std::setw(8) << m(y, x) << ' ';
                    std::cout << '\n';
                }
                std::cout << '\n';
            };

            printMatrix(channel1, "Channel1 (Y)");
            printMatrix(channel2, "Channel2 (Cb)");
            printMatrix(channel3, "Channel3 (Cr)");
        }
    }

    void writeDataToFile(const std::string &filename) const {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Fehler: Datei konnte nicht geöffnet werden: " << filename << std::endl;
            return;
        }
        file << "Channel1 size: " << channel1.cols() << "x" << channel1.rows() << '\n';
        file << "Channel2 size: " << channel2.cols() << "x" << channel2.rows() << '\n';
        file << "Channel3 size: " << channel3.cols() << "x" << channel3.rows() << '\n';
        file << "sx: " << sx << ", sy: " << sy << ", isSampled: " << std::boolalpha << isSampled << "\n\n";

        if constexpr (std::is_same_v<T, uint8_t>) {
            auto writeMatrix = [&](const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &m, const std::string &name) {
                file << name << " (" << m.cols() << "x" << m.rows() << "):\n";
                for (int y = 0; y < m.rows(); ++y) {
                    for (int x = 0; x < m.cols(); ++x)
                        file << std::setw(3) << static_cast<int>(m(y, x)) << ' ';
                    file << '\n';
                }
                file << '\n';
            };

            writeMatrix(channel1, "Channel1 (Y)");
            writeMatrix(channel2, "Channel2 (Cb)");
            writeMatrix(channel3, "Channel3 (Cr)");
        } else {
            auto writeMatrix = [&](const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &m, const std::string &name) {
                file << name << " (" << m.cols() << "x" << m.rows() << "):\n";
                file << std::fixed << std::setprecision(3);
                for (int y = 0; y < m.rows(); ++y) {
                    for (int x = 0; x < m.cols(); ++x)
                        file << std::setw(8) << m(y, x) << ' ';
                    file << '\n';
                }
                file << '\n';
            };

            writeMatrix(channel1, "Channel1 (Y)");
            writeMatrix(channel2, "Channel2 (Cb)");
            writeMatrix(channel3, "Channel3 (Cr)");
        }

        file.close();
        std::cout << "Daten erfolgreich in " << filename << " geschrieben.\n";
    }
};

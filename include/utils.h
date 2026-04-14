#pragma once

#ifndef JPEG_PROJEKT_UTILS_H
#define JPEG_PROJEKT_UTILS_H

#include "Eigen/Core"
#include "Image2D.h"
#include <concepts>
#include <random>

template<typename T>
concept FloatOrDouble = std::same_as<T, float> || std::same_as<T, double>;

template<FloatOrDouble T>
Image2D<T> RGBtoYCbCr(const Image2D<uint8_t>& src, bool normalize) {
    using Vec3x1 = Eigen::Matrix<T, 3, 1>;
    using Mat3x3 = Eigen::Matrix<T, 3, 3>;

    static const Mat3x3 M{
            {0.299, 0.587, 0.114},
            {-0.1687, -0.3312, 0.5},
            {0.5, -0.4186, -0.0813}
    };
    static const Vec3x1 b{0.0, 128, 128};
    static const Vec3x1 n{128, 128, 128};

    Image2D<T> dst(src.width, src.height, src.maxVal);

    const auto& ch1 = src.getChannel1();
    const auto& ch2 = src.getChannel2();
    const auto& ch3 = src.getChannel3();

    for (int y = 0; y < src.height; ++y) {
        for (int x = 0; x < src.width; ++x) {
            Vec3x1 rgb(
                static_cast<T>(ch1(y, x)),
                static_cast<T>(ch2(y, x)),
                static_cast<T>(ch3(y, x))
            );
            Vec3x1 yCbCr = (b + M * rgb) - n;

            if (normalize) {
                yCbCr /= static_cast<T>(src.maxVal);
            }

            dst.setPixel(x, y, yCbCr.x(), yCbCr.y(), yCbCr.z());
        }
    }
    return dst;
}


inline Image2D<uint8_t> loadPPM(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open PPM file: " + filename);
    }

    std::string magicNumber;
    file >> magicNumber;
    if (magicNumber != "P3") {
        throw std::runtime_error("Unsupported PPM format (expected P3)");
    }

    auto eatComment = [](std::ifstream& f) -> int {
        std::string token;
        while (f >> token) {
            if (token[0] == '#') {
                std::getline(f, token); // Skip comments
                continue;
            }
            return std::stoi(token);
        }
        throw std::runtime_error("Unexpected end of PPM data while reading header");
    };

    int w = eatComment(file);
    int h = eatComment(file);
    int maxV = eatComment(file);

    Image2D<uint8_t> img(w, h, maxV);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int r, g, b;
            if (!(file >> r >> g >> b)) {
                throw std::runtime_error("Unexpected end of PPM data");
            }
            img.setPixel(x, y, static_cast<uint8_t>(r), static_cast<uint8_t>(g), static_cast<uint8_t>(b));
        }
    }

    return img;
}

inline bool compareFiles(const std::string &file1, const std::string &file2) {
    std::ifstream f1(file1);
    std::ifstream f2(file2);

    if (!f1.is_open()) {
        std::cerr << "Fehler: Datei konnte nicht geöffnet werden: " << file1 << std::endl;
        return false;
    }
    if (!f2.is_open()) {
        std::cerr << "Fehler: Datei konnte nicht geöffnet werden: " << file2 << std::endl;
        return false;
    }

    std::string line1, line2;
    int line_number = 1;

    while (true) {
        bool eof1 = !std::getline(f1, line1);
        bool eof2 = !std::getline(f2, line2);

        if (eof1 && eof2) {
            return true;
        }

        if (eof1 != eof2 || line1 != line2) {
            std::cout << "Unterschied in Zeile " << line_number << ":\n";
            std::cout << "Datei 1: " << line1 << "\n";
            std::cout << "Datei 2: " << line2 << "\n";
            return false;
        }
        line_number++;
    }
}

std::vector<int> generateRandomSymbolList(
    int symbolCount,
    int minSymbolValue,
    int maxSymbolValue,
    int minFrequency,
    int maxFrequency,
    int& totalElements
)
{
    totalElements = std::min(totalElements, symbolCount * maxFrequency);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> symbolDist(minSymbolValue, maxSymbolValue);
    std::uniform_int_distribution<> freqDist(minFrequency, maxFrequency);

    std::vector<int> result;
    result.reserve(totalElements);

    while (result.size() < totalElements) {
        int symbol = symbolDist(gen);
        int freq = freqDist(gen);

        for (int i = 0; i < freq && result.size() < totalElements; ++i) {
            result.push_back(symbol);
        }
    }
    return result;
}

int bitsNeeded(int values) {
    return std::ceil(std::log2(values));
}

void printChannel(const Eigen::MatrixXf& channel)
{
    const int rows = channel.rows();
    const int cols = channel.cols();

    // First pass: determine max width of any formatted number
    const int precision = 2;
    int maxWidth = 0;

    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(precision) << channel(r, c);
            int w = static_cast<int>(oss.str().size());
            if (w > maxWidth) maxWidth = w;
        }
    }

    std::cout << "Channel (" << rows << "x" << cols << "):\n";

    // Second pass: print with uniform width (includes space between columns)
    for (int r = 0; r < rows; ++r)
    {
        for (int c = 0; c < cols; ++c)
        {
            std::cout << std::setw(maxWidth + 1)  // +1 for spacing between columns
                      << std::fixed << std::setprecision(precision)
                      << channel(r, c);
        }
        std::cout << '\n';
    }

    std::cout << std::endl;
}

void create4k()
{
    const int width = 3840;
    const int height = 2160;

    std::ofstream file("4k.txt");
    if (!file.is_open()) {
        std::cerr << "Fehler beim Öffnen der Datei!" << std::endl;
        return;
    }

    // P3-PPM header
    file << "P3\n";
    file << width << " " << height << "\n";
    file << "255\n";

    // Write image data
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            int m = (x + y * 8) % 256;

            // RGB = m,m,m
            file << m << " " << m << " " << m << " ";
        }
        file << "\n";
    }

    file.close();
}

#endif //JPEG_PROJEKT_UTILS_H
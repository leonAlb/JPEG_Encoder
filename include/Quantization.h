#ifndef JPEG_PROJEKT_QUANTIZATION_H
#define JPEG_PROJEKT_QUANTIZATION_H
#include <vector>

#include <Eigen/Dense>

// Standard JPEG Y-component quantization table
const Eigen::Matrix<int, 8, 8, Eigen::RowMajor> jpeg_luma_quantization_table{
    {16, 11, 10, 16, 24, 40, 51, 61},
    {12, 12, 14, 19, 26, 58, 60, 55},
    {14, 13, 16, 24, 40, 57, 69, 56},
    {14, 17, 22, 29, 51, 87, 80, 62},
    {18, 22, 37, 56, 68, 109, 103, 77},
    {24, 35, 55, 64, 81, 104, 113, 92},
    {49, 64, 78, 87, 103, 121, 120, 101},
    {72, 92, 95, 98, 112, 100, 103, 99}
};

// Standard JPEG CbCr-component quantization table
const Eigen::Matrix<int, 8, 8> jpeg_chroma_quantization_table{
    {17, 18, 24, 47, 99, 99, 99, 99},
    {18, 21, 26, 66, 99, 99, 99, 99},
    {24, 26, 56, 99, 99, 99, 99, 99},
    {47, 66, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99},
    {99, 99, 99, 99, 99, 99, 99, 99}
};

inline void quantize_channel(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &channel, int height, int width,
                             Eigen::Matrix<int, 8, 8> quantTable) {
#pragma omp parallel for collapse(2)
    for (int by = 0; by < height; by += 8) {
        for (int bx = 0; bx < width; bx += 8) {
            float block[8][8];
            float inputBlock[8][8];

            // Local memory copy
            for (int x = 0; x < 8; x++) {
                for (int y = 0; y < 8; y++) {
                    inputBlock[x][y] = channel(by + x, bx + y);
                }
            }

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    block[i][j] = std::round(inputBlock[i][j] / static_cast<float>(quantTable(i, j)));
                }
            }

            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    channel(by + i, bx + j) = block[i][j];
                }
            }
        }
    }
}

// Zig-zag scan of an 8x8 block (JPEG order).
template <typename Derived>
std::vector<int> zigzag_order_block(const Eigen::MatrixBase<Derived>& block)
{
    static_assert(Derived::RowsAtCompileTime == 8 && Derived::ColsAtCompileTime == 8,
                  "zigzag_order_block expects an 8x8 matrix");

    std::vector<int> zigzag_block(64);
    int row = 0, col = 0;
    bool walkRight = true;

    for (int i = 0; i < 64; i++) {
        zigzag_block[i] = static_cast<int>(block(row, col));

        if (walkRight) { row--; col++; }
        else           { row++; col--; }

        if (row == 8 && col == -1) {
            row = row - 1;
            col = col + 2;
            walkRight = true;
            continue;
        }
        if (row == -1) { row = 0; walkRight = false; }
        if (col == -1) { col = 0; walkRight = true;  }
        if (col == 8)  { col = 7; row = row + 2; walkRight = false; }
        if (row == 8)  { row = 7; col = col + 2; walkRight = true;  }
    }
    return zigzag_block;
}

inline std::vector<std::vector<int>> get_zigzagged_blocks_mcu(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &ch,
    int sx, int sy)
{
    const int blocksX = ch.cols() / 8;
    const int blocksY = ch.rows() / 8;
    const int mcuX = blocksX / sx;
    const int mcuY = blocksY / sy;

    std::vector<std::vector<int>> out;
    out.reserve((size_t)blocksX * (size_t)blocksY);

    for (int my = 0; my < mcuY; ++my) {
        for (int mx = 0; mx < mcuX; ++mx) {
            for (int vy = 0; vy < sy; ++vy) {
                for (int hx = 0; hx < sx; ++hx) {
                    const int br = my * sy + vy;
                    const int bc = mx * sx + hx;

                    auto block = ch.block<8,8>(br*8, bc*8);
                    out.push_back(zigzag_order_block(block));
                }
            }
        }
    }
    return out;
}


#endif //JPEG_PROJEKT_QUANTIZATION_H

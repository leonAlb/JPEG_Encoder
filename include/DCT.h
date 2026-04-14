#include <cmath>
#include <Eigen/Dense>

#if defined(__i386__) || defined(__x86_64__) // x86/x86_64 SIMD intrinsics
#include <immintrin.h>
#endif

#if !defined(__APPLE__)
#include <omp.h>
#endif

#if defined(__APPLE__)
    #include <BS_thread_pool.hpp>
    #include <future> // std::future (used by the thread pool API)
#endif
#include <xsimd/xsimd.hpp>

// ---------------------------------------- DCT 1 ----------------------------------------
inline void dctDirectImp(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &channel, int height, int width) {
    const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    const int N = 8;

    static float cosineLUT[8][8];
    static float scaleLUT[8][8];
    static bool initialized = false;
    if (!initialized) {
        for (int xy = 0; xy < 8; ++xy) {
            for (int ij = 0; ij < 8; ++ij) {
                cosineLUT[xy][ij] = static_cast<float>(std::cos((2 * xy + 1) * ij * M_PI / (2.0 * N)));
            }
        }
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                float Ci = (i == 0) ? inv_sqrt2 : 1.0f;
                float Cj = (j == 0) ? inv_sqrt2 : 1.0f;
                scaleLUT[i][j] = 2.0f / N * Ci * Cj;
            }
        }
        initialized = true;
    }

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
                    float sum = 0.0f;
                    for (int x = 0; x < 8; x++) {
                        for (int y = 0; y < 8; y++) {
                            sum += inputBlock[x][y] * cosineLUT[x][i] * cosineLUT[y][j];
                        }
                    }
                    block[i][j] = scaleLUT[i][j] * sum;
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

inline void inverseDCTDirectImp(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &channel, int height, int width) {
    const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);
    const int N = 8;

    static float cosineLUT[8][8];
    static float scaleLUT[8][8];
    static bool initialized = false;
    if (!initialized) {
        for (int xy = 0; xy < 8; ++xy) {
            for (int ij = 0; ij < 8; ++ij) {
                cosineLUT[xy][ij] = static_cast<float>(std::cos((2 * xy + 1) * ij * M_PI / (2.0 * N)));
            }
        }
        for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
                float Ci = (i == 0) ? inv_sqrt2 : 1.0f;
                float Cj = (j == 0) ? inv_sqrt2 : 1.0f;
                scaleLUT[i][j] = 2.0f / N * Ci * Cj;
            }
        }
        initialized = true;
    }

#pragma omp parallel for collapse(2)
    for (int by = 0; by < height; by += 8) {
        for (int bx = 0; bx < width; bx += 8) {
            float block[8][8];
            float inputBlock[8][8];

            // Local memory copy (coefficients)
            for (int i = 0; i < 8; i++) {
                for (int j = 0; j < 8; j++) {
                    inputBlock[i][j] = channel(by + i, bx + j);
                }
            }

            for (int x = 0; x < 8; x++) {
                for (int y = 0; y < 8; y++) {
                    float sum = 0.0f;
                    for (int i = 0; i < 8; i++) {
                        for (int j = 0; j < 8; j++) {
                            sum += scaleLUT[i][j] * inputBlock[i][j] * cosineLUT[x][i] * cosineLUT[y][j];
                        }
                    }
                    block[x][y] = sum;
                }
            }

            // Write back (pixels)
            for (int x = 0; x < 8; x++) {
                for (int y = 0; y < 8; y++) {
                    channel(by + x, bx + y) = block[x][y];
                }
            }
        }
    }
}


// ---------------------------------------- DCT 2 ----------------------------------------
alignas(32) static float cosi[8][8];
alignas(32) static float cosi_t[8][8];
alignas(32) static float scale[8][8];
inline bool initialized = false;

inline void initDCTLUT()
{
    const float inv_sqrt2 = 1.0f / std::sqrt(2.0f);

    for (int x = 0; x < 8; x++) {
        for (int k = 0; k < 8; k++) {
            cosi[x][k] = std::cos((2*x+1)*k*M_PI/16.0f);
            cosi_t[k][x] = cosi[x][k];
        }
    }

    for (int u = 0; u < 8; u++) {
        for (int v = 0; v < 8; v++) {
            scale[u][v] = 0.25f * ((u==0)?inv_sqrt2:1.f) * ((v==0)?inv_sqrt2:1.f);
        }
    }

    initialized = true;
}

// Special 1D-DCT
inline void dct1D_8(float * x, float cosMat[8][8])

{
    alignas(32) float tmp[8];

    tmp[0] = x[0] * cosMat[0][0] + x[1] * cosMat[1][0] + x[2] * cosMat[2][0] + x[3] * cosMat[3][0] +
    x[4] * cosMat[4][0] + x[5] * cosMat[5][0] + x[6] * cosMat[6][0] + x[7] * cosMat[7][0];

    tmp[1] = x[0] * cosMat[0][1] + x[1] * cosMat[1][1] + x[2] * cosMat[2][1] + x[3] * cosMat[3][1] +
    x[4] * cosMat[4][1] + x[5] * cosMat[5][1] + x[6] * cosMat[6][1] + x[7] * cosMat[7][1];

    tmp[2] = x[0] * cosMat[0][2] + x[1] * cosMat[1][2] + x[2] * cosMat[2][2] + x[3] * cosMat[3][2] +
    x[4] * cosMat[4][2] + x[5] * cosMat[5][2] + x[6] * cosMat[6][2] + x[7] * cosMat[7][2];

    tmp[3] = x[0] * cosMat[0][3] + x[1] * cosMat[1][3] + x[2] * cosMat[2][3] + x[3] * cosMat[3][3] +
    x[4] * cosMat[4][3] + x[5] * cosMat[5][3] + x[6] * cosMat[6][3] + x[7] * cosMat[7][3];

    tmp[4] = x[0] * cosMat[0][4] + x[1] * cosMat[1][4] + x[2] * cosMat[2][4] + x[3] * cosMat[3][4] +
    x[4] * cosMat[4][4] + x[5] * cosMat[5][4] + x[6] * cosMat[6][4] + x[7] * cosMat[7][4];

    tmp[5] = x[0] * cosMat[0][5] + x[1] * cosMat[1][5] + x[2] * cosMat[2][5] + x[3] * cosMat[3][5] +
    x[4] * cosMat[4][5] + x[5] * cosMat[5][5] + x[6] * cosMat[6][5] + x[7] * cosMat[7][5];

    tmp[6] = x[0] * cosMat[0][6] + x[1] * cosMat[1][6] + x[2] * cosMat[2][6] + x[3] * cosMat[3][6] +
    x[4] * cosMat[4][6] + x[5] * cosMat[5][6] + x[6] * cosMat[6][6] + x[7] * cosMat[7][6];

    tmp[7] = x[0] * cosMat[0][7] + x[1] * cosMat[1][7] + x[2] * cosMat[2][7] + x[3] * cosMat[3][7] +
    x[4] * cosMat[4][7] + x[5] * cosMat[5][7] + x[6] * cosMat[6][7] + x[7] * cosMat[7][7];

    for (int i = 0; i < 8; ++i)
        x[i] = tmp[i];
}

inline void dctSeparated(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>& input)
{
    if (!initialized)
        initDCTLUT();

    const int H = input.rows();
    const int W = input.cols();

    alignas(32) float blockData[8][8];

#pragma omp parallel for collapse(2) schedule(static) firstprivate(blockData)
    for (int by = 0; by < H; by += 8)
    {
        for (int bx = 0; bx < W; bx += 8)
        {
            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    blockData[r][c] = input(by + r, bx + c);
                }
            }

            // 1D-DCT Zeilen
            #pragma omp simd
            for (int r = 0; r < 8; r++) {
                dct1D_8(blockData[r], cosi);
            }

            // 1D-DCT Spalten
            alignas(32) float tmp_col[8];
            for (int c = 0; c < 8; c++)
            {
                for (int r = 0; r < 8; r++)
                {
                    tmp_col[r] = blockData[r][c];
                }

                dct1D_8(tmp_col, cosi);

                for (int r = 0; r < 8; r++) {
                    blockData[r][c] = tmp_col[r] * scale[r][c];
                }
            }

            for (int r = 0; r < 8; r++) {
                for (int c = 0; c < 8; c++) {
                    input(by + r, bx + c) = blockData[r][c];
                }
            }
        }
    }
}
// ---------------------------------------- DCT 3 ----------------------------------------
using batchf = xsimd::batch<float>;
using Block8x8 = Eigen::Block<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>, 8, 8>;
using BlockT8x8 = Eigen::Block<Eigen::Transpose<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>, 8, 8>;

template<typename T>
concept RowOrCol =
std::same_as<T, Eigen::Block<Block8x8, 8, 1, true>> || std::same_as<T, Eigen::Block<BlockT8x8, 8, 1>> || std::same_as<T, Eigen::Vector<float, 8>>;

#if defined(__APPLE__)
// calculates C params
float Ck(const int &k) {
    return static_cast<float>(std::cos((k*M_PI) / 16.0f));
}


// calculates params for a
void calc_params_a(float *a) {
    a[0] =  Ck(4);
    a[1] = (Ck(2) - Ck(6));
    a[2] =  Ck(4);
    a[3] = (Ck(6) + Ck(2));
    a[4] =  Ck(6);
}


// calculates params for s
void calc_params_s(float *s) {
    s[0] = static_cast<float>(1.0 / (2*std::sqrt(2)));

    for(int i=1; i<=7; i++) {
        s[i] = 1.0f / (4 * Ck(i));
    }
}


void negate_elements_of_batch(batchf &ba, float a, float b, float c, float d) {
    batchf nb = {a, b, c, d};
    ba = ba * nb;
}


// xvecf + resize: 8.39505 ms / float arr init 4: 1.26576 ms / discarded vec: 1.25622 ms
template<RowOrCol T>
void aan_core(T x, const float *a, const float *s) {
    // -------------------------------------------------- first iteration (b) of calculation --> from x to first dot in signal flow diagram (script) --------------------------------------------------
    batchf fs_1 = {x(0) /* = x0 */, x(1) /* = x1 */, x(2) /* = x2 */, x(3) /* = x3 */ };
    batchf ss_1 = {x(7) /* = x7 */, x(6) /* = x6 */, x(5) /* = x5 */, x(4) /* = x4 */ };

    batchf fs_2 = {x(3) /* = x3 */, x(2) /* = x2 */, x(1) /* = x1 */, x(0) /* = x0 */ };
    batchf ss_2 = {-x(4) /* = -x4 */, -x(5) /* = -x5 */, -x(6) /* = -x6 */, -x(7) /* = -x7 */ };

    batchf result_1 = fs_1 + ss_1; // b 0 - 3
    batchf result_2 = fs_2 + ss_2; // b 4 - 7



    // -------------------------------------------------- second iteration (c) of calculation --> from first dot to second dot in signal flow diagram (script) --------------------------------------------------
    fs_1 = xsimd::shuffle(result_1, result_2, xsimd::batch_constant<int, xsimd::default_arch, 0, 1, 1, 0>());
    ss_1 = xsimd::shuffle(result_1, result_2, xsimd::batch_constant<int, xsimd::default_arch, 3, 2, 2, 3>());
    negate_elements_of_batch(ss_1, 1, 1, -1, -1);

    fs_2 = xsimd::shuffle(result_1, result_2, xsimd::batch_constant<int, xsimd::default_arch, 4, 5, 6, 7>());
    negate_elements_of_batch(fs_2, -1, 1, 1, 1);
    ss_2 = xsimd::shuffle(result_1, result_2, xsimd::batch_constant<int, xsimd::default_arch, 5, 6, 7, 1>());
    negate_elements_of_batch(ss_2, -1, 1, 1, 0);

    result_1 = fs_1 + ss_1; // c 0 - 3
    result_2 = fs_2 + ss_2; // c 4 - 7



    // -------------------------------------------------- third iteration (d) --------------------------------------------------
    batchf tmp = xsimd::shuffle(result_1, result_2, xsimd::batch_constant<int, xsimd::default_arch, 3, 3, 3, 3>()); // tmp: {c3=d3, c3=d3, c3=d3, c3=d3}

    fs_1 = xsimd::shuffle(result_1, result_2, xsimd::batch_constant<int, xsimd::default_arch, 0, 0, 2, 4>());
    ss_1 = xsimd::shuffle(result_1, result_2, xsimd::batch_constant<int, xsimd::default_arch, 1, 1, 3, 6>());
    negate_elements_of_batch(ss_1, 1, -1, 1, 1);

    result_1 = fs_1 + ss_1; // d0, d1, d2, d8



    // -------------------------------------------------- fourth iteration (e) --------------------------------------------------
    fs_1 = xsimd::shuffle(result_1, tmp, xsimd::batch_constant<int, xsimd::default_arch, 0, 1, 2, 4>());
    ss_1 = {1, 1, a[0] /*a1*/, 1};

    fs_2 = result_2;
    ss_2 = {a[1] /*a2*/, a[2] /*a3*/, a[3] /*a4*/, 1};
    tmp = fs_2 * ss_2;
    negate_elements_of_batch(tmp, -1, 1, 1, 1);

    batchf tmp_1 = xsimd::shuffle(result_1, result_2, xsimd::batch_constant<int, xsimd::default_arch, 3, 0, 3, 0>());
    batchf tmp_2 = {a[4] /*a5*/, 0, a[4] /*a5*/, 0};
    batchf tmp_res = tmp_1 * tmp_2;
    negate_elements_of_batch(tmp_res, -1, 0, -1, 0);

    result_1 = fs_1 * ss_1; // e0, e1, e2, e3
    result_2 = tmp + tmp_res; // e4, e5, e6, e7



    // -------------------------------------------------- fifth iteration (f) --------------------------------------------------
    fs_1 = xsimd::shuffle(result_1, result_2, xsimd::batch_constant<int, xsimd::default_arch, 0, 1, 2, 2>());
    ss_1 = xsimd::shuffle(result_1, result_2, xsimd::batch_constant<int, xsimd::default_arch, 0, 0, 3, 3>());
    negate_elements_of_batch(fs_1, 1, 1, 1, -1);
    negate_elements_of_batch(ss_1, 0, 0, 1, 1);

    fs_2 = xsimd::shuffle(result_1, result_2, xsimd::batch_constant<int, xsimd::default_arch, 4, 5, 6, 5>());
    ss_2 = xsimd::shuffle(result_1, result_2, xsimd::batch_constant<int, xsimd::default_arch, 0, 7, 0, 7>());
    negate_elements_of_batch(fs_2, 1, 1, 1, -1);
    negate_elements_of_batch(ss_2, 0, 1, 0, 1);

    result_1 = fs_1 + ss_1; // f0, f1, f2, f3
    result_2 = fs_2 + ss_2; // f4, f5, f6, f7



    // -------------------------------------------------- sixth iteration (g) --------------------------------------------------
    // result_1 g0, g1, g2, g3
    fs_2 = xsimd::shuffle(result_1, result_2, xsimd::batch_constant<int, xsimd::default_arch, 4, 5, 5, 4>());
    ss_2 = xsimd::shuffle(result_1, result_2, xsimd::batch_constant<int, xsimd::default_arch, 7, 6, 6, 7>());
    negate_elements_of_batch(fs_2, 1, 1, 1, -1);
    negate_elements_of_batch(ss_2, 1, 1, -1, 1);

    result_2 = fs_2 + ss_2; // g4, g5, g6, g7



    // -------------------------------------------------- seventh iteration (y) --------------------------------------------------
    fs_1 = result_1;
    ss_1 = {s[0], s[4], s[2], s[6]};

    fs_2 = result_2;
    ss_2 = {s[5], s[1], s[7], s[3]};

    result_1 = fs_1 * ss_1;
    result_2 = fs_2 * ss_2;

    std::array<float, 4> res_1{}, res_2{};
    result_1.store_aligned(res_1.data());
    result_2.store_aligned(res_2.data());


    // set values
    x(0) = res_1[0];
    x(1) = res_2[1];
    x(2) = res_1[2];
    x(3) = res_2[3];
    x(4) = res_1[1];
    x(5) = res_2[0];
    x(6) = res_1[3];
    x(7) = res_2[2];
}


void calc_aan_dct(Block8x8 block, BlockT8x8 blockT, const float *a, const float *s) {
    // row-wise transformation -> transposed due to column major nature of Eigen (trying to prevent cache misses)
    for(int i=0; i<8; i++) {
        aan_core(blockT.col(i), a, s);
    }

    // column-wise transformation
    for(int i=0; i<8; i++) {
        aan_core(block.col(i), a, s);
    }
}

void aan_dct(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &channel) {
    float a[5], s[8];
    calc_params_a(a);
    calc_params_s(s);

    Eigen::Transpose<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>> channelTransposed = channel.transpose();

    std::size_t amountTiles = (channel.cols() / 8) * (channel.rows() / 8);
    int tile_width = static_cast<int>(channel.cols()) / 8;
    BS::thread_pool pool;

    const BS::multi_future<void> multiFuture = pool.submit_loop(std::size_t{0}, amountTiles, [&channel, &a, &s, &tile_width, &channelTransposed](const std::size_t i) {
        int row_coord = (static_cast<int>(i) / tile_width) * 8;
        int col_coord = (static_cast<int>(i) % tile_width) * 8;

        Eigen::Block<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>, 8, 8> block = channel.block<8, 8>(row_coord, col_coord);
        Eigen::Block<Eigen::Transpose<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>>, 8, 8> blockTransposed = channelTransposed.block<8, 8>(col_coord, row_coord);

        calc_aan_dct(block, blockTransposed, a, s);
    });

    multiFuture.wait();
}
#endif


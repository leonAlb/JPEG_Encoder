// Run-length encoding (RLE) and Huffman bitstream generation helpers.
#ifndef JPEG_PROJEKT_RLE_H
#define JPEG_PROJEKT_RLE_H

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <type_traits>
#include <iostream>

#include "Image2D.h"
#include "Quantization.h"
#include "Huffman.h"


struct bit_stream_data {
    uint32_t huffman_code;
    int huffman_code_length;
    uint32_t code;
    int code_length;
    bool is_eob = false;
};


// ---------- Forward declarations ----------
inline std::vector<bit_stream_data> generate_luminance_bit_stream_data(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &quantized_channel_y, int sx, int sy);
inline std::vector<bit_stream_data> generate_chrominance_bit_stream_data(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &quantized_channel_cb,Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &quantized_channel_cr);
inline std::vector<bit_stream_data> calc_dc_luminance_bit_stream_representation(const std::vector<std::vector<int>> &zigzagged_blocks, Builder<int> &builder);
inline std::vector<bit_stream_data> calc_dc_chrominance_bit_stream_representation(const std::vector<std::vector<int>> &zigzagged_blocks_cb,const std::vector<std::vector<int>> &zigzagged_blocks_cr,Builder<int> &builder);
inline int get_category(const int &elem);
inline int get_bit_representation(const int &elem);
inline std::vector<std::byte> get_ac_symbols_for_huffman(const std::vector<std::vector<int>> &cat_bit_reps);
template<typename T>
std::unordered_map<T, int> get_frequency_map(const std::vector<T> &symbols, int step);
inline std::vector<std::vector<int>> combine_ac_cat_bit_reps_of_channels(const std::vector<std::vector<int>> &cat_bit_reps_cb,const std::vector<std::vector<int>> &cat_bit_reps_cr);
inline void push_ac_bsr(const std::vector<std::pair<uint32_t, int>> &huffman_encoding,const std::vector<std::pair<uint32_t, int>> &value_encoding,const std::vector<std::byte> &symbols,std::vector<bit_stream_data> &res);
inline std::vector<int> get_dc_differences(const std::vector<int> &dc_coefficients);
inline std::vector<std::vector<int>> get_zigzagged_blocks(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &quantized_channel);
inline std::vector<int> get_dc_coeffs(const std::vector<std::vector<int>> &zigzagged_blocks);
inline std::vector<int> combine_dc_cat_bit_reps_of_channels(const std::vector<int> &cat_bit_rep_cb,const std::vector<int> &cat_bit_rep_cr);
inline void push_dc_bsr(const std::vector<int> &dc_cat_bit_rep,std::vector<bit_stream_data> &dc_bit_stream_rep,const Builder<int> &builder);
template<typename T, typename M>
void build_huffman_tree(Builder<T> &builder, bool print_tree, bool print_codes, const M &frequency_map);
inline std::vector<std::vector<int>> calc_ac_rle(const std::vector<std::vector<int>> &zigzagged_blocks);
inline std::vector<std::vector<int>> calc_ac_category_bit_representation(const std::vector<std::vector<int>> &rle_encoded_blocks);
inline std::vector<std::pair<uint32_t, int>> calc_huffman_encoding(const std::vector<std::vector<int>> &cat_bit_reps,Builder<std::byte> &builder);
inline std::vector<std::pair<uint32_t, int>> calc_ac_category_values_encoding(const std::vector<std::vector<int>> &cat_bit_reps);


inline Builder<std::byte> y_ac_tree;
inline Builder<int> y_dc_tree;
inline Builder<std::byte> c_ac_tree;
inline Builder<int> c_dc_tree;


static std::vector<std::vector<bit_stream_data>> build_blocks_from_dc_and_ac(const std::vector<bit_stream_data>& dc,const std::vector<bit_stream_data>& ac_flat,const std::vector<size_t>& ac_counts_per_block)
{
    std::vector<std::vector<bit_stream_data>> blocks;
    blocks.reserve(dc.size());

    size_t ac_idx = 0;
    for (size_t b = 0; b < dc.size(); ++b)
    {
        std::vector<bit_stream_data> block;
        block.reserve(1 + ac_counts_per_block[b]);

        // DC
        block.push_back(dc[b]);

        // AC dazu
        for (size_t j = 0; j < ac_counts_per_block[b]; ++j)
        {
            block.push_back(ac_flat[ac_idx++]);
        }
        blocks.push_back(std::move(block));
    }
    return blocks;
}


static std::vector<std::vector<bit_stream_data>> generate_luminance_blocks(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &quantized_channel_y,
    int sx, int sy)
{
    const auto zigzagged_blocks = get_zigzagged_blocks_mcu(quantized_channel_y, sx, sy);

    const auto dc = calc_dc_luminance_bit_stream_representation(zigzagged_blocks, y_dc_tree);

    const auto rle = calc_ac_rle(zigzagged_blocks);
    const auto cbr = calc_ac_category_bit_representation(rle);

    std::vector<size_t> ac_counts;
    ac_counts.reserve(cbr.size());
    for (const auto& block : cbr) ac_counts.push_back(block.size() / 3);

    const auto huff = calc_huffman_encoding(cbr, y_ac_tree); // huffman code + length
    const auto vals = calc_ac_category_values_encoding(cbr); // bits + length

    std::vector<bit_stream_data> ac_flat;
    ac_flat.reserve(huff.size());
    const auto symbols = get_ac_symbols_for_huffman(cbr);
    push_ac_bsr(huff, vals, symbols, ac_flat);

    return build_blocks_from_dc_and_ac(dc, ac_flat, ac_counts);
}

static std::vector<std::vector<bit_stream_data>> generate_chrominance_blocks(
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &quantized_channel_cb,
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &quantized_channel_cr)
{
    const auto zig_cb = get_zigzagged_blocks(quantized_channel_cb);
    const auto zig_cr = get_zigzagged_blocks(quantized_channel_cr);

    // DC
    const auto dc = calc_dc_chrominance_bit_stream_representation(zig_cb, zig_cr, c_dc_tree);

    // AC
    const auto rle_cb = calc_ac_rle(zig_cb);
    const auto cbr_cb = calc_ac_category_bit_representation(rle_cb);

    const auto rle_cr = calc_ac_rle(zig_cr);
    const auto cbr_cr = calc_ac_category_bit_representation(rle_cr);

    const auto combined = combine_ac_cat_bit_reps_of_channels(cbr_cb, cbr_cr);

    std::vector<size_t> ac_counts;
    ac_counts.reserve(combined.size());
    for (const auto& block : combined) ac_counts.push_back(block.size() / 3);

    const auto huff = calc_huffman_encoding(combined, c_ac_tree);
    const auto vals = calc_ac_category_values_encoding(combined);

    std::vector<bit_stream_data> ac_flat;
    ac_flat.reserve(huff.size());
    const auto symbols = get_ac_symbols_for_huffman(combined);
    push_ac_bsr(huff, vals, symbols, ac_flat);

    return build_blocks_from_dc_and_ac(dc, ac_flat, ac_counts);
}


// ============================================================================
//  Bitstream assembly

// Note: Channels must be quantized before bitstream generation.
inline std::vector<bit_stream_data> generate_bit_stream_data_for_image(Image2D<float> &img)
{
    std::vector<bit_stream_data> out;

    auto y_blocks = generate_luminance_blocks(img.getChannel1(), img.sx, img.sy);
    auto c_blocks = generate_chrominance_blocks(img.getChannel2(), img.getChannel3()); // cb1,cr1,cb2,cr2,...

    // Calculate MCU
    auto &Y = img.getChannel1();
    const int amountYBlocksX = Y.cols() / 8;
    const int amountYBlocksY = Y.rows() / 8;

    const int yBlocks = img.sx * img.sy; // Number of Y blocks per MCU
    const int mcuX = amountYBlocksX / img.sx;
    const int mcuY = amountYBlocksY / img.sy;
    const size_t numMCU = static_cast<size_t>(mcuX) * static_cast<size_t>(mcuY);

    for (size_t mcu = 0; mcu < numMCU; ++mcu) {
        for (int k = 0; k < yBlocks; ++k) {
            const auto &blockY = y_blocks[mcu * static_cast<size_t>(yBlocks) + static_cast<size_t>(k)];
            out.insert(out.end(), blockY.begin(), blockY.end());
        }
        const auto &blockCb = c_blocks[mcu * 2 + 0];
        const auto &blockCr = c_blocks[mcu * 2 + 1];
        out.insert(out.end(), blockCb.begin(), blockCb.end());
        out.insert(out.end(), blockCr.begin(), blockCr.end());
    }

    return out;
}

// ============================================================================
//  AC PART

inline std::vector<std::vector<int>> calc_ac_rle(const std::vector<std::vector<int>> &zigzagged_blocks)
{
    std::vector<std::vector<int>> rle_encoded_blocks(zigzagged_blocks.size());
    int index = 0;

    for (const auto &zigzagged_block : zigzagged_blocks) {
        std::vector<int> rle_encoded_block;
        rle_encoded_block.reserve(24);

        int eob = 0;
        for (int i = 63; i > 0; --i) {
            if (zigzagged_block[i] == 0) eob++;
            else break;
        }

        const int last = 63 - eob;

        int count_zeros = 0;

        for (int i = 1; i <= last; ++i)
            {
            const int element = zigzagged_block[i];

            if (element == 0) {
                if (count_zeros < 15) {
                    count_zeros++;
                } else {
                    // 16 UNOS => (15,0)
                    rle_encoded_block.push_back(15);
                    rle_encoded_block.push_back(0);
                    count_zeros = 0;
                }
            } else {
                rle_encoded_block.push_back(count_zeros);
                rle_encoded_block.push_back(element);
                count_zeros = 0;
            }
        }

        if (eob > 0) {
            rle_encoded_block.push_back(0);
            rle_encoded_block.push_back(0);
        }

        rle_encoded_blocks[index++] = std::move(rle_encoded_block);
    }

    return rle_encoded_blocks;
}


inline std::vector<std::vector<int>> calc_ac_category_bit_representation(const std::vector<std::vector<int>> &rle_encoded_blocks)
{
    std::vector<std::vector<int>> cat_bit_reps(rle_encoded_blocks.size());
    int index = 0;

    for (const auto &rle_encoded_block : rle_encoded_blocks) {
        std::vector<int> cat_bit_rep;
        cat_bit_rep.reserve(rle_encoded_block.size() / 2 * 3);

        for (int i = 0; i < (int)rle_encoded_block.size(); i += 2) {
            const int zero_run  = rle_encoded_block[i];
            const int value     = rle_encoded_block[i + 1];

            if (value != 0) {
                cat_bit_rep.push_back(zero_run);
                cat_bit_rep.push_back(get_category(value));
                cat_bit_rep.push_back(get_bit_representation(value));
            } else if (zero_run == 0) {
                // EOB
                cat_bit_rep.push_back(0);
                cat_bit_rep.push_back(0);
                cat_bit_rep.push_back(0);
            } else {
                // (15,0)
                cat_bit_rep.push_back(zero_run);
                cat_bit_rep.push_back(0);
                cat_bit_rep.push_back(0);
            }
        }

        cat_bit_reps[index++] = std::move(cat_bit_rep);
    }

    return cat_bit_reps;
}


inline std::vector<std::pair<uint32_t, int>> calc_huffman_encoding(
    const std::vector<std::vector<int>> &cat_bit_reps,
    Builder<std::byte> &builder)
{
    std::vector<std::pair<uint32_t, int>> res;

    const std::vector<std::byte> symbols = get_ac_symbols_for_huffman(cat_bit_reps);
    const std::unordered_map<std::byte, int> symbol_frequency_map = get_frequency_map<std::byte>(symbols, 1);
    res.reserve(symbols.size());

    build_huffman_tree<std::byte, std::unordered_map<std::byte, int>>(builder, false, false, symbol_frequency_map);

    for (auto &element : symbols) {
        auto search = builder.codeMap.find(element);
        if (search == builder.codeMap.end()) {
            std::cerr << "Element not found in builder.codeMap\n";
            std::exit(1);
        }
        res.push_back(search->second);
    }

    return res;
}


inline std::vector<std::pair<uint32_t, int>> calc_ac_category_values_encoding(const std::vector<std::vector<int>> &cat_bit_reps)
{
    std::vector<std::pair<uint32_t, int>> res;
    size_t reservation = 0;

    for (const auto &cbr : cat_bit_reps) reservation += cbr.size();
    res.reserve(reservation);

    for (const auto &cat_bit_rep : cat_bit_reps) {
        for (int i = 0; i + 2 < (int)cat_bit_rep.size(); i += 3) {
            const int category = cat_bit_rep[i + 1];
            const int value = cat_bit_rep[i + 2];
            res.push_back(std::make_pair((uint32_t)value, category));
        }
    }

    return res;
}


// ------------------------------ AC Helper functions ------------------------------

inline std::vector<std::byte> get_ac_symbols_for_huffman(const std::vector<std::vector<int>> &cat_bit_reps)
{
    std::vector<std::byte> res;
    size_t reservation = 0;

    for (const auto &cbr : cat_bit_reps) reservation += cbr.size();
    res.reserve(reservation);

    for (const auto &cat_bit_rep : cat_bit_reps) {
        for (int i = 0; i < (int)cat_bit_rep.size(); i += 3) {
            const int zero_part = cat_bit_rep[i];
            const int category = cat_bit_rep[i + 1];

            auto b = static_cast<std::byte>(zero_part);
            b = b << 4;
            const auto c = static_cast<std::byte>(category);
            b |= c;

            res.push_back(b);
        }
    }
    return res;
}


inline std::vector<std::vector<int>> combine_ac_cat_bit_reps_of_channels(
    const std::vector<std::vector<int>> &cat_bit_reps_cb,
    const std::vector<std::vector<int>> &cat_bit_reps_cr)
{
    std::vector<std::vector<int>> res;
    res.reserve(cat_bit_reps_cb.size() + cat_bit_reps_cr.size());

    for (int i = 0; i < (int)cat_bit_reps_cb.size(); i++) {
        res.push_back(cat_bit_reps_cb[i]);
        res.push_back(cat_bit_reps_cr[i]);
    }

    return res;
}


inline void push_ac_bsr(
    const std::vector<std::pair<uint32_t, int>> &huffman_encoding,
    const std::vector<std::pair<uint32_t, int>> &value_encoding,
    const std::vector<std::byte> &symbols,
    std::vector<bit_stream_data> &res)
{
    if (huffman_encoding.size() != value_encoding.size() || symbols.size() != huffman_encoding.size()) {
        std::cerr << "push_ac_bsr: size mismatch\n";
        std::exit(1);
    }

    for (int i = 0; i < (int)huffman_encoding.size(); i++) {
        const auto &h = huffman_encoding[i];
        const auto &v = value_encoding[i];

        bit_stream_data bsd{
            h.first,
            h.second,
            v.first,
            v.second,
            symbols[i] == std::byte{0x00}
        };

        res.push_back(bsd);
    }
}


// ============================================================================
//  DC PART

inline std::vector<int> calc_dc_category_values_encoding(const std::vector<int> &dc_differences)
{
    std::vector<int> cat_bit_rep;
    cat_bit_rep.reserve(dc_differences.size() * 2);

    for (int i = 0; i < (int)dc_differences.size(); i++) {
        int category = get_category(dc_differences[i]);
        int encoding = get_bit_representation(dc_differences[i]);

        cat_bit_rep.push_back(category);
        cat_bit_rep.push_back(encoding);
    }

    return cat_bit_rep;
}


inline std::vector<bit_stream_data> calc_dc_luminance_bit_stream_representation(
    const std::vector<std::vector<int>> &zigzagged_blocks,
    Builder<int> &builder)
{
    std::vector<bit_stream_data> dc_bit_stream_rep;

    const std::vector<int> dc_coeffs = get_dc_coeffs(zigzagged_blocks);
    const std::vector<int> dc_differences = get_dc_differences(dc_coeffs);
    dc_bit_stream_rep.reserve(dc_differences.size() * 2);

    const auto dc_cat_bit_rep = calc_dc_category_values_encoding(dc_differences);

    const auto dc_frequency_map = get_frequency_map<int>(dc_cat_bit_rep, 2);

    build_huffman_tree<int, std::unordered_map<int, int>>(builder, true, true, dc_frequency_map);

    push_dc_bsr(dc_cat_bit_rep, dc_bit_stream_rep, builder);

    return dc_bit_stream_rep;
}


inline std::vector<bit_stream_data> calc_dc_chrominance_bit_stream_representation(
    const std::vector<std::vector<int>> &zigzagged_blocks_cb,
    const std::vector<std::vector<int>> &zigzagged_blocks_cr,
    Builder<int> &builder)
{
    std::vector<bit_stream_data> dc_bit_stream_rep;

    const auto dc_coeffs_cb = get_dc_coeffs(zigzagged_blocks_cb);
    const auto dc_differences_cb = get_dc_differences(dc_coeffs_cb);
    const auto cat_bit_rep_cb = calc_dc_category_values_encoding(dc_differences_cb);

    const auto dc_coeffs_cr = get_dc_coeffs(zigzagged_blocks_cr);
    const auto dc_differences_cr = get_dc_differences(dc_coeffs_cr);
    const auto cat_bit_rep_cr = calc_dc_category_values_encoding(dc_differences_cr);

    const auto combined_cat_bit_reps = combine_dc_cat_bit_reps_of_channels(cat_bit_rep_cb, cat_bit_rep_cr);
    const auto frequency_map = get_frequency_map<int>(combined_cat_bit_reps, 2);

    build_huffman_tree<int, std::unordered_map<int, int>>(builder, false, false, frequency_map);

    push_dc_bsr(combined_cat_bit_reps, dc_bit_stream_rep, builder);

    return dc_bit_stream_rep;
}


// ------------------------------ DC Helper functions ------------------------------

inline std::vector<int> get_dc_differences(const std::vector<int> &dc_coefficients)
{
    std::vector<int> dc_differences;
    dc_differences.reserve(dc_coefficients.size());

    dc_differences.push_back(dc_coefficients[0]); // dc_-1 = 0
    for (int i = 1; i < (int)dc_coefficients.size(); i++) {
        dc_differences.push_back(dc_coefficients[i] - dc_coefficients[i - 1]);
    }

    return dc_differences;
}


inline std::vector<int> get_dc_coeffs(const std::vector<std::vector<int>> &zigzagged_blocks)
{
    std::vector<int> dc_coefficients;
    dc_coefficients.reserve(zigzagged_blocks.size());

    for (int i = 0; i < (int)zigzagged_blocks.size(); i++) {
        dc_coefficients.push_back(zigzagged_blocks[i][0]);
    }

    return dc_coefficients;
}


inline std::vector<int> combine_dc_cat_bit_reps_of_channels(
    const std::vector<int> &cat_bit_rep_cb,
    const std::vector<int> &cat_bit_rep_cr)
{
    std::vector<int> combined_dc_cat_bit_reps;
    combined_dc_cat_bit_reps.reserve(cat_bit_rep_cb.size() + cat_bit_rep_cr.size());

    for (int i = 0; i < (int)cat_bit_rep_cb.size(); i += 2) {
        combined_dc_cat_bit_reps.push_back(cat_bit_rep_cb[i]);
        combined_dc_cat_bit_reps.push_back(cat_bit_rep_cb[i + 1]);
        combined_dc_cat_bit_reps.push_back(cat_bit_rep_cr[i]);
        combined_dc_cat_bit_reps.push_back(cat_bit_rep_cr[i + 1]);
    }

    return combined_dc_cat_bit_reps;
}


inline void push_dc_bsr(
    const std::vector<int> &dc_cat_bit_rep,
    std::vector<bit_stream_data> &dc_bit_stream_rep,
    const Builder<int> &builder)
{
    for (int i = 0; i < (int)dc_cat_bit_rep.size(); i += 2) {
        auto category = dc_cat_bit_rep[i];
        const auto value = dc_cat_bit_rep[i + 1];

        auto search = builder.codeMap.find(category);
        if (search == builder.codeMap.end()) {
            std::cerr << "Element not found in builder.codeMap (DC)\n";
            std::exit(1);
        }

        bit_stream_data bsd{
            search->second.first,
            search->second.second,
            static_cast<uint32_t>(value),
            category,
            false
        };

        dc_bit_stream_rep.push_back(bsd);
    }
}


// ------------------------------ Allgemeine Helper functions ------------------------------

template<typename T, typename M>
void build_huffman_tree(Builder<T> &builder, const bool print_tree, const bool print_codes, const M &frequency_map)
{
    builder.codeMap.clear();

    // only 1 symbol
    if (frequency_map.size() == 1) {
        const T onlySym = frequency_map.begin()->first;
        builder.codeMap.emplace(onlySym, std::make_pair(0u, 1));

        if (print_tree) {
            std::cout << "\n--- Huffman Tree (single symbol) ---\n";
            if constexpr (std::is_same_v<T, std::byte>) {
                std::cout << "--- '" << std::hex
                          << (int)std::to_integer<unsigned char>(onlySym)
                          << std::dec << "':" << frequency_map.begin()->second << "\n";
            } else {
                std::cout << "--- '" << onlySym << "':" << frequency_map.begin()->second << "\n";
            }
            std::cout << "--------------------\n";
        }

        if (print_codes) {
            if constexpr (std::is_same_v<T, std::byte>) {
                std::cout << "Code for "
                          << std::hex << (int)std::to_integer<unsigned char>(onlySym)
                          << std::dec << ": 0 (0)\n";
            } else {
                std::cout << "Code for " << onlySym << ": 0 (0)\n";
            }
        }
        return;
    }

    // normal case (>=2)
    builder.set_frequency_map(frequency_map);
    builder.buildTree();
    builder.createBinaryTreeAs2DVector();
    builder.buildDepthLimitedTree(15);
    builder.buildRightGrowingTreeFrom2DVector();

    if (print_tree)  builder.printTreeHex();
    if (print_codes) builder.printCodes(builder.root, 0);
}


inline std::vector<std::vector<int>> get_zigzagged_blocks(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &quantized_channel)
{
    const std::size_t amountTiles = (quantized_channel.cols() / 8) * (quantized_channel.rows() / 8);
    const int tile_width = static_cast<int>(quantized_channel.cols()) / 8;

    std::vector<std::vector<int>> zigzagged_blocks(amountTiles);

#pragma omp parallel for collapse(1)
    for (int i = 0; i < (int)amountTiles; i++) {
        const int row_coord = i / tile_width * 8;
        const int col_coord = i % tile_width * 8;

        Eigen::Block<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>, 8, 8> block =
            quantized_channel.block<8, 8>(row_coord, col_coord);

        zigzagged_blocks[i] = zigzag_order_block(block);
    }

    return zigzagged_blocks;
}


template<typename T>
std::unordered_map<T, int> get_frequency_map(const std::vector<T> &symbols, const int step)
{
    std::unordered_map<T, int> res;
    for (int i = 0; i < (int)symbols.size(); i += step) {
        ++res[symbols[i]];
    }
    return res;
}


inline int get_category(const int &elem)
{
    if (elem == 0) return 0;
    return (int)std::floor(std::log2(std::abs(elem))) + 1;
}

inline int get_bit_representation(const int &elem)
{
    if (elem > 0) return elem;
    return ~std::abs(elem); // one's complement
}

#endif // JPEG_PROJEKT_RLE_H

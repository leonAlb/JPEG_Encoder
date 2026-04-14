#pragma once

#ifndef JPEG_PROJEKT_SAMPLING_H
#define JPEG_PROJEKT_SAMPLING_H

#include <Eigen/Core>
#include <thread>
#include <chrono>
#include <functional>

#if defined(__APPLE__)
#include <BS_thread_pool.hpp>
#include <future> // std::future (used by the thread pool API)
#endif

#include "Image2D.h"


template<typename T>
double get_local_avg(int, int, int, int, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &);

template<typename T>
void average_block(int, int, int, int, Image2D<T> &img);

template<typename T>
void local_avg_parallel(Image2D<T> &img);

template<typename T>
void local_avg_sampling(Image2D<T> &img);

/*
 * Downsamples the Cb and Cr colour channels of an image by keeping every
 * Cb_cols/Cb_rows-th and Cr_cols/Cr_rows-th pixel respectively.
 * A stride of 1 on both axes means no sampling is applied to that channel.
 */
template<typename T>
void sample_picture(int Cb_cols, int Cb_rows, int Cr_cols, int Cr_rows, Image2D<T> &img) {
    img.isSampled = true;

    if (!(Cb_cols == 1 && Cb_rows == 1)) {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> nc2 =
                img.getChannel2()(Eigen::seq(0, Eigen::last, Cb_rows), Eigen::seq(0, Eigen::last, Cb_cols)).eval();

        img.setChannel2(nc2);
    }

    if (!(Cr_cols == 1 && Cr_rows == 1)) {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> nc3 =
                img.getChannel3()(Eigen::seq(0, Eigen::last, Cr_rows), Eigen::seq(0, Eigen::last, Cr_cols)).eval();

        img.setChannel3(nc3);
    }
}


template<typename T>
void local_avg_sampling(Image2D<T> &img) {
    int w = img.sx;
    int h = img.sy;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int row = 0; row < img.height; row += h) {
        for (int col = 0; col < img.width; col += w) {
            average_block(row, h, col, w, img);
        }
    }

    sample_picture(w, h, w, h, img);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::cout << "local_avg_sampling: " << ms << " ms\n";
}

template<typename T>
void local_avg_parallel(Image2D<T> &img) {
    int w = img.sx; // step width
    // int w = 2; // step width
    int h = img.sy; // step height
    // int h = 2; // step height

    auto start_time = std::chrono::high_resolution_clock::now();

    int valid_height = img.height;
    int valid_width = img.width;

#if defined(__APPLE__)
    std::size_t amountTiles = (valid_height / h) * (valid_width / w);
    BS::thread_pool pool;
    const BS::multi_future<void> multiFuture = pool.submit_loop(std::size_t{0}, amountTiles, [&img, h, w](const std::size_t i) {
        int tile_width = img.width / img.sx; //2; //img.sx; // TODO: check if not img.width / tile_width || img.width / amountTiles
        int row_coord = (static_cast<int>(i) / tile_width) * img.sx; //2;
        int col_coord = (static_cast<int>(i) % tile_width) * img.sy; //2;

        average_block(row_coord, h, col_coord, w, img);
    });

    multiFuture.wait();

#else
    #pragma omp parallel for collapse(2)
    for (int row = 0; row < valid_height; row += h) {
        for (int col = 0; col < valid_width; col += w) {
            average_block(row, h, col, w, img);
        }
    }
#endif

    sample_picture(w, h, w, h, img);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    std::cout << "local_avg_parallel: " << ms << " ms\n";
}

template<typename T>
void average_block(int row, int h, int col, int w, Image2D<T> &img) {
    img.setColourPixel(col, row,
                       get_local_avg(row, col, h, w, img.getChannel2()),
                       get_local_avg(row, col, h, w, img.getChannel3())
    );
}

template<typename T>
double get_local_avg(int row, int col, int h, int w, Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &cc) {
    double sum = 0.0;

    for (int i = row; i < (row + h); i++) {
        for (int j = col; j < (col + w); j++) {
            sum += cc(i, j);
        }
    }

    return (sum / (h * w));
}

#endif //JPEG_PROJEKT_SAMPLING_H

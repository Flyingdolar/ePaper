#pragma once

#ifndef HALFTONE_HPP
#define HALFTONE_HPP

#include "Functions.hpp"

namespace halftone {

extern std::string verbosePath;

// Threshold Map Matrix for Dithering
const cv::Mat1b tMap2 = (cv::Mat1b(2, 2) << 0, 2, 3, 1);
const cv::Mat1b tMap4 = (cv::Mat1b(4, 4) << 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5);
const cv::Mat1b tMap8 = (cv::Mat1b(8, 8) << 0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26,
                         12, 44, 4, 36, 14, 46, 6, 38, 60, 28, 52, 20, 62, 30, 54, 22,
                         3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25,
                         15, 47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21);

// Error Diffusion Kernels
const cv::Mat1b kFloydSteinberg = (cv::Mat1b(3, 3) << 0, 0, 7, 3, 5, 1, 0, 0, 0);
const cv::Mat1b kJJN = (cv::Mat1b(3, 5) << 0, 0, 0, 7, 5, 3, 5, 7, 5, 3, 1, 3, 5, 3, 1);

/**
 * @brief Direct Binary Search (DBS) Halftoning
 * @param img Input image (Single Channel, 0-1, float)
 * @param initImg Initial image for DBS (default: empty->random)
 * @param kernelSize Kernel size for DBS
 * @param sigma Sigma value for Point Spread Function (PSF) (default: 1.0)
 * @param iters Number of iterations for DBS (default: 10)
 * @param verbose Verbose mode (default: false)
 * @return Halftoned image (Single Channel, 0-1, float)
 *
 * @note If initImg is empty, random initialization is used.
 */
cv::Mat1f DBS(const cv::Mat1f img, cv::Mat1f initImg, int kernelSize, float sigma = 1.0f, int iters = 10, bool verbose = false, std::string savePath = "");
inline cv::Mat1f DBS(const cv::Mat1f grayImg, int kernelSize, float sigma = 1.0f, int iters = 10, bool verbose = false, std::string savePath = "") {
    cv::Mat1f resImg = cv::Mat1f::zeros(grayImg.size());
    for (int row = 0; row < grayImg.rows; row++)  // Random Initialization
        for (int col = 0; col < grayImg.cols; col++) resImg.at<float>(row, col) = (rand() % 2 == 0) ? 0 : 1;
    return DBS(grayImg, resImg, kernelSize, sigma, iters, verbose, savePath);
}

/**
 * @brief Halftone by Dithering
 * @param grayImg Input image (Single Channel, 0-1, float)
 * @param kernelSize Kernel size for Dithering (2, 4, 8)
 * @param verbose Verbose mode (default: false)
 * @return Halftoned image (Single Channel, 0-1, float)
 */
cv::Mat1f Dither(const cv::Mat1f grayImg, int kernelSize = 2, bool verbose = false);

/**
 * @brief Halftone by Error Diffusion
 * @param grayImg Input image (Single Channel, 0-1, float)
 * @param kernelSize Kernel size for Error Diffusion (3: Floyd-Steinberg, 5: JJN)
 * @param verbose Verbose mode (default: false)
 * @return Halftoned image (Single Channel, 0-1, float)
 */
cv::Mat1f ErrDiff(const cv::Mat1f grayImg, int kernelSize = 3, bool verbose = false);

namespace detail {
cv::Mat1f getGSF(int kSize, float sigma);

float deltaLpErr(const cv::Mat1f lpErrImg, cv::Vec3i posCent, cv::Vec3i posSwap, int kSize, const cv::Mat1f gskMat);

cv::Mat1f altLpErr(const cv::Mat1f lpErrImg, cv::Vec3i posPix, int kernelSize, const cv::Mat1f gskMat);

cv::Mat3f viewErr(cv::Mat1f errImg);

}  // namespace detail
}  // namespace halftone

#endif  // HALFTONE_HPP

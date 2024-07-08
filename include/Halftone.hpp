#pragma once

#ifndef HALFTONE_HPP
#define HALFTONE_HPP

#include "Functions.hpp"

namespace halftone {

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
 * @param kernelSize Kernel size for DBS
 * @param sigma Sigma value for Point Spread Function (PSF)
 * @param iters Number of iterations for DBS
 * @param verbose Verbose mode (default: false)
 * @return Halftoned image (Single Channel, 0-1, float)
 */
cv::Mat1f DBS(const cv::Mat1f img, int kernelSize, float sigma, int iters, bool verbose = false);

/**
 * @brief Random Tiled Block Direct Binary Search (RTBDBS) Halftoning
 * @param img Input image (Single Channel, 0-1, float)
 * @param kernelSize Kernel size for DBS
 * @param sigma Sigma value for Point Spread Function (PSF)
 * @param iters Number of iterations for DBS
 * @param verbose Verbose mode (default: false)
 * @return Halftoned image (Single Channel, 0-1, float)
 */
cv::Mat1f RTBDBS(const cv::Mat1f img, int kernelSize, float sigma, int iters, bool verbose = false);

/**
 * @brief Halftone by Dithering
 * @param grayImg Input image (Single Channel, 0-1, float)
 * @param kernelSize Kernel size for Dithering (2, 4, 8)
 * @param verbose Verbose mode (default: false)
 * @return Halftoned image (Single Channel, 0-1, float)
 */
cv::Mat1f Dither(const cv::Mat1f grayImg, int kernelSize, bool verbose = false);

namespace detail {

cv::Mat1f altLpErr(const cv::Mat1f lpErrImg, std::vector<cv::Vec3i> pos, int kernelSize, const cv::Mat1f gskMat2D);

cv::Vec2i getMinEPos(const cv::Mat1f resImg, const cv::Mat1f lpErrImg, cv::Vec2i centPos, int kernelSize, const cv::Mat1f gskMat2D);

cv::Mat3f viewErr(cv::Mat1f errImg);

}  // namespace detail

}  // namespace halftone

#endif  // HALFTONE_HPP

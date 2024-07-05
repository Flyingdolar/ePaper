#pragma once

#ifndef HALFTONE_HPP
#define HALFTONE_HPP

#include "Functions.hpp"

namespace halftone {

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
 * @brief Random Threshold Binary Direct Binary Search (RTBDBS) Halftoning
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
 * @param kernelSize Kernel size for Dithering
 * @param verbose Verbose mode (default: false)
 * @return Halftoned image (Single Channel, 0-1, float)
 */
cv::Mat1f Dither(const cv::Mat1f grayImg, int kernelSize, bool verbose = false);

}  // namespace halftone

#endif  // HALFTONE_HPP

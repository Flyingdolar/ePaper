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
 * @param kernelSize Kernel size for Point Spread Function (PSF) (default: 3)
 * @param sigma Sigma value for Point Spread Function (PSF) (default: 1.0)
 * @param iters Number of iterations for DBS (default: 10)
 * @param verbose Verbose mode (default: false)
 * @return Halftoned image (Single Channel, 0-1, float)
 *
 * @note If initImg is empty, random initialization is used.
 */
cv::Mat1f DBS(const cv::Mat1f img, cv::Mat1f initImg, int kernelSize = 3, float sigma = 1.0f, int iters = 10, bool verbose = false, std::string savePath = "");
cv::Mat1f DBS(const cv::Mat1f grayImg, int kernelSize = 3, float sigma = 1.0f, int iters = 10, bool verbose = false, std::string savePath = "");

/**
 * @brief Random Tiled Blocks Direct Binary Search (RTB-DBS) Halftoning
 * @param img Input image (Single Channel, 0-1, float)
 * @param initImg Initial image for RTB-DBS (default: empty->random)
 * @param blkMap Block map for RTB-DBS (Single Channel, 0-blkSize[0]*blkSize[1]-1, int)
 * @param kernelSize Kernel size for Point Spread Function (PSF) (default: 3)
 * @param sigma Sigma value for Point Spread Function (PSF) (default: 1.0)
 * @param iters Number of iterations for RTB-DBS (default: 10)
 * @param verbose Verbose mode (default: false)
 * @return Halftoned image (Single Channel, 0-1, float)
 *
 * @note If initImg is empty, random initialization is used.
 */
cv::Mat1f RTBDBS(const cv::Mat1f grayImg, cv::Mat1f initImg, cv::Mat1i blkMap, int kernelSize = 3, float sigma = 1.0f, int iters = 10, bool verbose = false, std::string savePath = "");
cv::Mat1f RTBDBS(const cv::Mat1f grayImg, cv::Mat1i blkMap, int kernelSize = 3, float sigma = 1.0f, int iters = 10, bool verbose = false, std::string savePath = "");
cv::Mat1f RTBDBS(const cv::Mat1f grayImg, cv::Mat1f initImg, int blkSize, int kernelSize = 3, float sigma = 1.0f, int iters = 10, bool verbose = false, std::string savePath = "");
cv::Mat1f RTBDBS(const cv::Mat1f grayImg, int blkSize, int kernelSize = 3, float sigma = 1.0f, int iters = 10, bool verbose = false, std::string savePath = "");

/**
 * @brief Halftone by Dithering
 * @param grayImg Input image (Single Channel, 0-1, float)
 * @param kernelSize Kernel size for Dithering (2, 4, 8)
 * @param verbose Verbose mode (default: false)
 * @return Halftoned image (Single Channel, 0-1, float)
 */
cv::Mat1f Dither(const cv::Mat1f grayImg, int kernelSize = 2, bool verbose = false);
/**
 * @brief Halftone by Dithering
 * @param grayImg Input image (Single Channel, 0-1, float)
 * @param dithMap Dithering map (Same size as input image, 0-1, float)
 * @param verbose Verbose mode (default: false)
 * @return Halftoned image (Single Channel, 0-1, float)
 * @note It would dithering by thresholding with dithMap.
 */
cv::Mat1f Dither(const cv::Mat1f grayImg, cv::Mat1f dithMap, bool verbose = false);

/**
 * @brief Halftone by Error Diffusion
 * @param grayImg Input image (Single Channel, 0-1, float)
 * @param kernelSize Kernel size for Error Diffusion (3: Floyd-Steinberg, 5: JJN)
 * @param verbose Verbose mode (default: false)
 * @return Halftoned image (Single Channel, 0-1, float)
 */
cv::Mat1f ErrDiff(const cv::Mat1f grayImg, int kernelSize = 3, bool verbose = false);

/**
 * @brief Void & Cluster Dither Array Generation
 * @param img Input image (Single Channel, 0-1, float)
 * @param blkSize Block size for Void & Cluster Dithering
 * @param kernelSize Kernel size for Void & Cluster Dithering (default: 3)
 * @param sigma Sigma value for Point Spread Function (PSF) (default: 1.0)
 * @param normalize Normalize the dither array to 0-1 (default: false)
 * @param verbose Verbose mode (default: false)
 * @note If not normalized, the dither array would be 0-(blkSize[0]*blkSize[1]-1)
 * @return Dither array for Void & Cluster Dithering
 */
cv::Mat1f VoidCluster(const cv::Mat1f binImg, int kernelSize = 3, float sigma = 1.0, bool normalize = false, bool verbose = false);

/**
 * @brief Generate Random Binary Image
 * @param imgSize Size of the binary image (height, width)
 * @return Random binary image (Single Channel, 0-1, float)
 */
cv::Mat1f getRandBin(cv::Vec2i imgSize);

namespace detail {
cv::Mat1f getGSF(int kSize, float sigma);

cv::Mat1f VCFilter(const cv::Mat1f blkImg, int kSize, float sigma);

cv::Mat1i VCP1(const cv::Mat1f bkImg, int kSize, float sigma);

void VCP2(cv::Mat1f bkImg, cv::Mat1i rkImg, int kSize, float sigma);

void VCP3(const cv::Mat1f bkImg, cv::Mat1i rkImg, int kSize, float sigma);

float deltaLpErr(const cv::Mat1f lpErrImg, cv::Vec3i posCent, cv::Vec3i posSwap, int kSize, const cv::Mat1f gskMat);

cv::Mat1f altLpErr(const cv::Mat1f lpErrImg, cv::Vec3i posPix, int kernelSize, const cv::Mat1f gskMat);

cv::Mat3f viewErr(cv::Mat1f errImg);

}  // namespace detail
}  // namespace halftone

#endif  // HALFTONE_HPP

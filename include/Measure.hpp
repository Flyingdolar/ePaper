#pragma once

#include "Functions.hpp"

namespace measure {

/**
 * @brief Measure PSNR between two images
 * @param testImg Test image (Single Channel, 0-1, float)
 * @param refImg Reference image (Single Channel, 0-1, float)
 * @return PSNR value
 */
double PSNR(const cv::Mat1f testImg, const cv::Mat1f refImg);

/**
 * @brief Measure SSIM between two images
 * @param testImg Test image (Single Channel, 0-1, float)
 * @param refImg Reference image (Single Channel, 0-1, float)
 * @param kSize Kernel Size (Default: 11)
 * @param sigma Sigma Value (Default: 1.5)
 * @param cst1 Constant C1 (Default: 6.5025)
 * @param cst2 Constant C2 (Default: 58.5225)
 * @return SSIM value
 */
double SSIM(const cv::Mat1f testImg, const cv::Mat1f refImg, int kSize = 11, float sigma = 1.5, double cst1 = 6.5025, double cst2 = 58.5225);

}  // namespace measure

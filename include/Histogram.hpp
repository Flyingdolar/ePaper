#pragma once

#ifndef HISTOGRAM_HPP
#define HISTOGRAM_HPP

#include <opencv2/opencv.hpp>
#include <vector>

namespace histogram {

/**
 * @brief Create a histogram from an image
 * @param img Image
 * @param bins Number of bins
 * @param range Range of image values
 * @return Histogram (vector of integers)
 */
std::vector<int> getHist(cv::Mat img, int bins, std::pair<float, float> range);

/**
 * @brief Create a histogram from an image with a mask
 * @param img Image
 * @param bins Number of bins
 * @param range Range of image values
 * @param mask Mask(0 for ignore, 1 for include)
 * @return Histogram (vector of integers)
 */
std::vector<int> getHist(cv::Mat img, int bins, std::pair<float, float> range, cv::Mat mask);

/**
 * @brief Create a Cumulative Distribution Function (CDF) from a histogram
 * @param hist Histogram
 * @return CDF (vector of floats)
 */
std::vector<int> getCDF(std::vector<int> hist);

/**
 * @brief CDF but set min and max ratio limits
 * @param hist Histogram
 * @param min_rate Minimum ratio
 * @param max_rate Maximum ratio
 * @param min_inc Minimum increase
 * @param max_inc Maximum increase
 * @return CDF (vector of floats)
 */
std::vector<int> limRateCDF(std::vector<int> hist, float min_rate, float max_rate, float min_inc, float max_inc);

/**
 * @brief Equalize an image using a CDF
 * @param img Image
 * @param cdf CDF
 * @param bins Number of bins
 * @param range Range of image values
 * @return Equalized image
 */
cv::Mat equalize(cv::Mat img, std::vector<int> cdf, int bins, std::pair<float, float> range);

}  // namespace histogram

#endif  // HISTOGRAM_HPP

#pragma once

#ifndef FILTER_HPP
#define FILTER_HPP

#include <opencv2/opencv.hpp>
#include <vector>

#include "SaveData.hpp"

namespace filter {

/**
 * @brief Do Convolution with the Image and Parallel Kernel
 * @param img Input Image (Single Channel)
 * @param kernel Kernel Matrix (Should be Parallel)
 * @return cv::Mat Convolved Image
 */
cv::Mat plConv(cv::Mat img, cv::Mat kernel);

/**
 * @brief Do Convolution with the Image and Kernel
 * @param img Input Image (Single Channel)
 * @param kernel Kernel Matrix
 * @return cv::Mat Convolved Image
 */
cv::Mat conv(cv::Mat img, cv::Mat kernel);

/**
 * @brief Apply Local Edge Preserving Filter to the Image
 * @param img Input Image (Single Channel)
 * @param kernelSize Size of the Kernel (Default: 3)
 * @param alpha Alpha Value (Default: 0.1)
 * @param beta Beta Value (Default: 1.0)
 * @param iters Number of Iterations (Default: 1)
 */
cv::Mat localEP(cv::Mat img, int kernelSize = 3, float alpha = 0.1, float beta = 1.0, int iters = 1);

/**
 * @brief Apply Bilateral Filter to the Image
 * @param img Input Image (Single Channel)
 * @param kernelSize Size of the Kernel (Default: 3)
 * @param sigmaS Spatial Distance Sigma (Default: 1.0)
 * @param sigmaR Intensity Distance Sigma (Default: 1.0)
 * @return cv::Mat Filtered Image
 */
cv::Mat bilateral(cv::Mat img, int kernelSize = 3, float sigmaS = 1.0, float sigmaR = 1.0);

/**
 * @brief Apply Fast Bilateral Filter to the Image
 * @param img Input Image (Single Channel)
 * @param kernelSize Size of the Kernel (Default: 3)
 * @param segment Number of Segments (Default: 8)
 * @param sigmaS Spatial Distance Sigma (Default: 1.0)
 * @param sigmaR Intensity Distance Sigma (Default: 1.0)
 * @return cv::Mat Filtered Image
 */
cv::Mat fastBilateral(cv::Mat img, int kernelSize = 3, int segment = 8, float sigmaS = 1.0, float sigmaR = 1.0);

/**
 * @brief Apply Similar Filter to the Image
 * @param img Input Image (Single Channel)
 * @param kernelSize Size of the Kernel (Default: 3)
 * @param lowRate Low Rate (Default: 0.9)
 * @param highRate High Rate (Default: 1.1)
 * @return cv::Mat Filtered Image
 */
cv::Mat similar(cv::Mat img, int kernelSize = 3, float lowRate = 0.9, float highRate = 1.1);

/**
 * @brief Apply Mean Filter to the Image
 * @param img Input Image (Single Channel)
 * @param kernelSize Size of the Kernel (Default: 3)
 * @return cv::Mat Filtered Image
 */
cv::Mat mean(cv::Mat img, int kernelSize = 3);

/**
 * @brief Apply Median Filter to the Image
 * @param img Input Image (Single Channel)
 * @param kernelSize Size of the Kernel (Default: 3)
 * @return cv::Mat Filtered Image
 */
cv::Mat median(cv::Mat img, int kernelSize = 3);

/**
 * @brief Apply Sub Window Box Filter to the Image
 * @param img Input Image (Single Channel)
 * @param kernelSize Size of the Kernel (Default: 3)
 * @param iterations Number of Iterations (Default: 1)
 * @return cv::Mat Filtered Image
 */
cv::Mat subWBox(cv::Mat img, int kernelSize = 3, int iterations = 1);

/**
 * @brief Find Canny Edge from the Image
 * @param img Input Image (Single Channel)
 * @param kernelSize Size of the Kernel (Default: 3)
 * @param lowThr Low Threshold (Default: 20)
 * @param highThr High Threshold (Default: 100)
 * @param domain Domain Value (Default: 255)
 * @return cv::Mat Edge Image
 */
cv::Mat cannyEdge(cv::Mat img, float lowThr = 20, float highThr = 100, int domain = 255);

/**
 * @brief Find Neighbor Edge from the Image
 * @param img Input Image (Single Channel)
 * @param threshold Threshold Value (Default: 0.001)
 * @return cv::Mat Edge Image
 */
cv::Mat neighborEdge(cv::Mat img, float threshold = 0.001);

/**
 * @brief Get Corner Value from the Image
 * @param img Input Image (Single Channel)
 * @param row Row Index
 * @param col Column Index
 * @param kernelSize Size of the Kernel (Default: 3)
 * @return cv::Vec4f Corner Value (Top-Left, Top-Right, Bottom-Left, Bottom-Right)
 */
cv::Vec4f getCorner(cv::Mat img, int row, int col, int kernelSize);

/**
 * @brief Get Border Value from the Image
 * @param img Input Image (Single Channel)
 * @param row Row Index
 * @param col Column Index
 * @param kernelSize Size of the Kernel (Default: 3)
 * @return cv::Vec4f Border Value (Top, Bottom, Left, Right)
 */
cv::Vec4f getBorder(cv::Mat img, int row, int col, int kernelSize);

/**
 * @brief Get High Frequency Image
 * @param oriImg Original Image
 * @param lpfImg Low Pass Filtered Image
 * @param clipNeg Clip Negative Values (Default: false)
 * @return cv::Mat High Frequency Image
 */
cv::Mat getHPF(cv::Mat oriImg, cv::Mat lpfImg, bool clipNeg = false);

/**
 * @brief Multi Exposure Filter
 * @param imgList List of different exposure images
 * @param func Function to apply to the image
 * @param mode Mode of the filter <or, and, add> (Default: or)
 * @return cv::Mat Filtered Image
 */
cv::Mat multiExpF(std::vector<cv::Mat> imgList, std::function<cv::Mat(cv::Mat)> func, std::string mode = "or");
}  // namespace filter

#endif  // FILTER_HPP

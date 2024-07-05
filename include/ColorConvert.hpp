#pragma once

#ifndef COLORCONVERT_HPP
#define COLORCONVERT_HPP

#include <opencv2/opencv.hpp>

// Define Parameters
#define sRGB_GM 2.4
#define sRGB_TH 0.04045
// ... Color Convert Matrices
#define sRGBMXYZ (cv::Mat_<float>(3, 3) << 0.4124564, 0.3575761, 0.1804375, 0.2126729, 0.7151522, 0.0721750, 0.0193339, 0.1191920, 0.9503041)
#define XYZMsRGB (cv::Mat_<float>(3, 3) << 3.2404542, -1.5371385, -0.4985314, -0.9692660, 1.8760108, 0.0415560, 0.0556434, -0.2040259, 1.0572252)
#define XYZMLMS (cv::Mat_<float>(3, 3) << 0.8189330101, 0.3618667424, -0.1288597137, 0.0329845436, 0.9293118715, 0.0361456387, 0.0482003018, 0.2643662691, 0.6338517070)
#define LMSMXYZ (cv::Mat_<float>(3, 3) << 1.2270138511, -0.5577999807, 0.2812560149, -0.0405801784, 1.1122568696, -0.0716766787, -0.0763812845, -0.4214819784, 1.5861632204)
#define LMSMOKL (cv::Mat_<float>(3, 3) << 0.2104542553, 0.7936177850, -0.0040720468, 1.9779984951, -2.4285922050, 0.4505937099, 0.0259040371, 0.7827717662, -0.8086757660)
#define OKLMLMS (cv::Mat_<float>(3, 3) << 0.9999999985, 0.3963377922, 0.2158037581, 1.0000000089, -0.1055613423, -0.0638541748, 1.0000000547, -0.0894841821, -1.2914855379)
// ... White Points (XYZ)
#define D65_WP cv::Vec3f(0.95047, 1.00000, 1.08883)
#define D50_WP cv::Vec3f(0.96422, 1.00000, 0.82521)
#define A_WP cv::Vec3f(1.09850, 1.00000, 0.35585)
#define C_WP cv::Vec3f(0.98074, 1.00000, 1.18232)
#define E_WP cv::Vec3f(1.00000, 1.00000, 1.00000)
// ... White Points (xy)
#define D65_xy cv::Vec3f(0.3127, 0.3290)
// ... Scale
#define ONE 1.0
#define HUNDRED 100.0
#define BIT8 255.0
#define BIT10 1023.0
#define BIT12 4095.0
#define BIT14 16383.0
#define BIT16 65535.0
// Using namespace colorconvert for ColorConvert
namespace colorconvert {
// Global variables
inline std::vector<cv::Mat> _cvtMat_RGB2XYZ = {sRGBMXYZ, XYZMsRGB};  // RGB2XYZ, XYZ2RGB
inline std::vector<cv::Mat> _cvtMat_LMS2OKL = {LMSMOKL, OKLMLMS};    // LMS2OKL, OKL2LMS
inline std::vector<cv::Mat> _cvtMat_XYZ2LMS = {XYZMLMS, LMSMXYZ};    // XYZ2LMS, LMS2XYZ
inline cv::Vec3f _WP_XYZ = D65_WP;                                   // White Point for XYZ
inline cv::Vec3f _WP_xy = D65_xy;                                    // White Point for xy
inline float _gamma = sRGB_GM;                                       // Gamma value
inline float _threshold = 0.04045;                                   // Threshold value for gamma correction

// =========================================== Image Color Processing =========================================== //
/**
 * @brief Image Color Conversion
 * @param img Input image
 * @param cvtFunc Conversion function
 * @param scaleIn Scale value for input image (default: 1.0)
 * @param scaleOut Scale value for output image (default: 1.0)
 * @return cv::Mat Output image
 */
cv::Mat cvtColor(cv::Mat img, cv::Vec3f (*cvtFunc)(cv::Vec3f, float, float), float scaleIn = ONE, float scaleOut = ONE);
cv::Mat cvtColor(cv::Mat img, float (*cvtFunc)(float, float, float), float scaleIn = ONE, float scaleOut = ONE);

/**
 * @brief Combine Image Color Channels
 * @param imgChs Image channels
 * @return cv::Mat Combined image
 */
cv::Mat mergeCh(std::vector<cv::Mat> imgChs);
cv::Mat mergeCh(cv::Mat ch1, cv::Mat ch2, cv::Mat ch3);

/**
 * @brief Split Image Color Channels
 * @param img Input image
 * @return std::vector<cv::Mat> Image channels
 */
std::vector<cv::Mat> splitCh(cv::Mat img);

/**
 * @brief Get Specific Color Channel
 * @param img Input image
 * @param ch Channel index
 * @return cv::Mat Specific color channel
 */
cv::Mat getCh(cv::Mat img, int ch);

// ========================================= Color Conversion Functions ========================================= //

/**
 * @brief Gamma correction - From linear RGB to gamma RGB
 * @param pixel Pixel to be corrected
 * @param gamma Gamma value (default: 2.4)
 * @param threshold Threshold value (default: 0.04045)
 * @param scaleIn Scale value (default: 1.0)
 * @param scaleOut Scale value (default: 1.0)
 * @return cv::Vec3f Corrected pixel
 */
cv::Vec3f lRGB2gRGB(cv::Vec3f pixel, float gamma = sRGB_GM, float threshold = sRGB_TH, float scaleIn = ONE, float scaleOut = ONE);
static inline cv::Vec3f lRGB2gRGB(cv::Vec3f pixel, float scaleIn, float scaleOut) {
    return lRGB2gRGB(pixel, _gamma, _threshold, scaleIn, scaleOut);
}

/**
 * @brief Linear RGB correction - From gamma RGB to linear RGB
 * @param pixel Linear RGB pixel R[0,1], G[0,1], B[0,1]
 * @param gamma Gamma value (default: 2.4)
 * @param threshold Threshold value (default: 0.04045)
 * @param scaleIn Scale value (default: 1.0)
 * @param scaleOut Scale value (default: 1.0)
 * @return cv::Vec3f Corrected pixel
 */
cv::Vec3f gRGB2lRGB(cv::Vec3f pixel, float gamma = sRGB_GM, float threshold = sRGB_TH, float scaleIn = ONE, float scaleOut = ONE);
static inline cv::Vec3f gRGB2lRGB(cv::Vec3f pixel, float scaleIn, float scaleOut) {
    return gRGB2lRGB(pixel, _gamma, _threshold, scaleIn, scaleOut);
}

/**
 * @brief RGB to XYZ conversion
 * @param pixel Pixel to be converted
 * @param matrix Matrix to be used (default: _sRGB2XYZ_mat)
 * @param scaleIn Scale value (default: 1.0)
 * @param scaleOut Scale value (default: 1.0)
 * @return cv::Vec3f Converted pixel
 */
cv::Vec3f RGB2XYZ(cv::Vec3f pixel, cv::Mat matrix = sRGBMXYZ, float scaleIn = ONE, float scaleOut = ONE);
static inline cv::Vec3f RGB2XYZ(cv::Vec3f pixel, float scaleIn, float scaleOut) {
    return RGB2XYZ(pixel, _cvtMat_RGB2XYZ[0], scaleIn, scaleOut);
}

/**
 * @brief XYZ to RGB conversion
 * @param pixel Pixel to be converted
 * @param matrix Matrix to be used (default: _XYZ2sRGB_mat)
 * @param scaleIn Scale value (default: 1.0)
 * @param scaleOut Scale value (default: 1.0)
 * @return cv::Vec3f Converted pixel
 */
cv::Vec3f XYZ2RGB(cv::Vec3f pixel, cv::Mat matrix = XYZMsRGB, float scaleIn = ONE, float scaleOut = ONE);
static inline cv::Vec3f XYZ2RGB(cv::Vec3f pixel, float scaleIn, float scaleOut) {
    return XYZ2RGB(pixel, _cvtMat_RGB2XYZ[1], scaleIn, scaleOut);
}

/**
 * @brief XYZ to Lab conversion
 * @param pixel Pixel to be converted
 * @param white_point White point to be used (default: D65)
 * @param scaleIn Scale value for input image (default: 1.0)
 * @param scaleOut Scale value for output image (default: 1.0)
 * @return cv::Vec3f Converted pixel
 */
cv::Vec3f XYZ2Lab(cv::Vec3f pixel, cv::Vec3f white_point = D65_WP, float scaleIn = ONE, float scaleOut = ONE);
static inline cv::Vec3f XYZ2Lab(cv::Vec3f pixel, float scaleIn, float scaleOut) {
    return XYZ2Lab(pixel, _WP_XYZ, scaleIn, scaleOut);
}

/**
 * @brief Lab to XYZ conversion
 * @param pixel Pixel to be converted
 * @param white_point White point to be used (default: D65)
 * @param scaleIn Scale value for input image (default: 1.0)
 * @param scaleOut Scale value for output image (default: 1.0)
 * @return cv::Vec3f Converted pixel
 */
cv::Vec3f Lab2XYZ(cv::Vec3f pixel, cv::Vec3f white_point = D65_WP, float scaleIn = ONE, float scaleOut = ONE);
static inline cv::Vec3f Lab2XYZ(cv::Vec3f pixel, float scaleIn, float scaleOut) {
    return Lab2XYZ(pixel, _WP_XYZ, scaleIn, scaleOut);
}

/**
 * @brief Y to L conversion
 * @param pixel Pixel to be converted
 * @param scaleIn Scale value for input image (default: 1.0)
 * @param scaleOut Scale value for output image (default: 1.0)
 * @return float Converted pixel
 */
float Y2L(float pixel, float scaleIn = ONE, float scaleOut = ONE);

/**
 * @brief L to Y conversion
 * @param pixel Pixel to be converted
 * @param scaleIn Scale value for input image (default: 1.0)
 * @param scaleOut Scale value for output image (default: 1.0)
 * @return float Converted pixel
 */
float L2Y(float pixel, float scaleIn = ONE, float scaleOut = ONE);

/**
 * @brief XYZ to OKLAB conversion
 * @param pixel Pixel to be converted
 * @param matXYZ2LMS Matrix to be used (default: XYZMLMS)
 * @param matLMS2OKL Matrix to be used (default: LMSMOKL)
 * @param scaleIn Scale value for input image (default: 1.0)
 * @param scaleOut Scale value for output image (default: 1.0)
 * @return cv::Vec3f Converted pixel
 */
cv::Vec3f XYZ2OKLAB(cv::Vec3f pixel, cv::Mat matXYZ2LMS = XYZMLMS, cv::Mat matLMS2OKL = LMSMOKL, float scaleIn = ONE, float scaleOut = ONE);
static inline cv::Vec3f XYZ2OKLAB(cv::Vec3f pixel, float scaleIn, float scaleOut) {
    return XYZ2OKLAB(pixel, _cvtMat_XYZ2LMS[0], _cvtMat_LMS2OKL[0], scaleIn, scaleOut);
}

/**
 * @brief OKLAB to XYZ conversion
 * @param pixel Pixel to be converted
 * @param matOKL2LMS Matrix to be used (default: OKLMLMS)
 * @param matLMS2XYZ Matrix to be used (default: LMSMXYZ)
 * @param scaleIn Scale value for input image (default: 1.0)
 * @param scaleOut Scale value for output image (default: 1.0)
 * @return cv::Vec3f Converted pixel
 */
cv::Vec3f OKLAB2XYZ(cv::Vec3f pixel, cv::Mat matOKL2LMS = OKLMLMS, cv::Mat matLMS2XYZ = LMSMXYZ, float scaleIn = ONE, float scaleOut = ONE);
static inline cv::Vec3f OKLAB2XYZ(cv::Vec3f pixel, float scaleIn, float scaleOut) {
    return OKLAB2XYZ(pixel, _cvtMat_LMS2OKL[1], _cvtMat_XYZ2LMS[1], scaleIn, scaleOut);
}

/**
 * @brief XYZ to Yxy conversion
 * @param pixel Pixel to be converted
 * @param white_xy White point to be used (default: D65)
 * @param scaleIn Scale value for input image (default: 1.0)
 * @param scaleOut Scale value for output image (default: 1.0)
 * @return cv::Vec3f Converted pixel
 */
cv::Vec3f XYZ2Yxy(cv::Vec3f pixel, cv::Vec3f white_xy = D65_xy, float scaleIn = ONE, float scaleOut = ONE);
static inline cv::Vec3f XYZ2Yxy(cv::Vec3f pixel, float scaleIn, float scaleOut) {
    return XYZ2Yxy(pixel, _WP_xy, scaleIn, scaleOut);
}

/**
 * @brief Yxy to XYZ conversion
 * @param pixel Pixel to be converted
 * @param scaleIn Scale value for input image (default: 1.0)
 * @param scaleOut Scale value for output image (default: 1.0)
 * @return cv::Vec3f Converted pixel
 */
cv::Vec3f Yxy2XYZ(cv::Vec3f pixel, float scaleIn = ONE, float scaleOut = ONE);
}  // namespace colorconvert
#endif  // COLORCONVERT_HPP

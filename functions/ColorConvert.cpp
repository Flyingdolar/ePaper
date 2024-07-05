#include "ColorConvert.hpp"

// Using namespace colorconvert for ColorConvert
namespace colorconvert {
// =========================================== Image Color Processing =========================================== //
// Image Color Conversion
cv::Mat cvtColor(cv::Mat img, cv::Vec3f (*cvtFunc)(cv::Vec3f, float, float), float scaleIn, float scaleOut) {
    cv::Mat imgOut = cv::Mat::zeros(img.size(), img.type());
    for (int row = 0; row < img.rows; row++)
        for (int col = 0; col < img.cols; col++) {
            cv::Vec3f pixel = img.at<cv::Vec3f>(row, col);
            imgOut.at<cv::Vec3f>(row, col) = cvtFunc(pixel, scaleIn, scaleOut);
        }
    return imgOut;
}
cv::Mat cvtColor(cv::Mat img, float (*cvtFunc)(float, float, float), float scaleIn, float scaleOut) {
    cv::Mat imgOut = cv::Mat::zeros(img.size(), img.type());
    for (int row = 0; row < img.rows; row++)
        for (int col = 0; col < img.cols; col++) {
            float pixel = img.at<float>(row, col);
            imgOut.at<float>(row, col) = cvtFunc(pixel, scaleIn, scaleOut);
        }
    return imgOut;
}
// Combine Image Color Channels
cv::Mat mergeCh(std::vector<cv::Mat> imgChs) {
    cv::Mat imgOut = cv::Mat::zeros(imgChs[0].size(), CV_32FC3);
    for (int row = 0; row < imgChs[0].rows; row++)
        for (int col = 0; col < imgChs[0].cols; col++)
            for (int ch = 0; ch < imgChs.size(); ch++)
                imgOut.at<cv::Vec3f>(row, col)[ch] = imgChs[ch].at<float>(row, col);
    return imgOut;
}
cv::Mat mergeCh(cv::Mat ch1, cv::Mat ch2, cv::Mat ch3) {
    std::vector<cv::Mat> imgChs = {ch1, ch2, ch3};
    return mergeCh(imgChs);
}
// Split Image Color Channels
std::vector<cv::Mat> splitCh(cv::Mat img) {
    std::vector<cv::Mat> imgChs;
    for (int ch = 0; ch < img.channels(); ch++) {
        cv::Mat imgCh = cv::Mat::zeros(img.size(), CV_32FC1);
        for (int row = 0; row < img.rows; row++)
            for (int col = 0; col < img.cols; col++)
                imgCh.at<float>(row, col) = img.at<cv::Vec3f>(row, col)[ch];
        imgChs.push_back(imgCh);
    }
    return imgChs;
}
// Get Image Color Channel
cv::Mat getCh(cv::Mat img, int ch) {
    std::vector<cv::Mat> imgChs = splitCh(img);
    return imgChs[ch];
}
// =========================================== Color Conversion Functions =========================================== //
// Linear RGB to gamma RGB
cv::Vec3f lRGB2gRGB(cv::Vec3f pixel, float gamma, float threshold, float scaleIn, float scaleOut) {
    cv::Vec3f gmPixel, linePixel = pixel;  // Gamma-corrected pixel
    float ratio = threshold / std::pow((threshold + 0.055) / 1.055, gamma);
    threshold = threshold / ratio;
    for (size_t ch = 0; ch < 3; ch++) {
        linePixel[ch] = linePixel[ch] / scaleIn;  // Scale input
        if (linePixel[ch] <= threshold)           // Linear Part
            gmPixel[ch] = linePixel[ch] * ratio;
        else  // Gamma Part
            gmPixel[ch] = 1.055 * std::pow(linePixel[ch], 1 / gamma) - 0.055;
    }
    return gmPixel * scaleOut;  // Scale output
}
// Gamma RGB to linear RGB
cv::Vec3f gRGB2lRGB(cv::Vec3f pixel, float gamma, float threshold, float scaleIn, float scaleOut) {
    cv::Vec3f linePixel, gmPixel = pixel;  // Linear-corrected pixel
    float ratio = std::pow((threshold + 0.055) / 1.055, gamma) / threshold;
    for (size_t ch = 0; ch < 3; ch++) {
        gmPixel[ch] = gmPixel[ch] / scaleIn;  // Scale input
        if (gmPixel[ch] <= threshold)         // Linear Part
            linePixel[ch] = gmPixel[ch] * ratio;
        else  // Gamma Part
            linePixel[ch] = std::pow((gmPixel[ch] + 0.055) / 1.055, gamma);
    }
    return linePixel * scaleOut;
}
// RGB to XYZ conversion
cv::Vec3f RGB2XYZ(cv::Vec3f pixel, cv::Mat matrix, float scaleIn, float scaleOut) {
    cv::Vec3f XYZPixel;  // XYZ color space pixel
    // Scale input
    cv::Mat pixelMat = (cv::Mat_<float>(3, 1) << pixel[0] / scaleIn, pixel[1] / scaleIn, pixel[2] / scaleIn);
    cv::Mat XYZMat = matrix * pixelMat;  // Convert RGB to XYZ
    XYZPixel[0] = XYZMat.at<float>(0, 0), XYZPixel[1] = XYZMat.at<float>(1, 0), XYZPixel[2] = XYZMat.at<float>(2, 0);
    return XYZPixel * scaleOut;  // Scale output
}
// XYZ to RGB conversion
cv::Vec3f XYZ2RGB(cv::Vec3f pixel, cv::Mat matrix, float scaleIn, float scaleOut) {
    cv::Vec3f RGBPixel;  // RGB color space pixel
    // Scale input
    cv::Mat pixelMat = (cv::Mat_<float>(3, 1) << pixel[0] / scaleIn, pixel[1] / scaleIn, pixel[2] / scaleIn);
    cv::Mat RGBMat = matrix * pixelMat;  // Convert XYZ to RGB
    RGBPixel[0] = RGBMat.at<float>(0, 0), RGBPixel[1] = RGBMat.at<float>(1, 0), RGBPixel[2] = RGBMat.at<float>(2, 0);
    return RGBPixel * scaleOut;  // Scale output
}
// XYZ to Lab conversion
cv::Vec3f XYZ2Lab(cv::Vec3f pixel, cv::Vec3f white_point, float scaleIn, float scaleOut) {
    cv::Vec3f LabPixel;  // Lab color space pixel
    // Scale input
    cv::Mat pixelMat = (cv::Mat_<float>(3, 1) << pixel[0] / scaleIn, pixel[1] / scaleIn, pixel[2] / scaleIn);
    cv::Mat whitePointMat = (cv::Mat_<float>(3, 1) << white_point[0], white_point[1], white_point[2]);
    cv::Mat XYZMat = pixelMat / whitePointMat;  // Adjust XYZ by white point
    for (size_t ch = 0; ch < 3; ch++) {
        // Threshold = 0.008856 (216 / 24389)
        if (XYZMat.at<float>(ch, 0) > 0.008856) {  // f(x) = x^(1/3)
            XYZMat.at<float>(ch, 0) = std::pow(XYZMat.at<float>(ch, 0), 1.0 / 3.0);
        } else {  // f(x) = (903.3 * x + 16) / 116  --> 903.3 = 24389/27
            XYZMat.at<float>(ch, 0) = (903.3 * XYZMat.at<float>(ch, 0) + 16.0) / 116.0;
        }
    }
    LabPixel[0] = 1.16 * XYZMat.at<float>(1, 0) - 0.16;                   // L = 116 * Y^(1/3) - 16
    LabPixel[1] = 5 * (XYZMat.at<float>(0, 0) - XYZMat.at<float>(1, 0));  // a = 5 * (X^(1/3) - Y^(1/3))
    LabPixel[2] = 2 * (XYZMat.at<float>(1, 0) - XYZMat.at<float>(2, 0));  // b = 2 * (Y^(1/3) - Z^(1/3))
    // Scale output
    if (scaleOut != ONE && scaleOut != HUNDRED) {
        LabPixel[0] = LabPixel[0] * scaleOut;
        LabPixel[1] = (LabPixel[1] + 1) / 2 * scaleOut;
        LabPixel[2] = (LabPixel[2] + 1) / 2 * scaleOut;
    } else
        LabPixel = LabPixel * scaleOut;
    return LabPixel;
}
// Lab to XYZ conversion
cv::Vec3f Lab2XYZ(cv::Vec3f pixel, cv::Vec3f white_point, float scaleIn, float scaleOut) {
    cv::Vec3f XYZPixel, LABPixel = pixel;
    // Scale input
    if (scaleIn != ONE && scaleIn != HUNDRED) {
        LABPixel[0] = LABPixel[0] / scaleIn;
        LABPixel[1] = LABPixel[1] / scaleIn * 2 - 1;
        LABPixel[2] = LABPixel[2] / scaleIn * 2 - 1;
    } else
        LABPixel = LABPixel / scaleIn;
    cv::Mat pixelMat = (cv::Mat_<float>(3, 1) << LABPixel[0], LABPixel[1], LABPixel[2]);
    cv::Mat whitePointMat = (cv::Mat_<float>(3, 1) << white_point[0], white_point[1], white_point[2]);
    cv::Mat XYZMat = cv::Mat::zeros(3, 1, CV_32FC1);
    XYZMat.at<float>(1, 0) = (pixelMat.at<float>(0, 0) + 0.16) / 1.16;               // Y = (L + 16) / 116
    XYZMat.at<float>(0, 0) = XYZMat.at<float>(1, 0) + pixelMat.at<float>(1, 0) / 5;  // X = Y + a / 5
    XYZMat.at<float>(2, 0) = XYZMat.at<float>(1, 0) - pixelMat.at<float>(2, 0) / 2;  // Z = Y - b / 2
    for (size_t ch = 0; ch < 3; ch++) {
        if (std::pow(XYZMat.at<float>(ch, 0), 3) > 0.008856) {  // f(x) = x^3
            XYZMat.at<float>(ch, 0) = std::pow(XYZMat.at<float>(ch, 0), 3);
        } else {  // f(x) = (x - 16 / 116) / 7.787
            XYZMat.at<float>(ch, 0) = (XYZMat.at<float>(ch, 0) - 16.0 / 116.0) / 7.787;
        }
    }
    XYZMat = XYZMat.mul(whitePointMat);  // Adjust XYZ by white point
    XYZPixel[0] = XYZMat.at<float>(0, 0), XYZPixel[1] = XYZMat.at<float>(1, 0), XYZPixel[2] = XYZMat.at<float>(2, 0);
    return XYZPixel * scaleOut;  // Scale output
}
// Y to L conversion
float Y2L(float pixel, float scaleIn, float scaleOut) {
    float LPixel, YPixel = pixel / scaleIn;  // Scale input
    if (YPixel > 0.008856)
        LPixel = 1.16 * std::pow(YPixel, 1.0 / 3.0) - 0.16;  // L = 116 * Y^(1/3) - 16
    else
        LPixel = 9.033 * YPixel;  // f(x) = 903.3 * x
    return LPixel * scaleOut;     // Scale output
}
// L to Y conversion
float L2Y(float pixel, float scaleIn, float scaleOut) {
    float YPixel, LPixel = pixel / scaleIn;  // Scale input
    if (LPixel > 0.08)
        YPixel = std::pow((LPixel + 0.16) / 1.16, 3);  // Y = ((L + 16) / 116)^3
    else
        YPixel = LPixel / 9.033;  // f(x) = x / 903.3
    return YPixel * scaleOut;     // Scale output
}
// XYZ to OKLab conversion
cv::Vec3f XYZ2OKLAB(cv::Vec3f pixel, cv::Mat matXYZ2LMS, cv::Mat matLMS2OKL, float scaleIn, float scaleOut) {
    cv::Vec3f OKLabPixel;  // OKLab color space pixel
    // Scale input
    cv::Mat XYZMat = (cv::Mat_<float>(3, 1) << pixel[0] / scaleIn, pixel[1] / scaleIn, pixel[2] / scaleIn);
    cv::Mat LMSMat = matXYZ2LMS * XYZMat;  // Convert XYZ to LMS
    for (size_t ch = 0; ch < 3; ch++)      // f(x) = x^(1/3)
        LMSMat.at<float>(ch, 0) = std::pow(LMSMat.at<float>(ch, 0), 1.0 / 3.0);
    cv::Mat OKLMat = matLMS2OKL * LMSMat;  // Convert LMS to OKLab
    OKLabPixel[0] = OKLMat.at<float>(0, 0), OKLabPixel[1] = OKLMat.at<float>(1, 0), OKLabPixel[2] = OKLMat.at<float>(2, 0);
    return OKLabPixel * scaleOut;  // Scale output
}
// OKLab to XYZ conversion
cv::Vec3f OKLAB2XYZ(cv::Vec3f pixel, cv::Mat matOKL2LMS, cv::Mat matLMS2XYZ, float scaleIn, float scaleOut) {
    cv::Vec3f XYZPixel;  // XYZ color space pixel
    // Scale input
    cv::Mat OKLMat = (cv::Mat_<float>(3, 1) << pixel[0] / scaleIn, pixel[1] / scaleIn, pixel[2] / scaleIn);
    cv::Mat LMSMat = matOKL2LMS * OKLMat;  // Convert OKLab to LMS
    for (size_t ch = 0; ch < 3; ch++)      // f(x) = x^3
        LMSMat.at<float>(ch, 0) = std::pow(LMSMat.at<float>(ch, 0), 3);
    cv::Mat XYZMat = matLMS2XYZ * LMSMat;  // Convert LMS to XYZ
    XYZPixel[0] = XYZMat.at<float>(0, 0), XYZPixel[1] = XYZMat.at<float>(1, 0), XYZPixel[2] = XYZMat.at<float>(2, 0);
    return XYZPixel * scaleOut;  // Scale output
}
// XYZ to Yxy conversion
cv::Vec3f XYZ2Yxy(cv::Vec3f pixel, cv::Vec3f white_xy, float scaleIn, float scaleOut) {
    cv::Vec3f YxyPixel, XYZPixel = pixel / scaleIn;  // Scale input
    float sum = XYZPixel[0] + XYZPixel[1] + XYZPixel[2];
    YxyPixel[0] = XYZPixel[1];  // Y = Y
    if (sum != 0)
        // x = X / (X + Y + Z), y = Y / (X + Y + Z)
        YxyPixel[1] = XYZPixel[0] / sum, YxyPixel[2] = XYZPixel[1] / sum;
    else  // Use white point if X+Y+Z=0
        YxyPixel[1] = white_xy[0], YxyPixel[2] = white_xy[1];

    // Scale output
    if (scaleOut != ONE || scaleOut != HUNDRED) {
        YxyPixel[0] = YxyPixel[0] * scaleOut;
        YxyPixel[1] = (YxyPixel[1] + 1) / 2 * scaleOut;
        YxyPixel[2] = (YxyPixel[2] + 1) / 2 * scaleOut;
    } else
        YxyPixel = YxyPixel * scaleOut;
    return YxyPixel;
}
// Yxy to XYZ conversion
cv::Vec3f Yxy2XYZ(cv::Vec3f pixel, float scaleIn, float scaleOut) {
    cv::Vec3f XYZPixel, YxyPixel = pixel;
    // Scale input
    if (scaleIn != ONE || scaleIn != HUNDRED) {
        YxyPixel[0] = YxyPixel[0] / scaleIn;
        YxyPixel[1] = YxyPixel[1] / scaleIn * 2 - 1;
        YxyPixel[2] = YxyPixel[2] / scaleIn * 2 - 1;
    } else
        YxyPixel = YxyPixel / scaleIn;
    if (YxyPixel[2] != 0) {
        XYZPixel[0] = YxyPixel[0] * YxyPixel[1] / YxyPixel[2];                      // X = x * Y / y
        XYZPixel[1] = YxyPixel[0];                                                  // Y = Y
        XYZPixel[2] = (1 - YxyPixel[1] - YxyPixel[2]) * YxyPixel[0] / YxyPixel[2];  // Z = (1 - x - y) * Y / y
    } else
        XYZPixel = cv::Vec3f(0, 0, 0);  // Use white point if y=0
    return XYZPixel * scaleOut;         // Scale output
}
}  // namespace colorconvert

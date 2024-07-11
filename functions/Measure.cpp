#include "Measure.hpp"

namespace measure {

// Measure PSNR between two images
double PSNR(const cv::Mat1f testImg, const cv::Mat1f refImg) {
    // 1. Get Difference Image
    cv::Mat1f diffImg;  // Difference Image (Absolute)
    cv::absdiff(testImg, refImg, diffImg);
    diffImg = diffImg.mul(diffImg);  // Square of Difference Image

    // 2. Calculate MSE
    double mse = cv::mean(diffImg)[0];
    if (mse <= 0.0) return -1.0;  // Invalid Value

    // 3. Calculate PSNR
    double psnr = 10.0 * std::log10(1.0 / mse);
    return psnr;
}

// Measure SSIM between two images
double SSIM(const cv::Mat1f testImg, const cv::Mat1f refImg, int kSize, float sigma, double cst1, double cst2) {
    cv::Mat1f sqTest = testImg.mul(testImg), sqRef = refImg.mul(refImg), mulImg = testImg.mul(refImg);
    cv::Mat1f muTest, muRef, muMul, sigmaTest, sigmaRef, sigmaMul;

    // 1. Calculate Mean
    cv::GaussianBlur(testImg, muTest, cv::Size(kSize, kSize), sigma);
    cv::GaussianBlur(refImg, muRef, cv::Size(kSize, kSize), sigma);

    // 2. Calculate Variance
    cv::GaussianBlur(sqTest, sigmaTest, cv::Size(kSize, kSize), sigma);
    cv::subtract(sigmaTest, muTest.mul(muTest), sigmaTest);
    cv::GaussianBlur(sqRef, sigmaRef, cv::Size(kSize, kSize), sigma);
    cv::subtract(sigmaRef, muRef.mul(muRef), sigmaRef);

    // 3. Calculate Covariance
    cv::GaussianBlur(mulImg, muMul, cv::Size(11, 11), 1.5);
    cv::subtract(muMul, muTest.mul(muRef), muMul);

    // 4. Calculate SSIM
    cv::Mat1f num = (2.0 * muMul + cst1) * (2.0 * sigmaMul + cst2);
    cv::Mat1f den = (muTest.mul(muTest) + muRef.mul(muRef) + cst1) * (sigmaTest + sigmaRef + cst2);
    cv::Mat1f ssimMap;
    cv::divide(num, den, ssimMap);
    return cv::mean(ssimMap)[0];
}

}  // namespace measure

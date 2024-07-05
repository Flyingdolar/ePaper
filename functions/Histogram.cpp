#include "Histogram.hpp"

namespace histogram {

std::vector<int> getHist(cv::Mat img, int bins, std::pair<float, float> range) {
    std::vector<int> hist(bins, 0);
    float gap = (range.second - range.first) / bins;

    for (int row = 0; row < img.rows; row++)
        for (int col = 0; col < img.cols; col++) {
            float idx = (img.at<float>(row, col) - range.first) / gap;
            hist[int(idx)]++;
        }

    return hist;
}

std::vector<int> getHist(cv::Mat img, int bins, std::pair<float, float> range, cv::Mat mask) {
    if (img.rows != mask.rows || img.cols != mask.cols) {
        std::cerr << "Error: Image and mask must have the same size" << std::endl;
        return std::vector<int>();
    }

    int imgType = img.type(), maskType = mask.type();
    img.convertTo(img, CV_32F), mask.convertTo(mask, CV_32F);
    std::vector<int> hist(bins, 0);
    float gap = (range.second - range.first) / bins;
    int count = 0;

    for (int row = 0; row < img.rows; row++)
        for (int col = 0; col < img.cols; col++)
            if (mask.at<float>(row, col) > 0) {
                float idx = (img.at<float>(row, col) - range.first) / gap;
                hist[int(idx)]++, count++;
            }
    img.convertTo(img, imgType);
    return hist;
}

std::vector<int> getCDF(std::vector<int> hist) {
    std::vector<int> cdf(hist.size(), 0);
    cdf[0] = hist[0];
    for (int idx = 1; idx < hist.size(); idx++)
        cdf[idx] = cdf[idx - 1] + hist[idx];

    return cdf;
}

std::vector<int> limRateCDF(std::vector<int> hist, float min_rate, float max_rate, float min_inc, float max_inc) {
    std::vector<int> cdf = getCDF(hist), limCDF = cdf;
    std::vector<bool> incFlags(cdf.size(), false);
    int bins = hist.size(), nums = cdf[bins - 1];

    // Find & Set minimum and maximum limits
    for (int idx = 0; idx < bins; idx++) {
        float limMin = min_rate * idx / bins, limMax = max_rate * idx / bins;
        if (cdf[idx] < (limMin * nums)) limCDF[idx] = limMin * nums;
        if (cdf[idx] > (limMax * nums)) limCDF[idx] = limMax * nums;
    }
    // Find minimum and maximum increments
    incFlags[0] = true;
    for (int idx = 0; idx < bins; idx++) {
        if (idx == 0) continue;
        float incVal = (limCDF[idx] - limCDF[idx - 1]) / nums;
        if (incVal < min_inc || incVal > max_inc) incFlags[idx] = true;
    }
    // Apply increments
    for (int idx = 0; idx < bins; idx++) {
        if (incFlags[idx] == false) continue;
        int prev = idx, next = idx;
        while (prev > 0 && incFlags[prev] == true) prev--;
        while (next < bins - 1 && incFlags[next] == true) next++;
        int incVal = (limCDF[next] - limCDF[prev]) / (next - prev);
    }
    // Normalize
    float numRate = nums / limCDF[bins - 1];
    for (int idx = 0; idx < bins; idx++) limCDF[idx] *= numRate;

    return limCDF;
}

cv::Mat equalize(cv::Mat img, std::vector<int> cdf, int bins, std::pair<float, float> range) {
    cv::Mat eqImg = img.clone();
    int imgType = img.type();
    float gap = (range.second - range.first) / bins;

    img.convertTo(img, CV_32F), eqImg.convertTo(eqImg, CV_32F);
    for (int row = 0; row < img.rows; row++)
        for (int col = 0; col < img.cols; col++) {
            int idx = (img.at<float>(row, col) - range.first) / gap;
            eqImg.at<float>(row, col) = float(cdf[idx]) / float(cdf[cdf.size() - 1]) * (range.second - range.first) + range.first;
        }

    img.convertTo(img, imgType), eqImg.convertTo(eqImg, imgType);
    return eqImg;
}

}  // namespace histogram

#pragma once

#include "Functions.hpp"

namespace whitebalance {

#define BLKPOS std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>>

//================================ Variables =================================//
// Block Position that situated in the image
//  - these blocks are used to calculate the white balance with force method
const BLKPOS grayBKPos = {
    {{843, 551}, {903, 611}},  // 24: Black
    {{843, 635}, {903, 695}},  // 23: Neutral 3.5
    {{843, 719}, {903, 779}},  // 22: Neutral 5
    {{843, 803}, {903, 863}},  // 21: Neutral 6.5
    {{843, 887}, {903, 947}},  // 20: Neutral 8
    {{843, 971}, {903, 1031}}  // 19: White
};

// {{626, 618}, {677, 684}},    // 19: White
// {{627, 720}, {685, 786}},    // 20: Neutral 8
// {{625, 828}, {682, 893}},    // 21: Neutral 6.5
// {{622, 938}, {685, 1000}},   // 22: Neutral 5
// {{626, 1044}, {685, 1109}},  // 23: Neutral 3.5
// {{622, 1146}, {684, 1208}},  // 24: Black
//================================ Functions =================================//
/**
 * @brief Force White Balance - Calculate the white balance with force method
 * @param img Image to be white balanced
 * @param blkPos Block position that situated in the image
 * @return White balance bias & gain [(biasR, biasG, biasB), (gainR, gainG, gainB]
 * @note The white balance gain is calculated by the average of the blocks in the image
 */
std::pair<cv::Vec3f, cv::Vec3f> forceWB(cv::Mat img, BLKPOS blkPos = grayBKPos);

/**
 * @brief Force White Balance - Calculate the white balance with force method
 * @param bkVal Black value of the image [(R, G, B)]
 * @return White balance bias & gain [(biasR, biasG, biasB), (gainR, gainG, gainB)]
 * @note The white balance gain is calculated by the average of the blocks in the image
 */
std::pair<cv::Vec3f, cv::Vec3f> forceWB(std::vector<cv::Vec3f> bkVal);
std::pair<cv::Vec3f, cv::Vec3f> forceWB(cv::Mat3f imgVal, cv::Mat3f gtVal);

/**
 * @brief Apply White Balance - Apply the white balance to the image
 * @param img Image to be white balanced
 * @param wbGain White balance bias & gain [(biasR, biasG, biasB), (gainR, gainG, gainB)]
 * @return White balanced image
 * @note The white balance gain is calculated by the average of the blocks in the image
 */
cv::Mat applyWB(cv::Mat img, std::pair<cv::Vec3f, cv::Vec3f> wbGain);

/**
 * @brief Get Block Average - Calculate the average of the blocks in the image
 * @param img Image to be calculated
 * @param blkPos Block position that situated in the image
 * @return Average of the blocks in the image
 */
inline std::vector<cv::Vec3f> getBlkAvg(cv::Mat img, BLKPOS blkPos) {
    std::vector<cv::Vec3f> blkAvg;

    // Calculate the average of the blocks in the image
    for (auto blk : blkPos) {
        cv::Vec3f blkSum = {0.0f, 0.0f, 0.0f};
        for (int row = blk.first.first; row < blk.second.first; row++)
            for (int col = blk.first.second; col < blk.second.second; col++)
                blkSum += img.at<cv::Vec3f>(row, col);
        blkAvg.push_back(blkSum / ((blk.second.first - blk.first.first) * (blk.second.second - blk.first.second)));
    }

    return blkAvg;
}
}  // namespace whitebalance

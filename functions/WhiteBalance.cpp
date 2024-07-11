#include "WhiteBalance.hpp"

namespace whitebalance {
//=============================== Inner Macros ===============================//
#define ROW_ST first.first
#define ROW_ED second.first
#define COL_ST first.second
#define COL_ED second.second

//================================ Functions =================================//
// Force White Balance - Calculate the white balance with force method
std::pair<cv::Vec3f, cv::Vec3f> forceWB(cv::Mat img, BLKPOS blkPos) {
    std::pair<cv::Vec3f, cv::Vec3f> wbGain;
    std::vector<cv::Vec3f> blkAvg;

    // Calculate the average of the blocks in the image
    for (auto blk : blkPos) {
        cv::Vec3f blkSum = {0.0f, 0.0f, 0.0f};
        for (int row = blk.ROW_ST; row < blk.ROW_ED; row++)
            for (int col = blk.COL_ST; col < blk.COL_ED; col++)
                blkSum += img.at<cv::Vec3f>(row, col);
        blkAvg.push_back(blkSum / ((blk.ROW_ED - blk.ROW_ST) * (blk.COL_ED - blk.COL_ST)));
    }

    // Linearize the Color Checker Gray Scale
    std::vector<float> ccGray = {8.75665, 23.1646, 49.6275, 89.6408, 147.283, 228.549};

    // Get regression line of the blocks
    std::vector<cv::Point2f> ptsR, ptsG, ptsB;
    cv::Vec4f lineR, lineG, lineB;
    float rateR, rateG, rateB, biasR, biasG, biasB;

    for (size_t idx = 0; idx < 6; idx++)
        ccGray[idx] /= 255.0f;
    for (size_t idx = 0; idx < 6; idx++) {
        ptsR.push_back(cv::Point2f(ccGray[idx], blkAvg[idx][2]));
        ptsG.push_back(cv::Point2f(ccGray[idx], blkAvg[idx][1]));
        ptsB.push_back(cv::Point2f(ccGray[idx], blkAvg[idx][0]));
    }

    cv::fitLine(ptsR, lineR, cv::DIST_L2, 0, 1e-2, 1e-2);
    cv::fitLine(ptsG, lineG, cv::DIST_L2, 0, 1e-2, 1e-2);
    cv::fitLine(ptsB, lineB, cv::DIST_L2, 0, 1e-2, 1e-2);

    rateR = lineR[1] / lineR[0];
    rateG = lineG[1] / lineG[0];
    rateB = lineB[1] / lineB[0];
    biasR = lineR[3] - rateR * lineR[2];
    biasG = lineG[3] - rateG * lineG[2];
    biasB = lineB[3] - rateB * lineB[2];

    rateR = rateG / rateR;
    rateB = rateG / rateB;
    rateG = 1.0f;

    wbGain.first = cv::Vec3f(biasR, biasG, biasB);
    wbGain.second = cv::Vec3f(rateR, rateG, rateB);

    return wbGain;
}

// Block Force White Balance - Calculate the white balance with force method on blocks
std::pair<cv::Vec3f, cv::Vec3f> forceWB(std::vector<cv::Vec3f> bkVal) {
    // Linearize the Color Checker Gray Scale
    std::vector<float> ccGray = {8.75665, 23.1646, 49.6275, 89.6408, 147.283, 228.549};

    // Get regression line of the blocks
    std::vector<cv::Point2f> ptsR, ptsG, ptsB;
    cv::Vec4f lineR, lineG, lineB;
    float rateR, rateG, rateB, biasR, biasG, biasB;

    for (size_t idx = 0; idx < 6; idx++) {
        ptsR.push_back(cv::Point2f(ccGray[idx], bkVal[idx][0]));
        ptsG.push_back(cv::Point2f(ccGray[idx], bkVal[idx][1]));
        ptsB.push_back(cv::Point2f(ccGray[idx], bkVal[idx][2]));
    }

    cv::fitLine(ptsR, lineR, cv::DIST_L2, 0, 1e-2, 1e-2);
    cv::fitLine(ptsG, lineG, cv::DIST_L2, 0, 1e-2, 1e-2);
    cv::fitLine(ptsB, lineB, cv::DIST_L2, 0, 1e-2, 1e-2);

    rateR = lineR[1] / lineR[0];
    rateG = lineG[1] / lineG[0];
    rateB = lineB[1] / lineB[0];
    biasR = lineR[3] - rateR * lineR[2];
    biasG = lineG[3] - rateG * lineG[2];
    biasB = lineB[3] - rateB * lineB[2];

    rateR = 1.0f / rateR, rateG = 1.0f / rateG, rateB = 1.0f / rateB;

    return std::make_pair(cv::Vec3f(biasR, biasG, biasB), cv::Vec3f(rateR, rateG, rateB));
}

std::pair<cv::Vec3f, cv::Vec3f> forceWB(cv::Mat3f imgVal, cv::Mat3f gtVal) {
    std::vector<cv::Point2f> ptsR, ptsG, ptsB;
    cv::Vec4f lineR, lineG, lineB;

    // Get the Points of {imgR, gtR}, {imgG, gtG}, {imgB, gtB}
    for (int row = 0; row < imgVal.rows; row++)
        for (int col = 0; col < imgVal.cols; col++) {
            ptsR.push_back(cv::Point2f(gtVal(row, col)[2], imgVal(row, col)[2]));
            ptsG.push_back(cv::Point2f(gtVal(row, col)[1], imgVal(row, col)[1]));
            ptsB.push_back(cv::Point2f(gtVal(row, col)[0], imgVal(row, col)[0]));
        }

    // Get the Regression Line of {imgR, gtR}, {imgG, gtG}, {imgB, gtB}
    cv::fitLine(ptsR, lineR, cv::DIST_L2, 0, 1e-2, 1e-2);
    cv::fitLine(ptsG, lineG, cv::DIST_L2, 0, 1e-2, 1e-2);
    cv::fitLine(ptsB, lineB, cv::DIST_L2, 0, 1e-2, 1e-2);

    // Get the Gain & Bias of {imgR, gtR}, {imgG, gtG}, {imgB, gtB}
    float rateR = lineR[1] / lineR[0], rateG = lineG[1] / lineG[0], rateB = lineB[1] / lineB[0];
    float biasR = lineR[3] - rateR * lineR[2], biasG = lineG[3] - rateG * lineG[2], biasB = lineB[3] - rateB * lineB[2];

    // Get the White Balance Gain & Bias
    rateR = 1.0f / rateR, rateG = 1.0f / rateG, rateB = 1.0f / rateB;
    return std::make_pair(cv::Vec3f(biasR, biasG, biasB), cv::Vec3f(rateR, rateG, rateB));
}

// Apply White Balance - Apply the white balance to the image
cv::Mat applyWB(cv::Mat img, std::pair<cv::Vec3f, cv::Vec3f> wbGain) {
    cv::Mat wbImg = img.clone();
    cv::cvtColor(wbImg, wbImg, cv::COLOR_BGR2RGB);
    for (int row = 0; row < img.rows; row++)
        for (int col = 0; col < img.cols; col++)
            for (int ch = 0; ch < 3; ch++) {
                wbImg.at<cv::Vec3f>(row, col)[ch] -= wbGain.first[ch];
                wbImg.at<cv::Vec3f>(row, col)[ch] *= wbGain.second[ch];
            }
    cv::cvtColor(wbImg, wbImg, cv::COLOR_RGB2BGR);
    return wbImg;
}

}  // namespace whitebalance

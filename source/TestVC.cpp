#include "Functions.hpp"

int main() {
    saveData::initVar("res/test/VoidCluster");
    // Generate Random Image (16x16) (32x32) (64x64) (128x128)
    std::vector<cv::Mat1f> imgList;
    for (int idx = 4; idx <= 7; idx++) {
        cv::Mat1f img = halftone::getRandBin(cv::Vec2i(1 << idx, 1 << idx));
        imgList.push_back(img);
        saveData::imgMat(img, "Rand_" + std::to_string(1 << idx));
    }
    // Do Void Cluster Algorithm
    for (int idx = 0; idx < imgList.size(); idx++) {
        cv::Mat1f img = halftone::VoidCluster(imgList[idx], 3, 1.0, true);
        saveData::imgMat(img, "VC_" + std::to_string(1 << (idx + 4)));
    }
    return 0;
}

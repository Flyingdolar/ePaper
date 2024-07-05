#include "Functions.hpp"

int kernelSize = 9;

int main(int argc, char** argv) {
    // Setup the Save Path
    std::string savePath = "image/ME/Dither" + std::to_string(kernelSize);
    if (system(("mkdir -p " + savePath).c_str()) != 0) return -1;
    saveData::initVar(savePath);

    // Read the Image
    cv::Mat img = cv::imread("data/Me.jpg");
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);

    // Downscale the Image
    cv::resize(img, img, cv::Size(), 0.5, 0.5);

    // Split the Image into 3 Channels
    cv::Mat1f imgR = colorconvert::getCh(img, 2);
    cv::Mat1f imgG = colorconvert::getCh(img, 1);
    cv::Mat1f imgB = colorconvert::getCh(img, 0);

    // Do the Halftoning
    saveData::imgMat(imgR, "oriRed");
    cv::Mat1f hfR = halftone::Dither(imgR, kernelSize, true);
    saveData::imgMat(hfR, "halfRed");
    saveData::imgMat(imgG, "oriGreen");
    cv::Mat1f hfG = halftone::Dither(imgG, kernelSize, true);
    saveData::imgMat(hfG, "halfGreen");
    saveData::imgMat(imgB, "oriBlue");
    cv::Mat1f hfB = halftone::Dither(imgB, kernelSize, true);
    saveData::imgMat(hfB, "halfBlue");

    // Merge the Halftone Image
    cv::Mat3f hfImg = colorconvert::mergeCh({hfB, hfG, hfR});
    saveData::imgMat(img, "HF_Input"), saveData::imgMat(hfImg, "HF_Result");
    return 0;
}

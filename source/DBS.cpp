#include "Functions.hpp"

int DBSKernelSize = 9;
float DBSSigma = 1.0;
int DBSIters = 10;

int main(int argc, char** argv) {
    // Load the Image
    std::string savePath = "image/ME/DBS_K" + std::to_string(DBSKernelSize) + "_S" + std::to_string((int)DBSSigma) + "_I" + std::to_string(DBSIters);
    cv::Mat img = cv::imread("image/Me.jpg");
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);

    // Downscale the Image
    cv::resize(img, img, cv::Size(), 0.5, 0.5);

    // Create the Directory
    if (system(("mkdir -p " + savePath).c_str()) == -1) return -1;
    if (system(("mkdir -p " + savePath + "/Red").c_str()) == -1) return -1;
    if (system(("mkdir -p " + savePath + "/Green").c_str()) == -1) return -1;
    if (system(("mkdir -p " + savePath + "/Blue").c_str()) == -1) return -1;

    // Split the Image into 3 Channels
    cv::Mat1f imgR = colorconvert::getCh(img, 2);
    cv::Mat1f imgG = colorconvert::getCh(img, 1);
    cv::Mat1f imgB = colorconvert::getCh(img, 0);

    // Do the Halftoning
    saveData::initVar(savePath + "/Red");
    saveData::imgMat(imgR, "Original");
    cv::Mat1f hfR = halftone::DBS(imgR, DBSKernelSize, DBSSigma, DBSIters, true);
    saveData::imgMat(hfR, "Halftone");
    saveData::initVar(savePath + "/Green");
    saveData::imgMat(imgG, "Original");
    cv::Mat1f hfG = halftone::DBS(imgG, DBSKernelSize, DBSSigma, DBSIters, true);
    saveData::imgMat(hfG, "Halftone");
    saveData::initVar(savePath + "/Blue");
    saveData::imgMat(imgB, "Original");
    cv::Mat1f hfB = halftone::DBS(imgB, DBSKernelSize, DBSSigma, DBSIters, true);
    saveData::imgMat(hfB, "Halftone");

    // Merge the Halftone Image
    saveData::initVar(savePath);
    cv::Mat3f hfImg = colorconvert::mergeCh({hfB, hfG, hfR});
    saveData::imgMat(img, "HF_Input"), saveData::imgMat(hfImg, "HF_Result");
    return 0;
}

#include <opencv2/opencv.hpp>
#include <vector>

#include "Functions.hpp"

// ==================================== Generate Test Images ==================================== //
cv::Mat1f genGradImg(int size) {
    cv::Mat1f img(size, size, 0.0);
    for (int rdx = 0; rdx < size; rdx++)
        for (int cdx = 0; cdx < size; cdx++)
            img(rdx, cdx) = (rdx + cdx) / (2.0 * size);
    return img;
}

cv::Mat1f genGBkImg(int size, int blkSize) {
    cv::Mat1f img(size, size, 0.0);
    int blkNum = size / blkSize;
    for (int rdx = 0; rdx < size; rdx++)
        for (int cdx = 0; cdx < size; cdx++)
            img(rdx, cdx) = ((rdx / blkSize) + (cdx / blkSize)) / (2.0 * blkNum);
    return img;
}

cv::Mat1f genRandImg(cv::Size size) {
    cv::Mat1f img(size, 0.0);
    cv::randu(img, 0.0, 1.0);
    return img;
}

// ==================================== Test Function ==================================== //
int testHFT(std::string testName, std::function<cv::Mat1f(cv::Mat1f)> func, std::vector<std::pair<std::string, cv::Mat1f>> imgList) {
    std::string oriSF = saveData::defFolder;    // Save Original Folder
    saveData::initVar(oriSF + "/" + testName);  // Set Test Folder
    for (auto imgData : imgList) {
        cv::Mat1f resImg = func(imgData.second);
        saveData::imgMat(resImg, imgData.first);
    }
    saveData::initVar(oriSF);  // Reset Original Folder
    return 0;
}

// ==================================== Main Function ==================================== //
int main(int argc, char** argv) {
    // Create Test Images
    cv::Mat1f imgGrad = genGradImg(1024), imgGBk = genGBkImg(1024, 64), imgRand = genRandImg(cv::Size(1024, 1024));
    std::vector<std::pair<std::string, cv::Mat1f>> imgList = {{"Grad", imgGrad}, {"GBk", imgGBk}, {"Rand", imgRand}};
    saveData::initVar("image/test/Halftone");

    // Test 0: Original Image
    std::cout << "Test 0: Original Image" << std::endl;
    std::function tfPass = [](cv::Mat1f img) { return img; };
    testHFT("0_Original", tfPass, imgList);

    // Test 1: Halftone by Dithering
    std::cout << "Test 1: Halftone by Dithering" << std::endl;
    std::function tfDither2 = [](cv::Mat1f img) { return halftone::Dither(img, 2); };
    std::function tfDither4 = [](cv::Mat1f img) { return halftone::Dither(img, 4); };
    std::function tfDither8 = [](cv::Mat1f img) { return halftone::Dither(img, 8); };
    testHFT("1_Dither_2", tfDither2, imgList), testHFT("1_Dither_4", tfDither4, imgList), testHFT("1_Dither_8", tfDither8, imgList);

    // Test 2: Halftone by Error Diffusion
    std::cout << "Test 2: Halftone by Error Diffusion" << std::endl;
    std::function tfErrDiff3 = [](cv::Mat1f img) { return halftone::ErrDiff(img, 3); };
    std::function tfErrDiff5 = [](cv::Mat1f img) { return halftone::ErrDiff(img, 5); };
    testHFT("2_ErrDiff_FS", tfErrDiff3, imgList), testHFT("2_ErrDiff_JJN", tfErrDiff5, imgList);

    // Test 3: Direct Binary Search (DBS) Halftoning
    std::cout << "Test 3: Direct Binary Search (DBS) Halftoning" << std::endl;
    std::function tfDBSK3 = [](cv::Mat1f img) { return halftone::DBS(img, 3, 1.0, 10); };
    std::function tfDBSK5 = [](cv::Mat1f img) { return halftone::DBS(img, 5, 1.0, 10); };
    std::function tfDBSK13 = [](cv::Mat1f img) { return halftone::DBS(img, 13, 1.0, 10); };
    testHFT("3_DBS_K3", tfDBSK3, imgList), testHFT("3_DBS_K5", tfDBSK5, imgList), testHFT("3_DBS_K13", tfDBSK13, imgList);

    std::cout << "All Tests Done!" << std::endl;
    return 0;
}

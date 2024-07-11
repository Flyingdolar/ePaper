#include <opencv2/opencv.hpp>
#include <vector>

#include "Functions.hpp"

std::vector<cv::Mat1f> imgResList;

cv::Mat1f dithMap;

// ==================================== Generate Test Images ==================================== //
cv::Mat1f genGradImg(int imgSize) {
    cv::Mat1f img(imgSize, imgSize, 0.0);
    int centPos = imgSize / 2;
    for (int rdx = 0; rdx < imgSize; rdx++)
        for (int cdx = 0; cdx < imgSize; cdx++)
            img(rdx, cdx) = (float)(std::abs(rdx - centPos) + std::abs(cdx - centPos)) / imgSize;
    return img;
}

cv::Mat1f genGBkImg(int imgSize, int blkNum) {
    cv::Mat1f img(imgSize, imgSize, 0.0);
    int blkSize = imgSize / blkNum, midPos = imgSize / 2;
    for (int rdx = 0; rdx < imgSize; rdx++)
        for (int cdx = 0; cdx < imgSize; cdx++) {
            int bRdx = rdx / blkSize, bCdx = cdx / blkSize;
            img(rdx, cdx) = (float)(std::abs(bRdx - blkNum / 2) + std::abs(bCdx - blkNum / 2)) / blkNum;
        }
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
    imgResList.clear();
    for (auto imgData : imgList) {
        halftone::verbosePath = imgData.first + "_vb";  // Set Global Save Path
        cv::Mat1f resImg = func(imgData.second);
        saveData::imgMat(resImg, imgData.first);
        imgResList.push_back(resImg);
    }
    saveData::initVar(oriSF);  // Reset Original Folder
    return 0;
}

// ==================================== Main Function ==================================== //
int main(int argc, char** argv) {
    // Create Test Images & Read Images
    cv::Mat sampleImg = cv::imread("image/testME.png");
    cv::Mat1f imgGrad = genGradImg(512), imgGBk = genGBkImg(512, 7);
    sampleImg.convertTo(sampleImg, CV_32F, 1.0 / 255.0);
    cv::Mat1f spR = colorconvert::getCh(sampleImg, 2), spG = colorconvert::getCh(sampleImg, 1), spB = colorconvert::getCh(sampleImg, 0);
    std::vector<std::pair<std::string, cv::Mat1f>> imgList = {{"Grad", imgGrad}, {"GBk", imgGBk}, {"Sample_R", spR}, {"Sample_G", spG}, {"Sample_B", spB}};

    saveData::initVar("res/test/Halftone");

    // // Test 0: Original Image
    // std::cout << "Test 0: Original Image" << std::endl;
    // std::function tfPass = [](cv::Mat1f img) { return img; };
    // testHFT("0_Original", tfPass, imgList);
    // sampleImg = colorconvert::mergeCh(imgResList[4], imgResList[3], imgResList[2]);
    // saveData::imgMat(sampleImg, "SpColor");

    // // Test 1: Halftone by Dithering
    // std::cout << "Test 1: Halftone by Dithering" << std::endl;
    // std::function tfDither2 = [](cv::Mat1f img) { return halftone::Dither(img, 2); };
    // std::function tfDither4 = [](cv::Mat1f img) { return halftone::Dither(img, 4); };
    // std::function tfDither8 = [](cv::Mat1f img) { return halftone::Dither(img, 8); };
    // testHFT("1_Dither_2", tfDither2, imgList), testHFT("1_Dither_4", tfDither4, imgList), testHFT("1_Dither_8", tfDither8, imgList);
    // sampleImg = colorconvert::mergeCh(imgResList[4], imgResList[3], imgResList[2]);
    // saveData::imgMat(sampleImg, "SpColor_Dither");

    // // Test 2: Halftone by Error Diffusion
    // std::cout << "Test 2: Halftone by Error Diffusion" << std::endl;
    // std::function tfErrDiff3 = [](cv::Mat1f img) { return halftone::ErrDiff(img, 3); };
    // std::function tfErrDiff5 = [](cv::Mat1f img) { return halftone::ErrDiff(img, 5); };
    // testHFT("2_ErrDiff_FS", tfErrDiff3, imgList), testHFT("2_ErrDiff_JJN", tfErrDiff5, imgList);
    // sampleImg = colorconvert::mergeCh(imgResList[4], imgResList[3], imgResList[2]);
    // saveData::imgMat(sampleImg, "SpColor_ErrDiff");

    // // Test 3: Direct Binary Search (DBS) Halftoning
    // std::cout << "Test 3: Direct Binary Search (DBS) Halftoning" << std::endl;
    // std::function tfDBSK3 = [](cv::Mat1f img) { return halftone::DBS(img, 3, 1.0, 10, true); };
    // std::function tfDBSK5 = [](cv::Mat1f img) { return halftone::DBS(img, 5, 1.0, 10, true); };
    // std::function tfDBSK13 = [](cv::Mat1f img) { return halftone::DBS(img, 13, 1.0, 10, true); };
    // testHFT("3_DBS_K3", tfDBSK3, imgList), testHFT("3_DBS_K5", tfDBSK5, imgList), testHFT("3_DBS_K13", tfDBSK13, imgList);
    // sampleImg = colorconvert::mergeCh(imgResList[4], imgResList[3], imgResList[2]);
    // saveData::imgMat(sampleImg, "SpColor_DBS");

    // // Test 4: Direct Binary Search (DBS), Start with cv::Mat1f::zeros
    // std::cout << "Test 4: Direct Binary Search (DBS), Start with cv::Mat1f::zeros" << std::endl;
    // std::function tfDBSK3Z = [](cv::Mat1f img) { return halftone::DBS(img, cv::Mat1f::zeros(img.size()), 3, 1.0, 10, true); };
    // std::function tfDBSK5Z = [](cv::Mat1f img) { return halftone::DBS(img, cv::Mat1f::zeros(img.size()), 5, 1.0, 10, true); };
    // std::function tfDBSK13Z = [](cv::Mat1f img) { return halftone::DBS(img, cv::Mat1f::zeros(img.size()), 13, 1.0, 10, true); };
    // testHFT("4_DBS_K3Z", tfDBSK3Z, imgList), testHFT("4_DBS_K5Z", tfDBSK5Z, imgList), testHFT("4_DBS_K13Z", tfDBSK13Z, imgList);
    // sampleImg = colorconvert::mergeCh(imgResList[4], imgResList[3], imgResList[2]);
    // saveData::imgMat(sampleImg, "SpColor_DBSZ");

    // Test 5: Random Tiled Blocks Direct Binary Search (RTB-DBS)
    std::cout << "Test 5: Random Tiled Blocks Direct Binary Search (RTB-DBS)" << std::endl;
    dithMap = halftone::VoidCluster(halftone::getRandBin(cv::Vec2i(8, 8)), 3, 1.0f, false);
    std::function tfRTBDBSK3 = [](cv::Mat1f img) { return halftone::RTBDBS(img, cv::Mat1f::zeros(img.size()), dithMap, 3, 1.0, 10, true); };
    std::function tfRTBDBSK5 = [](cv::Mat1f img) { return halftone::RTBDBS(img, cv::Mat1f::zeros(img.size()), dithMap, 5, 1.0, 10, true); };
    std::function tfRTBDBSK13 = [](cv::Mat1f img) { return halftone::RTBDBS(img, cv::Mat1f::zeros(img.size()), dithMap, 13, 1.0, 10, true); };
    testHFT("5_RTBDBS_K3", tfRTBDBSK3, imgList), testHFT("5_RTBDBS_K5", tfRTBDBSK5, imgList), testHFT("5_RTBDBS_K13", tfRTBDBSK13, imgList);
    sampleImg = colorconvert::mergeCh(imgResList[4], imgResList[3], imgResList[2]);
    saveData::imgMat(sampleImg, "SpColor_RTBDBS");

    std::cout << "All Tests Done!" << std::endl;
    return 0;
}

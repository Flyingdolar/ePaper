#include "Functions.hpp"

// The LAB Value of the Color Checker
std::vector<cv::Vec3f> LABCC = {
    {27.6, -1.3, -1.6},  // 1: Dark Skin
    {37.6, -1.8, -2.1},  // 2: Light Skin
    {33.5, -3.7, -4.1},  // 3: Blue Sky
    {28.3, -3.2, -1.3},  // 4: Foliage
    {35.5, -2.8, -4.3},  // 5: Blue Flower
    {37.1, -5.8, -3.2},  // 6: Bluish Green

    {32.5, 0.5, 1.1},    // 7: Orange
    {30.6, -2.9, -5.3},  // 8: Purplish Blue
    {32.5, 0.8, -2.1},   // 9: Moderate Red
    {27.1, -1.3, -3.9},  // 10: Purple
    {34.6, -3.9, 1},     // 11: Yellow Green
    {35.3, -0.3, 1.4},   // 12: Orange Yellow

    {25.9, -2.7, -5.2},  // 13: Blue
    {28.9, -5, -0.7},    // 14: Green
    {26.7, 1.9, -1.6},   // 15: Red
    {36.8, -1.3, 2.9},   // 16: Yellow
    {34, 0.4, -4},       // 17: Magenta
    {26.9, -6.8, -4.4},  // 18: Cyan

    {25.1, -2.8, -12.3},  // 19: Blue (Max)
    {30, -11.9, 5.6},     // 20: Green (Max)
    {25.2, 9.4, 1.6},     // 21: Red (Max)
    {41.5, -2.2, 6.4},    // 22: Yellow (Max)
    {37.4, 5.4, -7.8},    // 23: Magenta (Max)
    {41.3, -11.4, -3.8},  // 24: Cyan (Max)

    {49.7, -3.3, -1.4},  // 25: White (Max)
    {47.2, -3.3, -1.8},  // 26: White
    {42.5, -3.2, -2.9},  // 27: Neutral 8
    {37.9, -3.2, -3},    // 28: Neutral 6.5
    {32.4, -2.7, -2.9},  // 29: Neutral 5
    {27.1, -2.4, -2.6},  // 30: Neutral 3.5
    {21.2, -2, -2.3},    // 31: Black
    {12.8, -1.1, -1.7},  // 32: Black (Max)
};

cv::Mat3f ViewAC(const cv::Mat3f valCC, const cv::Mat3f valCG, int blkSize = 100, int borderSize = 20, bool useGamma = true) {
    int rows = valCC.rows + valCG.rows, cols = std::max(valCC.cols, valCG.cols);
    int imgH = rows * blkSize + (rows + 1) * borderSize, imgW = cols * blkSize + (cols + 1) * borderSize;
    int heightCC = valCC.rows * blkSize + (valCC.rows + 1) * borderSize, heightCG = valCG.rows * blkSize + (valCG.rows + 1) * borderSize;
    int widthCC = valCC.cols * blkSize + (valCC.cols + 1) * borderSize, widthCG = valCG.cols * blkSize + (valCG.cols + 1) * borderSize;
    cv::Mat3f viewImg(imgH, imgW, 0.0f);

    // Draw Color Checker: Color Part
    for (int rowCC = 0; rowCC < valCC.rows; rowCC++)
        for (int colCC = 0; colCC < valCC.cols; colCC++) {
            int stRow = 0 + rowCC * blkSize + (rowCC + 1) * borderSize;
            int stCol = imgW / 2 - widthCC / 2 + colCC * blkSize + (colCC + 1) * borderSize;
            viewImg(cv::Rect(stCol, stRow, blkSize, blkSize)) = valCC(rowCC, colCC);
        }
    // Draw Color Checker: Gray Part
    for (int rowCG = 0; rowCG < valCG.rows; rowCG++)
        for (int colCG = 0; colCG < valCG.cols; colCG++) {
            int stRow = heightCC + rowCG * blkSize + rowCG * borderSize, blkHeight = blkSize;
            int stCol = imgW / 2 - widthCG / 2 + colCG * blkSize + (colCG + 1) * borderSize;
            if (colCG == 0 || colCG == valCG.cols - 1) stRow = borderSize, blkHeight = imgH - 2 * borderSize;
            viewImg(cv::Rect(stCol, stRow, blkSize, blkHeight)) = valCG(rowCG, colCG);
        }
    if (useGamma) viewImg = colorconvert::cvtColor(viewImg, colorconvert::lRGB2gRGB);
    cv::cvtColor(viewImg, viewImg, cv::COLOR_RGB2BGR);
    return viewImg;
}

int main(int argc, char** argv) {
    saveData::initVar("res/test/AdjustEPD", "AdjustEPD");
    cv::Mat3f imgRGBCC(4, 6), imgRGBGC(1, 8);
    cv::Mat3f GTRGBCC(4, 6), GTRGBGC(1, 8);

    // Initialize the Color Checker Ground Truth & Images
    for (int row = 0; row < 4; row++)
        for (int col = 0; col < 6; col++)
            imgRGBCC(row, col) = colorconvert::XYZ2RGB(colorconvert::Lab2XYZ(LABCC[row * 6 + col] / 100.0f)),
                          GTRGBCC(row, col) = colorconvert::gRGB2lRGB(colorchecker::checker32[row * 6 + col] / 255.0f);
    for (int col = 0; col < 8; col++)
        imgRGBGC(0, col) = colorconvert::XYZ2RGB(colorconvert::Lab2XYZ(LABCC[24 + col] / 100.0f)),
                    GTRGBGC(0, col) = colorconvert::gRGB2lRGB(colorchecker::checker32[24 + col] / 255.0f);
    saveData::imgMat(ViewAC(imgRGBCC, imgRGBGC), "Origin_CC_lRGB");
    saveData::imgMat(ViewAC(GTRGBCC, GTRGBGC), "GT_CC_lRGB");

    // Get White Balance Gain & Bias by Middle Gray Part
    cv::Mat3f imgMidG = imgRGBGC(cv::Rect(1, 0, 6, 1)).clone(), gtMidG = GTRGBGC(cv::Rect(1, 0, 6, 1)).clone();
    std::pair<cv::Vec3f, cv::Vec3f> wbGain = whitebalance::forceWB(imgMidG, gtMidG);
    saveData::logData("WB Bias", wbGain.first), saveData::logData("WB Gain", wbGain.second);
    saveData::logData("GrayScale(Origin)", imgRGBGC);
    imgRGBCC = whitebalance::applyWB(imgRGBCC, wbGain), imgRGBGC = whitebalance::applyWB(imgRGBGC, wbGain);
    imgRGBCC = cv::max(0.0f, cv::min(1.0f, imgRGBCC)), imgRGBGC = cv::max(0.0f, cv::min(1.0f, imgRGBGC));
    saveData::imgMat(ViewAC(imgRGBCC, imgRGBGC), "WB_CC_lRGB"), saveData::logData("GrayScale(WB)", imgRGBGC);

    // Get CCM by Color Checker
    colorcorrect::CCM_2D EPDCCM;
    EPDCCM.initImg(imgRGBCC, GTRGBCC);
    EPDCCM.optbyPSO(1000, -4.0, 4.0, 1e-10, 1e-10, 2);
    cv::Mat3f imgCCM = EPDCCM.applyCCM(imgRGBCC);
    saveData::imgMat(ViewAC(imgCCM, imgRGBGC), "CCM_CC_lRGB");
    saveData::logData("CCM Matrix", EPDCCM.getCCM());
    double finalLoss = EPDCCM();
    std::cout << "Final Loss: " << finalLoss << std::endl;
    return 0;
}

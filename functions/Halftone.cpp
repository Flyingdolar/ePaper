#include "Halftone.hpp"

namespace halftone {

std::string verbosePath = "DBS_verbose";  // Global Save Path for DBS Halftoning

// Direct Binary Search (DBS) Halftoning
cv::Mat1f DBS(const cv::Mat1f img, cv::Mat1f initImg, int kernelSize, float sigma, int iters, bool verbose, std::string savePath) {
    cv::Mat1f resImg = initImg.clone(), lsErrImg = initImg - img,  // Result & Low-pass Error Image
        psfMat = cv::Mat::zeros(kernelSize, kernelSize, CV_32F);   // Point Spread Function (PSF) Kernel
    std::string workFolder = saveData::defFolder;
    int workAmount = iters * img.rows * img.cols, workCount = 0;  // Recording Work Progress
    int swapCount = 0, pixNum = img.rows * img.cols;              // Recording Swap Rate
    savePath = savePath.empty() ? verbosePath : savePath;         // Set Save Path

    // 1. Initialize Gaussian PSF Kernel & Low-pass Error Image
    for (int row = 0; row < kernelSize; row++)
        for (int col = 0; col < kernelSize; col++) {  // Create Gaussian Kernel
            float rVal = (row - kernelSize / 2) * (row - kernelSize / 2), cVal = (col - kernelSize / 2) * (col - kernelSize / 2);
            psfMat.at<float>(row, col) = std::exp(-(rVal + cVal) / (2 * sigma * sigma));
        }
    lsErrImg = filter::plConv(lsErrImg, psfMat);  // Low-pass Error Image

    // 2. DBS Halftoning Iteration
    for (int iter = 0; iter < iters; iter++) {
        for (int row = 0; row < img.rows; row++)
            for (int col = 0; col < img.cols; col++) {
                float minErr = 0;
                cv::Vec2i minPos = {-1, -1};

                if (verbose) {  // Show Progress
                    std::string title = "DBS Itr: " + std::to_string(iter + 1);
                    std::string desc = "Swap Rate: " + std::to_string((int)((float)swapCount / (float)pixNum * 100)) + "%";
                    saveData::showProgress(title, (float)workCount / (float)workAmount, desc);
                }

                // For Every Pixel in the Kernel, Calculate whether to Swap/Toggle or Not by Min Error
                for (int rdx = -1; rdx <= 1; rdx++)
                    for (int cdx = -1; cdx <= 1; cdx++) {
                        // Swap Condition
                        int nRow = row + rdx, nCol = col + cdx;
                        if (nRow < 0 || nRow >= img.rows || nCol < 0 || nCol >= img.cols) continue;
                        if (resImg(nRow, nCol) == resImg(row, col)) continue;
                        cv::Vec3i posCent = {row, col, (int)resImg(row, col) == 0 ? 1 : -1};
                        cv::Vec3i posSwap = {nRow, nCol, (int)resImg(nRow, nCol) == 0 ? 1 : -1};
                        float deltaErr = detail::deltaLpErr(lsErrImg, posCent, posSwap, kernelSize, psfMat);
                        if (deltaErr < minErr) minErr = deltaErr, minPos = {nRow, nCol};
                    }
                // Toggle Condition
                cv::Vec3i posCent = {row, col, (int)resImg(row, col) == 0 ? 1 : -1}, posSwap = {row, col, 0};
                float deltaErr = detail::deltaLpErr(lsErrImg, posCent, posSwap, kernelSize, psfMat);
                if (deltaErr < minErr) minErr = deltaErr, minPos = {row, col};

                workCount++;  // Update the Work Progress, Skip if No Swap/Toggle
                if (minPos[0] == -1 || minPos[1] == -1) continue;

                // Update the Result Image & Low-pass Error Image
                cv::Vec3i newCent = {row, col, (int)resImg(row, col) == 0 ? 1 : -1};
                cv::Vec3i newSwap = {minPos[0], minPos[1], (int)resImg(minPos[0], minPos[1]) == 0 ? 1 : -1};
                if (newCent[0] == newSwap[0] && newCent[1] == newSwap[1]) newSwap[2] = 0;
                resImg(newCent[0], newCent[1]) += newCent[2], resImg(newSwap[0], newSwap[1]) += newSwap[2];
                lsErrImg = detail::altLpErr(lsErrImg, newCent, kernelSize, psfMat);
                lsErrImg = detail::altLpErr(lsErrImg, newSwap, kernelSize, psfMat);
                swapCount++;  // Update the Swap Rate & Work Progress
            }
        // Verbose Show the Result of Each Iteration
        if (verbose) saveData::initVar(workFolder + "/" + savePath, "DBSlog");
        if (verbose) saveData::imgMat(resImg, "resImg_" + std::to_string(iter + 1)), saveData::imgMat(lsErrImg, "errImg_" + std::to_string(iter + 1));
        // Verbose Save log for Each Iteration
        float errVal = 0;
        for (int row = 0; row < img.rows; row++)
            for (int col = 0; col < img.cols; col++) errVal += std::abs(lsErrImg(row, col));
        errVal /= (float)(img.rows * img.cols);
        if (verbose) saveData::logData("Iter " + std::to_string(iter + 1) + " Error", errVal);
        if (verbose) saveData::logData("Iter " + std::to_string(iter + 1) + " Swap Rate", (float)swapCount / (float)pixNum * 100.0f);
        if (verbose) saveData::initVar(workFolder);
        swapCount = 0;  // Reset the Swap Rate
    }

    return resImg;
}

// Dithering Halftoning
cv::Mat1f Dither(const cv::Mat1f grayImg, int kernelSize, bool verbose) {
    int height = grayImg.rows, width = grayImg.cols;
    cv::Mat1f resImg = grayImg.clone();

    // 1. Create Dither Matrix
    cv::Mat1b ditherMat;
    if (kernelSize == 2) ditherMat = tMap2;  // 2x2 Dithering Matrix
    if (kernelSize == 4) ditherMat = tMap4;  // 4x4 Dithering Matrix
    if (kernelSize == 8) ditherMat = tMap8;  // 8x8 Dithering Matrix
    if (ditherMat.empty()) {
        std::cerr << "Dithering Kernel Size is not Supported!" << std::endl;
        return resImg;
    }

    // 2. Dithering Process
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++) {
            int kRow = row % kernelSize, kCol = col % kernelSize;
            if (grayImg(row, col) > (float)ditherMat(kRow, kCol) / (float)(kernelSize * kernelSize))
                resImg(row, col) = 1;
            else
                resImg(row, col) = 0;
        }
    return resImg;
}

// Error Diffusion Halftoning
cv::Mat1f ErrDiff(const cv::Mat1f grayImg, int kernelSize, bool verbose) {
    int height = grayImg.rows, width = grayImg.cols;
    cv::Mat1f resImg = grayImg.clone(), errImg = cv::Mat1f::zeros(height, width);

    // 1. Create Error Diffusion Kernel
    cv::Mat1b errKernel;
    if (kernelSize == 3) errKernel = kFloydSteinberg;  // Floyd-Steinberg Kernel
    if (kernelSize == 5) errKernel = kJJN;             // JJN Kernel
    if (errKernel.empty()) {
        std::cerr << "Error Diffusion Kernel Size is not Supported!" << std::endl;
        return resImg;
    }

    // 2. Error Diffusion Process
    int kSum = cv::sum(errKernel)[0];
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++) {
            float grayVal = grayImg(row, col) + errImg(row, col);
            float diffVal = grayVal - ((grayVal > 0.5) ? 1 : 0);

            resImg(row, col) = (grayVal > 0.5) ? 1 : 0;  // Update the Result Image
            for (int rdx = 0; rdx < errKernel.rows; rdx++)
                for (int cdx = 0; cdx < errKernel.cols; cdx++) {  // Diffuse the Error
                    int nRow = row + rdx, nCol = (col + cdx) - (errKernel.cols / 2);
                    if (nRow < 0 || nRow >= height || nCol < 0 || nCol >= width) continue;
                    errImg(nRow, nCol) += (errKernel(rdx, cdx) / (float)kSum) * diffVal;
                }
        }
    return resImg;
}

}  // namespace halftone

namespace halftone::detail {  // Detail Functions
// Calculate Delta Error for Swap/Toggle Condition
float deltaLpErr(const cv::Mat1f lpErrImg, cv::Vec3i posCent, cv::Vec3i posSwap, int kSize, const cv::Mat1f gskMat) {
    float deltaErr = 0;
    int togCent = posCent[2], togSwap = posSwap[2];

    for (int rdx = (-kSize / 2) - 1; rdx <= (kSize / 2) + 1; rdx++)
        for (int cdx = (-kSize / 2) - 1; cdx <= (kSize / 2) + 1; cdx++) {
            float smallDeltaE = 0;
            int pixRow = posCent[0] + rdx, pixCol = posCent[1] + cdx;
            cv::Vec2i centDist = {rdx + kSize / 2, cdx + kSize / 2},
                      swapDist = {pixRow - posSwap[0] + kSize / 2, pixCol - posSwap[1] + kSize / 2};
            if (pixRow < 0 || pixRow >= lpErrImg.rows || pixCol < 0 || pixCol >= lpErrImg.cols) continue;
            if (centDist[0] >= 0 && centDist[1] >= 0 && centDist[0] < kSize && centDist[1] < kSize)
                smallDeltaE += gskMat(centDist[0], centDist[1]) * togCent;  // Calculate Small Delta E with Center Pixel
            if (swapDist[0] >= 0 && swapDist[1] >= 0 && swapDist[0] < kSize && swapDist[1] < kSize)
                smallDeltaE += gskMat(swapDist[0], swapDist[1]) * togSwap;  // Calculate Small Delta E with Swap Pixel
            deltaErr += std::pow(smallDeltaE + lpErrImg(pixRow, pixCol), 2) - std::pow(lpErrImg(pixRow, pixCol), 2);
        }

    return deltaErr;
}

// Alter & Update the low-pass Error Image by Swap/Toggle Condition
cv::Mat1f altLpErr(const cv::Mat1f lpErrImg, cv::Vec3i posPix, int kSize, const cv::Mat1f gskMat) {
    int height = lpErrImg.rows, width = lpErrImg.cols;
    cv::Mat1f resLpErr = lpErrImg.clone();

    for (int rdx = -kSize / 2; rdx <= kSize / 2; rdx++)
        for (int cdx = -kSize / 2; cdx <= kSize / 2; cdx++) {
            int nRow = posPix[0] + rdx, nCol = posPix[1] + cdx;
            int distRow = kSize / 2 + rdx, distCol = kSize / 2 + cdx;
            if (nRow < 0 || nRow >= height || nCol < 0 || nCol >= width) continue;
            resLpErr(nRow, nCol) += gskMat(distRow, distCol) * posPix[2];
        }
    return resLpErr;
}

// Visualize Error Image
cv::Mat3f viewErr(cv::Mat1f errImg) {
    cv::Mat3f visImg(errImg.size());
    for (int row = 0; row < errImg.rows; row++)
        for (int col = 0; col < errImg.cols; col++) {
            float val = errImg(row, col);
            if (val < 0) visImg(row, col) = cv::Vec3f(0, 0, -val);  // Red for Negative
            if (val >= 0) visImg(row, col) = cv::Vec3f(val, 0, 0);  // Blue for Positive
        }
    return visImg;
}

}  // namespace halftone::detail

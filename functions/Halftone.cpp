#include "Halftone.hpp"

namespace halftone {

// Direct Binary Search (DBS) Halftoning
cv::Mat1f DBS(const cv::Mat1f grayImg, int kernelSize, float sigma, int iters, bool verbose) {
    int height = grayImg.rows, width = grayImg.cols, halfK = kernelSize / 2;
    int workProgress = iters * height * width, pixNum = height * width, workCount = 0, swapNum = 0;
    std::string workFolder = saveData::defFolder;

    // 1. Initialize Result & Error Image
    cv::Mat1f resImg = cv::Mat1f::zeros(height, width);  // Result Image
    cv::Mat1f errImg = resImg - grayImg;                 // Error Image
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++) resImg.at<float>(row, col) = (rand() % 2 == 0) ? 0 : 1;
    if (verbose) saveData::initVar(workFolder + "/Iter"), saveData::imgMat(resImg, "resImg_0");
    if (verbose) saveData::initVar(workFolder + "/Err"), saveData::imgMat(detail::viewErr(errImg), "errImg_0");

    // 2. Create Gaussian Kernel
    cv::Mat1f gskMat2D = cv::Mat::zeros(kernelSize, kernelSize, CV_32F);
    for (int row = 0; row < kernelSize; row++)
        for (int col = 0; col < kernelSize; col++) {
            float rVal = (row - halfK) * (row - halfK), cVal = (col - halfK) * (col - halfK);
            gskMat2D.at<float>(row, col) = std::exp(-(rVal + cVal) / (2 * sigma * sigma));
        }
    if (verbose) saveData::initVar(workFolder + "/Kernel"), saveData::imgMat(gskMat2D, "GaussianKernel");

    // 3. Initialize the low-pass Error Image
    cv::Mat1f lpErrImg = cv::Mat1f::zeros(height, width);
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++)
            for (int rdx = -halfK; rdx <= halfK; rdx++)
                for (int cdx = -halfK; cdx <= halfK; cdx++) {
                    int nRow = row + rdx, nCol = col + cdx;
                    if (nRow < 0 || nRow >= height || nCol < 0 || nCol >= width) continue;
                    lpErrImg.at<float>(row, col) += gskMat2D(rdx + halfK, cdx + halfK) * errImg.at<float>(nRow, nCol);
                }

    // 4. Generate Random List
    std::vector<cv::Vec2i> randList;
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++) randList.push_back({row, col});
    std::random_shuffle(randList.begin(), randList.end());

    // 4. DBS Halftoning Iteration
    for (int iter = 0; iter < iters; iter++) {
        for (cv::Vec2i workPix : randList) {
            int row = workPix[0], col = workPix[1];
            // Show the Process
            if (verbose) workCount++;
            if (verbose && workCount % 100 == 0) {
                float workRate = (float)workCount / (float)workProgress * 100.0;
                float swapRate = (float)swapNum / (float)pixNum * 100.0;
                std::cout << "... DBS Halftone [Iteration " << iter + 1 << " | " << std::fixed << std::setprecision(2) << workRate << "% Completed |" << swapRate << "% Swapped]\r";
                std::cout.flush();  // Flush the Output
            }

            // 4-1. MinError Decision
            cv::Vec2i posME = detail::getMinEPos(resImg, lpErrImg, {row, col}, kernelSize, gskMat2D);
            if (posME[0] < 0 && posME[1] < 0) continue;  // Skip if no minimum error
            swapNum++;                                   // Increase the Swap Number if not skipped

            // 4-2. Update the Result & Error Image
            std::vector<cv::Vec3i> posAlt = {{posME[0], posME[1], (int)resImg.at<float>(posME[0], posME[1])}};
            if (posME[0] != row || posME[1] != col) posAlt.push_back({row, col, (int)resImg.at<float>(row, col)});
            for (cv::Vec3i pVal : posAlt) resImg.at<float>(pVal[0], pVal[1]) = (pVal[2] == 0) ? 1 : 0;  // Update the Result Image

            // 4-3. Update the low-pass Error Image
            lpErrImg = detail::altLpErr(lpErrImg, posAlt, kernelSize, gskMat2D);
        }
        // Update the Error Image
        if (verbose) errImg = resImg - grayImg;

        // Save the Iteration Result
        if (verbose) saveData::initVar(workFolder + "/Iter"), saveData::imgMat(resImg, "resImg_" + std::to_string(iter + 1)), swapNum = 0;
        if (verbose) saveData::initVar(workFolder + "/Err"), saveData::imgMat(detail::viewErr(errImg), "errImg_" + std::to_string(iter + 1));
        if (verbose) saveData::initVar(workFolder + "/LpErr"), saveData::imgMat(lpErrImg, "lpErrImg_" + std::to_string(iter + 1));
        if (verbose) saveData::initVar(workFolder);
    }
    return resImg;
}

// Random DBS Halftoning
cv::Mat1f RTBDBS(const cv::Mat1f grayImg, int kernelSize, float sigma, int iters, bool verbose) {
    int height = grayImg.rows, width = grayImg.cols, halfK = kernelSize / 2;
    int workProgress = iters * height * width, pixNum = height * width, workCount = 0, swapNum = 0;
    std::string workFolder = saveData::defFolder;

    // 1. Initialize Result & Error Image
    cv::Mat1f resImg = cv::Mat1f::zeros(height, width);  // Result Image
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++) resImg.at<float>(row, col) = (rand() % 2 == 0) ? 0 : 1;
    cv::Mat1f errImg = resImg - grayImg;  // Error Image
    if (verbose) saveData::initVar(workFolder + "/Iter"), saveData::imgMat(resImg, "resImg_0");
    if (verbose) saveData::initVar(workFolder + "/Err"), saveData::imgMat(detail::viewErr(errImg), "errImg_0");

    // 2. Create Gaussian Kernel
    cv::Mat1f gskMat2D = cv::Mat::zeros(kernelSize, kernelSize, CV_32F);
    for (int row = 0; row < kernelSize; row++)
        for (int col = 0; col < kernelSize; col++) {
            float rVal = (row - halfK) * (row - halfK), cVal = (col - halfK) * (col - halfK);
            gskMat2D.at<float>(row, col) = std::exp(-(rVal + cVal) / (2 * sigma * sigma));
        }
    if (verbose) saveData::initVar(workFolder + "/Kernel"), saveData::imgMat(gskMat2D, "GaussianKernel");

    // 3. Initialize the low-pass Error Image
    cv::Mat1f lpErrImg = cv::Mat1f::zeros(height, width);
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++)
            for (int rdx = -halfK; rdx <= halfK; rdx++)
                for (int cdx = -halfK; cdx <= halfK; cdx++) {
                    int nRow = row + rdx, nCol = col + cdx;
                    if (nRow < 0 || nRow >= height || nCol < 0 || nCol >= width) continue;
                    lpErrImg.at<float>(row, col) += gskMat2D(rdx + halfK, cdx + halfK) * errImg.at<float>(nRow, nCol);
                }

    // 4. DBS Halftoning Iteration
    for (int iter = 0; iter < iters; iter++) {
        for (int row = 0; row < height; row++)
            for (int col = 0; col < width; col++) {
                // Show the Process
                if (verbose) workCount++;
                if (verbose && workCount % 100 == 0) {
                    float workRate = (float)workCount / (float)workProgress * 100.0;
                    float swapRate = (float)swapNum / (float)pixNum * 100.0;
                    std::cout << "... DBS Halftone [Iteration " << iter + 1 << " | " << std::fixed << std::setprecision(2) << workRate << "% Completed |" << swapRate << "% Swapped]\r";
                    std::cout.flush();  // Flush the Output
                }

                // 4-1. MinError Decision
                cv::Vec2i posME = detail::getMinEPos(resImg, lpErrImg, {row, col}, kernelSize, gskMat2D);
                if (posME[0] < 0 && posME[1] < 0) continue;  // Skip if no minimum error
                swapNum++;                                   // Increase the Swap Number if not skipped

                // 4-2. Update the Result & Error Image
                std::vector<cv::Vec3i> posAlt = {{posME[0], posME[1], (int)resImg.at<float>(posME[0], posME[1])}};
                if (posME[0] != row || posME[1] != col) posAlt.push_back({row, col, (int)resImg.at<float>(row, col)});
                for (cv::Vec3i pVal : posAlt) resImg.at<float>(pVal[0], pVal[1]) = (pVal[2] == 0) ? 1 : 0;  // Update the Result Image

                // 4-3. Update the low-pass Error Image
                lpErrImg = detail::altLpErr(lpErrImg, posAlt, kernelSize, gskMat2D);
            }
        // Update the Error Image
        if (verbose) errImg = resImg - grayImg;

        // Save the Iteration Result
        if (verbose) saveData::initVar(workFolder + "/Iter"), saveData::imgMat(resImg, "resImg_" + std::to_string(iter + 1)), swapNum = 0;
        if (verbose) saveData::initVar(workFolder + "/Err"), saveData::imgMat(detail::viewErr(errImg), "errImg_" + std::to_string(iter + 1));
        if (verbose) saveData::initVar(workFolder + "/LpErr"), saveData::imgMat(detail::viewErr(lpErrImg), "lpErrImg_" + std::to_string(iter + 1));
        if (verbose) saveData::initVar(workFolder);
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
    return grayImg;
}

}  // namespace halftone

namespace halftone::detail {  // Detail Functions
// Alter & Update the low-pass Error Image
cv::Mat1f altLpErr(const cv::Mat1f lpErrImg, std::vector<cv::Vec3i> pos, int kernelSize, const cv::Mat1f gskMat2D) {
    int height = lpErrImg.rows, width = lpErrImg.cols, halfK = kernelSize / 2;
    cv::Mat1f newLpErrImg = lpErrImg.clone();

    for (cv::Vec3i pVal : pos)
        for (int rdx = -halfK; rdx <= halfK; rdx++)
            for (int cdx = -halfK; cdx <= halfK; cdx++) {
                int nRow = pVal[0] + rdx, nCol = pVal[1] + cdx;
                int distRow = kernelSize / 2 + rdx, distCol = kernelSize / 2 + cdx;
                if (nRow < 0 || nRow >= height || nCol < 0 || nCol >= width) continue;
                newLpErrImg.at<float>(nRow, nCol) += gskMat2D(distRow, distCol) * ((pVal[2] == 0) ? 1 : -1);
            }
    return newLpErrImg;
}

// Choose the Minimum Error Position
cv::Vec2i getMinEPos(const cv::Mat1f resImg, const cv::Mat1f lpErrImg, cv::Vec2i centPos, int kernelSize, const cv::Mat1f gskMat2D) {
    int height = resImg.rows, width = resImg.cols, halfK = kernelSize / 2;
    float minDscErr = 0;
    cv::Vec2i posME = {-1, -1};  // Minimum Error Position
    int movHalfK = 1;            // Moving Half Kernel Size

    // For Every Pixel in the Kernel, Calculate the Descending Error
    for (int rdx = -movHalfK; rdx <= movHalfK; rdx++)
        for (int cdx = -movHalfK; cdx <= movHalfK; cdx++) {
            cv::Vec2i nbrPos = {centPos[0] + rdx, centPos[1] + cdx};
            if (nbrPos[0] < 0 || nbrPos[0] >= height || nbrPos[1] < 0 || nbrPos[1] >= width) continue;
            if (centPos != nbrPos)  // Skip Neighbor with same value
                if (resImg.at<float>(nbrPos[0], nbrPos[1]) == resImg.at<float>(centPos[0], centPos[1]) && centPos != nbrPos) continue;
            int modCentPos = (resImg.at<float>(centPos[0], centPos[1]) == 0) ? 1 : -1;
            int modNPos = (resImg.at<float>(nbrPos[0], nbrPos[1]) == 0) ? 1 : -1;

            // Collect all Error Value Altered Pixels
            std::vector<cv::Vec6i> posVecs;  // {Row, Col, Distance to Center(Row, Col), Distance to Neighbor(Row, Col) }
            for (int rdxAlt = -movHalfK; rdxAlt <= movHalfK; rdxAlt++)
                for (int cdxAlt = -movHalfK; cdxAlt <= movHalfK; cdxAlt++) {
                    cv::Vec2i posAlt = {centPos[0] + rdxAlt, centPos[1] + cdxAlt}, centDist = {rdxAlt, cdxAlt};
                    cv::Vec2i nbrDist = {posAlt[0] - nbrPos[0], posAlt[1] - nbrPos[1]};
                    if (posAlt[0] < 0 || posAlt[0] >= height || posAlt[1] < 0 || posAlt[1] >= width) continue;
                    posVecs.push_back({posAlt[0], posAlt[1], centDist[0], centDist[1], nbrDist[0], nbrDist[1]});
                    if (centDist[0] < -halfK || centDist[0] > halfK || centDist[1] < -halfK || centDist[1] > halfK)
                        posVecs.back()[2] = -1, posVecs.back()[3] = -1;  // Skip the Position Far from the Center
                    if (nbrDist[0] < -halfK || nbrDist[0] > halfK || nbrDist[1] < -halfK || nbrDist[1] > halfK)
                        posVecs.back()[4] = -1, posVecs.back()[5] = -1;                      // Skip the Position Far from the Neighbor
                    if (centPos == nbrPos) posVecs.back()[4] = -1, posVecs.back()[5] = -1;   // Skip the Same Position
                    if (posVecs.back()[3] < 0 && posVecs.back()[5] < 0) posVecs.pop_back();  // Remove Not-Valid Position
                }

            // Calculate the Descending Error
            float dscErr = 0;
            for (cv::Vec6i pVal : posVecs) {
                cv::Vec2i centDist = {pVal[2] + halfK, pVal[3] + halfK}, nbrDist = {pVal[4] + halfK, pVal[5] + halfK};
                float deltaE = 0;
                // First Calculate Small Delta E
                if (centDist[0] > 0 && centDist[1] > 0) deltaE += gskMat2D(centDist[0], centDist[1]) * modCentPos;
                if (nbrDist[0] > 0 && nbrDist[1] > 0) deltaE += gskMat2D(nbrDist[0], nbrDist[1]) * modNPos;
                // Then Transfer to Big Delta E and get Sum
                dscErr += std::pow(deltaE + lpErrImg.at<float>(pVal[0], pVal[1]), 2) - std::pow(lpErrImg.at<float>(pVal[0], pVal[1]), 2);
            }
            if (dscErr < minDscErr) minDscErr = dscErr, posME = nbrPos;  // Update the Minimum Error Position
        }
    return posME;
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

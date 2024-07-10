#include "Filter.hpp"

namespace filter {

// Convolution on Parallel Kernel
cv::Mat plConv(cv::Mat img, cv::Mat kernel) {
    cv::Mat resImg = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    for (int row = 0; row < img.rows; row++)
        for (int col = 0; col < img.cols; col++)
            for (int rdx = 0; rdx <= kernel.rows / 2; rdx++)
                for (int cdx = 0; cdx <= kernel.cols / 2; cdx++) {
                    int nRow = row + rdx, nCol = col + cdx;
                    int kRow = rdx + kernel.rows / 2, kCol = cdx + kernel.cols / 2;
                    if (nRow < 0 || nRow >= img.rows || nCol < 0 || nCol >= img.cols) continue;
                    resImg.at<float>(nRow, nCol) += img.at<float>(row, col) * kernel.at<float>(kRow, kCol);
                    resImg.at<float>(row, col) += img.at<float>(nRow, nCol) * kernel.at<float>(kRow, kCol);
                }
    return resImg;
}

// Convolution on Normal Kernel
cv::Mat conv(cv::Mat img, cv::Mat kernel) {
    cv::Mat resImg = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    for (int row = 0; row < img.rows; row++)
        for (int col = 0; col < img.cols; col++)
            for (int rdx = -kernel.rows / 2; rdx <= kernel.rows / 2; rdx++)
                for (int cdx = -kernel.cols / 2; cdx <= kernel.cols / 2; cdx++) {
                    int nRow = row + rdx, nCol = col + cdx;
                    int kRow = rdx + kernel.rows / 2, kCol = cdx + kernel.cols / 2;
                    if (nRow < 0 || nRow >= img.rows || nCol < 0 || nCol >= img.cols) continue;
                    resImg.at<float>(row, col) += img.at<float>(nRow, nCol) * kernel.at<float>(kRow, kCol);
                }
    return resImg;
}

cv::Mat gaussian(cv::Mat img, int kernelSize, float sigma) {
    cv::Mat resImg = img.clone();
    resImg.convertTo(resImg, CV_32F);
    cv::Mat kernel = cv::Mat::zeros(kernelSize, kernelSize, CV_32F);
    float sum = 0;
    for (int row = 0; row < kernelSize; row++)
        for (int col = 0; col < kernelSize; col++) {
            float val = std::exp(-0.5 * (std::pow(row - kernelSize / 2, 2) + std::pow(col - kernelSize / 2, 2)) / std::pow(sigma, 2));
            kernel.at<float>(row, col) = val, sum += val;
        }
    kernel /= sum;
    for (int row = 0; row < img.rows; row++)
        for (int col = 0; col < img.cols; col++) {
            float sum = 0;
            for (int rdx = -kernelSize / 2; rdx <= kernelSize / 2; rdx++)
                for (int cdx = -kernelSize / 2; cdx <= kernelSize / 2; cdx++) {
                    int nRow = row + rdx, nCol = col + cdx;
                    if (nRow < 0 || nRow >= img.rows || nCol < 0 || nCol >= img.cols) continue;
                    sum += img.at<float>(nRow, nCol) * kernel.at<float>(rdx + kernelSize / 2, cdx + kernelSize / 2);
                }
            resImg.at<float>(row, col) = sum;
        }
    return resImg;
}

// Local Edge Preserving Filter
cv::Mat localEP(cv::Mat img, int kernelSize, float alpha, float beta, int iters) {
    cv::Mat meanImg = mean(img, kernelSize), varImg = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    cv::Mat gradB2Img = cv::Mat::zeros(img.rows, img.cols, CV_32F);  // Gradient Based on sum( |I(x) - I(y)| ^ (2 - beta) )
    cv::Mat coefA = cv::Mat::zeros(img.rows, img.cols, CV_32F), coefB = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    cv::Mat resImg = img.clone(), lossE = cv::Mat::zeros(img.rows, img.cols, CV_32F);

    // 1. Calculate Mean, Variance, and Gradient
    for (int row = 0; row < img.rows; row++)
        for (int col = 0; col < img.cols; col++) {
            int counter = 0;
            float diffSum = 0, gradSum = 0;
            for (int rdx = -kernelSize / 2; rdx <= kernelSize / 2; rdx++)
                for (int cdx = -kernelSize / 2; cdx <= kernelSize / 2; cdx++) {
                    int nRow = row + rdx, nCol = col + cdx;
                    if (nRow < 0 || nRow >= img.rows || nCol < 0 || nCol >= img.cols) continue;
                    // Calculate Difference and Gradient Sum
                    diffSum += std::pow(std::abs(img.at<float>(nRow, nCol) - meanImg.at<float>(nRow, nCol)), 2), counter++;
                    gradSum += std::pow(std::abs(img.at<float>(row, col) - img.at<float>(nRow, nCol)), 2 - beta);
                }
            varImg.at<float>(row, col) = diffSum, gradB2Img.at<float>(row, col) = gradSum / counter;
        }
    saveData::imgMat(meanImg, "res/testBug", "meanImg", 1, false);
    saveData::imgMat(varImg, "res/testBug", "varImg", 1, false);
    saveData::imgMat(gradB2Img, "res/testBug", "gradB2Img", 1, false);

    // 2. Initialize Coefficients and Loss Energy
    for (int row = 0; row < img.rows; row++)
        for (int col = 0; col < img.cols; col++) {
            int counter = 0;
            float varVal = varImg.at<float>(row, col), gradVal = gradB2Img.at<float>(row, col);
            float meanVal = meanImg.at<float>(row, col), lossVal = 0;
            // 2-1. Calculate Coefficients
            coefA.at<float>(row, col) = varVal <= 0.01 ? 0 : varVal * varVal / (varVal * varVal + gradVal * alpha);
            coefB.at<float>(row, col) = meanVal - coefA.at<float>(row, col) * meanVal;
            // 2-2. Calculate Loss Energy
            for (int rdx = -kernelSize / 2; rdx <= kernelSize / 2; rdx++)
                for (int cdx = -kernelSize / 2; cdx <= kernelSize / 2; cdx++) {
                    int nRow = row + rdx, nCol = col + cdx;
                    if (nRow < 0 || nRow >= img.rows || nCol < 0 || nCol >= img.cols) continue;
                    float imgVal = img.at<float>(nRow, nCol), coefAVal = coefA.at<float>(row, col), coefBVal = coefB.at<float>(row, col);
                    lossVal += std::pow((imgVal - coefAVal * imgVal - coefBVal), 2), counter++;
                }
            lossE.at<float>(row, col) = lossVal + alpha * gradB2Img.at<float>(row, col) * std::pow(coefA.at<float>(row, col), 2);
            lossE.at<float>(row, col) /= counter;
        }
    saveData::imgMat(coefA, "res/testBug", "coefA", 1, false);
    saveData::imgMat(coefB, "res/testBug", "coefB", 1, false);

    // 3. Iterative Update
    for (int iter = 0; iter < iters; iter++) {
        for (int row = 0; row < img.rows; row++)
            for (int col = 0; col < img.cols; col++) {
                int counter = 0;
                float nValA = 0, nValB = 0, nLoss = 0;
                // 3-1. Update Coefficients (Mean Value of Last Iteration)
                for (int rdx = -kernelSize / 2; rdx <= kernelSize / 2; rdx++)
                    for (int cdx = -kernelSize / 2; cdx <= kernelSize / 2; cdx++) {
                        int nRow = row + rdx, nCol = col + cdx;
                        if (nRow < 0 || nRow >= img.rows || nCol < 0 || nCol >= img.cols) continue;
                        nValA += coefA.at<float>(nRow, nCol), nValB += coefB.at<float>(nRow, nCol), counter++;
                    }
                nValA /= counter, nValB /= counter;
                counter = 0;
                // 3-2. Calculate Loss Energy with Updated Coefficients
                for (int rdx = -kernelSize / 2; rdx <= kernelSize / 2; rdx++)
                    for (int cdx = -kernelSize / 2; cdx <= kernelSize / 2; cdx++) {
                        int nRow = row + rdx, nCol = col + cdx;
                        if (nRow < 0 || nRow >= img.rows || nCol < 0 || nCol >= img.cols) continue;
                        float imgVal = img.at<float>(nRow, nCol);
                        nLoss += std::pow((imgVal - nValA * imgVal - nValB), 2), counter++;
                    }
                nLoss = (nLoss + alpha * gradB2Img.at<float>(row, col) * std::pow(nValA, 2)) / counter;
                // 3-3. If Loss Energy is Smaller, Update Coefficients
                if (nLoss < lossE.at<float>(row, col))
                    coefA.at<float>(row, col) = nValA, coefB.at<float>(row, col) = nValB, lossE.at<float>(row, col) = nLoss;
            }
        saveData::imgMat(coefA, "res/testBug", "coefA_" + std::to_string(iter), 1, false);
        saveData::imgMat(coefB, "res/testBug", "coefB_" + std::to_string(iter), 1, false);
    }

    // 4. Apply Filter
    for (int row = 0; row < img.rows; row++)
        for (int col = 0; col < img.cols; col++)
            resImg.at<float>(row, col) = coefA.at<float>(row, col) * img.at<float>(row, col) + coefB.at<float>(row, col);
    return resImg;
}

// Bilateral Filter
cv::Mat bilateral(cv::Mat img, int kernelSize, float sigmaS, float sigmaR) {
    cv::Mat resImg = img.clone();
    resImg.convertTo(resImg, CV_32F);
    for (int row = 0; row < img.rows; row++)
        for (int col = 0; col < img.cols; col++) {
            float sum = 0, wSum = 0;  // Value and Weight Sum
            for (int rdx = -kernelSize / 2; rdx <= kernelSize / 2; rdx++)
                for (int cdx = -kernelSize / 2; cdx <= kernelSize / 2; cdx++) {
                    int nRow = row + rdx, nCol = col + cdx;
                    if (nRow < 0 || nRow >= img.rows || nCol < 0 || nCol >= img.cols) continue;
                    // Spatial Distance Weight = e^((-1/2) * (r^2 + c^2) / (sigmaS^2))
                    float spDist = std::exp(-0.5 * (std::pow(rdx, 2) + std::pow(cdx, 2)) / std::pow(sigmaS, 2));
                    // Intensity Distance Weight = e^((-1/2) * (I1 - I2)^2 / (sigmaR^2))
                    float intDist = std::exp(-0.5 * std::pow(img.at<float>(row, col) - img.at<float>(nRow, nCol), 2) / std::pow(sigmaR, 2));
                    // Weight = Spatial Distance Weight * Intensity Distance Weight
                    float weight = spDist * intDist;
                    sum += img.at<float>(nRow, nCol) * weight;
                    wSum += weight;
                }
            resImg.at<float>(row, col) = sum / wSum;
        }
    return resImg;
}

// Fast Bilateral Filter
cv::Mat fastBilateral(cv::Mat img, int kernelSize, int segment, float sigmaS, float sigmaR) {
    std::vector<float> gaussS, gaussR;
    std::vector<cv::Mat> kernels;
    cv::Mat resImg = img.clone();
    float step = 3 * sigmaR / segment;

    // Generate Gaussian Weights
    for (int idx = 0; idx <= kernelSize / 2; idx++) {  // Spatial Weights
        float weight = std::exp(-0.5 * std::pow(idx, 2) / std::pow(sigmaS, 2));
        gaussS.push_back(weight);
    }
    for (int idx = 0; idx <= segment; idx++) {  // Intensity Weights
        float weight = std::exp(-0.5 * std::pow(idx * step, 2) / std::pow(sigmaR, 2));
        gaussR.push_back(weight);
    }

    // Evaluate Kernel Parameters
    for (int idx = 0; idx < segment; idx++) {
        cv::Mat kernel = cv::Mat::zeros(kernelSize + 1, kernelSize + 1, CV_32F);
        for (int rdx = -kernelSize / 2; rdx <= kernelSize / 2; rdx++)
            for (int cdx = -kernelSize / 2; cdx <= kernelSize / 2; cdx++) {
                float spWeight = gaussS[std::abs(rdx)] * gaussS[std::abs(cdx)];
                float intWeight = gaussR[std::abs(idx)];
                kernel.at<float>(rdx + kernelSize / 2, cdx + kernelSize / 2) = spWeight * intWeight;
            }
        kernels.push_back(kernel);
    }

    // Fast Bilateral Filter
    resImg.convertTo(resImg, CV_32F);
    for (int row = 0; row < img.rows; row++)
        for (int col = 0; col < img.cols; col++) {
            float sum = 0, wSum = 0;  // Value and Weight Sum
            for (int rdx = -kernelSize / 2; rdx <= kernelSize / 2; rdx++)
                for (int cdx = -kernelSize / 2; cdx <= kernelSize / 2; cdx++) {
                    int nRow = row + rdx, nCol = col + cdx, kRow = rdx + kernelSize / 2, kCol = cdx + kernelSize / 2;
                    if (nRow < 0 || nRow >= img.rows || nCol < 0 || nCol >= img.cols) continue;
                    int intGap = int(std::abs(img.at<float>(row, col) - img.at<float>(nRow, nCol)) / sigmaS);
                    if (intGap < 0 || intGap > 3) continue;  // Skip Large Intensity Gap
                    sum += img.at<float>(nRow, nCol) * kernels[intGap].at<float>(kRow, kCol);
                    wSum += kernels[intGap].at<float>(kRow, kCol);
                }
            if (wSum == 0) wSum = 1;
            resImg.at<float>(row, col) = sum / wSum;
        }
    return resImg;
}

// Similar Filter
cv::Mat similar(cv::Mat img, int kernelSize, float lowRate, float highRate) {
    cv::Mat resImg = img.clone();
    resImg.convertTo(resImg, CV_32F);
    for (int row = 0; row < img.rows; row++)
        for (int col = 0; col < img.cols; col++) {
            float sum = 0, wSum = 0;  // Value and Weight Sum
            for (int rdx = -kernelSize / 2; rdx <= kernelSize / 2; rdx++)
                for (int cdx = -kernelSize / 2; cdx <= kernelSize / 2; cdx++) {
                    int nRow = row + rdx, nCol = col + cdx;
                    if (nRow < 0 || nRow >= img.rows || nCol < 0 || nCol >= img.cols) continue;
                    if (img.at<float>(nRow, nCol) < img.at<float>(row, col) * lowRate) continue;
                    if (img.at<float>(nRow, nCol) > img.at<float>(row, col) * highRate) continue;
                    sum += img.at<float>(nRow, nCol), wSum++;
                }
            resImg.at<float>(row, col) = sum / wSum;
        }
    return resImg;
}

// Mean Filter
cv::Mat mean(cv::Mat img, int kernelSize) {
    cv::Mat resImg = img.clone();
    resImg.convertTo(resImg, CV_32F);
    for (int row = 0; row < img.rows; row++)
        for (int col = 0; col < img.cols; col++) {
            float sum = 0;  // Value Sum
            int count = 0;  // Value Count
            for (int rdx = -kernelSize / 2; rdx <= kernelSize / 2; rdx++)
                for (int cdx = -kernelSize / 2; cdx <= kernelSize / 2; cdx++) {
                    int nRow = row + rdx, nCol = col + cdx;
                    if (nRow < 0 || nRow >= img.rows || nCol < 0 || nCol >= img.cols) continue;
                    sum += img.at<float>(nRow, nCol);
                    count++;
                }
            resImg.at<float>(row, col) = sum / count;
        }
    return resImg;
}

// Median Filter
cv::Mat median(cv::Mat img, int kernelSize) {
    cv::Mat resImg = img.clone();
    resImg.convertTo(resImg, CV_32F);
    for (int row = 0; row < img.rows; row++)
        for (int col = 0; col < img.cols; col++) {
            std::vector<float> nbVals;  // Neighbour Values
            for (int i = -kernelSize / 2; i <= kernelSize / 2; i++)
                for (int j = -kernelSize / 2; j <= kernelSize / 2; j++) {
                    int nRow = row + i, nCol = col + j;
                    if (nRow >= 0 && nRow < img.rows && nCol >= 0 && nCol < img.cols)
                        nbVals.push_back(img.at<float>(nRow, nCol));
                }
            std::sort(nbVals.begin(), nbVals.end());                 // Sort Neighbour Values
            resImg.at<float>(row, col) = nbVals[nbVals.size() / 2];  // Get Median Value
        }
    return resImg;
}

// Sub Window Box Filter
cv::Mat subWBox(cv::Mat img, int kernelSize, int iterations) {
    cv::Mat resImg = img.clone();
    cv::Mat edgeFeat = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    resImg.convertTo(resImg, CV_32F);
    for (int iter = 0; iter < iterations; iter++) {
        cv::Mat tmpImg = resImg.clone();
        for (int row = 0; row < img.rows; row++)
            for (int col = 0; col < img.cols; col++) {
                cv::Vec4f corner = getCorner(tmpImg, row, col, kernelSize);
                cv::Vec4f border = getBorder(tmpImg, row, col, kernelSize);
                // Pick the Closest Value from Corner and Border
                if (iter == 0) {
                    float minDist = 1e9;
                    for (int idx = 0; idx < 4; idx++) {
                        float dist = std::abs(tmpImg.at<float>(row, col) - corner[idx]);
                        if (dist < minDist) minDist = dist, resImg.at<float>(row, col) = corner[idx], edgeFeat.at<float>(row, col) = idx;
                        dist = std::abs(tmpImg.at<float>(row, col) - border[idx]);
                        if (dist < minDist) minDist = dist, resImg.at<float>(row, col) = border[idx], edgeFeat.at<float>(row, col) = idx + 4;
                    }
                } else {
                    int idx = edgeFeat.at<float>(row, col);  // Use Edge Feature to Pick the value
                    if (idx < 4)
                        resImg.at<float>(row, col) = corner[idx];
                    else
                        resImg.at<float>(row, col) = border[idx - 4];
                }
            }
    }
    return resImg;
}

// Sub Window Bilateral Filter
cv::Mat subWBilateral(cv::Mat img, int kernelSize, int iterations, float sigmaS, float sigmaR) {
    cv::Mat resImg = img.clone();
    cv::Mat edgeFeat = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    resImg.convertTo(resImg, CV_32F);
    for (int iter = 0; iter < iterations; iter++) {
        cv::Mat tmpImg = resImg.clone();
        for (int row = 0; row < img.rows; row++)
            for (int col = 0; col < img.cols; col++) {
                cv::Vec4f corner = getCorner(tmpImg, row, col, kernelSize);
                cv::Vec4f border = getBorder(tmpImg, row, col, kernelSize);
                // Pick the Closest Value from Corner and Border
                if (iter == 0) {
                    float minDist = 1e9;
                    for (int idx = 0; idx < 4; idx++) {
                        float dist = std::abs(tmpImg.at<float>(row, col) - corner[idx]);
                        if (dist < minDist) minDist = dist, resImg.at<float>(row, col) = corner[idx], edgeFeat.at<float>(row, col) = idx;
                        dist = std::abs(tmpImg.at<float>(row, col) - border[idx]);
                        if (dist < minDist) minDist = dist, resImg.at<float>(row, col) = border[idx], edgeFeat.at<float>(row, col) = idx + 4;
                    }
                } else {
                    int idx = edgeFeat.at<float>(row, col);  // Use Edge Feature to Pick the value
                    if (idx < 4)
                        resImg.at<float>(row, col) = corner[idx];
                    else
                        resImg.at<float>(row, col) = border[idx - 4];
                }
            }
        resImg = bilateral(resImg, kernelSize, sigmaS, sigmaR);
    }
    return resImg;
}

// Canny Edge Detection
cv::Mat cannyEdge(cv::Mat img, float lowThr, float highThr, int domain) {
    cv::Mat resImg = img.clone() * 255 / domain, edgeImg = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    cv::Mat sobelX = cv::Mat::zeros(img.rows, img.cols, CV_16S), sobelY = cv::Mat::zeros(img.rows, img.cols, CV_16S);

    // Apply Sobel Filter
    resImg.convertTo(resImg, CV_8U);
    cv::Sobel(resImg, sobelX, CV_16S, 1, 0);
    cv::Sobel(resImg, sobelY, CV_16S, 0, 1);

    // Compute Gradient Magnitude and Direction
    cv::Mat gradMag = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    cv::Mat gradDir = cv::Mat::zeros(img.rows, img.cols, CV_32F);
    for (int row = 0; row < img.rows; row++)
        for (int col = 0; col < img.cols; col++) {
            float xVal = sobelX.at<short>(row, col), yVal = sobelY.at<short>(row, col);
            gradMag.at<float>(row, col) = std::sqrt(std::pow(xVal, 2) + std::pow(yVal, 2));
            gradDir.at<float>(row, col) = std::atan2(yVal, xVal) * 180 / M_PI;
        }

    // Non-Maximum Suppression
    for (int row = 1; row < img.rows - 1; row++)
        for (int col = 1; col < img.cols - 1; col++) {
            float dir = gradDir.at<float>(row, col);
            float mag = gradMag.at<float>(row, col);
            float val1 = 0, val2 = 0;
            if ((dir >= -22.5 && dir < 22.5) || (dir >= 157.5 && dir <= 180))
                val1 = gradMag.at<float>(row, col + 1), val2 = gradMag.at<float>(row, col - 1);
            else if (dir >= 22.5 && dir < 67.5)
                val1 = gradMag.at<float>(row - 1, col + 1), val2 = gradMag.at<float>(row + 1, col - 1);
            else if (dir >= 67.5 && dir < 112.5)
                val1 = gradMag.at<float>(row - 1, col), val2 = gradMag.at<float>(row + 1, col);
            else if (dir >= 112.5 && dir < 157.5)
                val1 = gradMag.at<float>(row - 1, col - 1), val2 = gradMag.at<float>(row + 1, col + 1);
            if (mag > val1 && mag > val2)
                edgeImg.at<float>(row, col) = 1;
        }

    // Hysteresis Thresholding
    for (int row = 1; row < img.rows - 1; row++)
        for (int col = 1; col < img.cols - 1; col++) {
            if (edgeImg.at<float>(row, col) > 0) {
                if (gradMag.at<float>(row, col) < lowThr) edgeImg.at<float>(row, col) = 0;
                if (gradMag.at<float>(row, col) > highThr) edgeImg.at<float>(row, col) = 1;
            }
        }
    return edgeImg;
}

// Neighbor Edge Detection(Method used in Macro Edge)
cv::Mat neighborEdge(cv::Mat img, float threshold) {
    cv::Mat edgeImg = cv::Mat::zeros(img.rows, img.cols, CV_8U);
    for (int row = 0; row < img.rows; row++)
        for (int col = 0; col < img.cols; col++) {
            if (row == 0 || row == img.rows - 1 || col == 0 || col == img.cols - 1) continue;
            std::vector<float> nbVals, nbDiffs;  // Neighbour Values and Differences
            nbVals.push_back(img.at<float>(row - 1, col)), nbVals.push_back(img.at<float>(row + 1, col));
            nbVals.push_back(img.at<float>(row, col - 1)), nbVals.push_back(img.at<float>(row, col + 1));
            for (int idx = 0; idx < 4; idx++) nbDiffs.push_back(std::pow(img.at<float>(row, col) - nbVals[idx], 2));
            for (int idx = 0; idx < 4; idx++) nbDiffs[idx] /= std::pow(img.at<float>(row, col) + nbVals[idx], 2);
            for (int idx = 0; idx < 4; idx++)  // Check if the difference is larger than threshold
                if (nbDiffs[idx] >= threshold) edgeImg.at<uchar>(row, col) = 255;
        }
    return edgeImg;
}

//============================== Internal Functions ==============================//

// Get Corner Value
cv::Vec4f getCorner(cv::Mat img, int row, int col, int kernelSize) {
    cv::Vec4f corner = cv::Vec4f(0, 0, 0, 0);
    float radius = kernelSize / 2;
    for (int rdx = -kernelSize / 2; rdx <= kernelSize / 2; rdx++)
        for (int cdx = -kernelSize / 2; cdx <= kernelSize / 2; cdx++) {
            int imgRow = row + rdx, imgCol = col + cdx;
            if (imgRow < 0 || imgRow >= img.rows || imgCol < 0 || imgCol >= img.cols) continue;
            if (rdx <= 0 && cdx <= 0) corner[0] += img.at<float>(imgRow, imgCol);  // Top Left
            if (rdx <= 0 && cdx >= 0) corner[1] += img.at<float>(imgRow, imgCol);  // Top Right
            if (rdx >= 0 && cdx <= 0) corner[2] += img.at<float>(imgRow, imgCol);  // Bottom Left
            if (rdx >= 0 && cdx >= 0) corner[3] += img.at<float>(imgRow, imgCol);  // Bottom Right
        }
    corner /= std::pow((1 + radius), 2);
    return corner;
}

// Get Border Value
cv::Vec4f getBorder(cv::Mat img, int row, int col, int kernelSize) {
    cv::Vec4f border = cv::Vec4f(0, 0, 0, 0);
    float radius = kernelSize / 2;
    // Get Border
    for (int rdx = -kernelSize / 2; rdx <= kernelSize / 2; rdx++)
        for (int cdx = -kernelSize / 2; cdx <= kernelSize / 2; cdx++) {
            int imgRow = row + rdx, imgCol = col + cdx;
            if (imgRow < 0 || imgRow >= img.rows || imgCol < 0 || imgCol >= img.cols) continue;
            if (rdx <= 0) border[0] += img.at<float>(imgRow, imgCol);  // Top
            if (rdx >= 0) border[2] += img.at<float>(imgRow, imgCol);  // Bottom
            if (cdx <= 0) border[1] += img.at<float>(imgRow, imgCol);  // Left
            if (cdx >= 0) border[3] += img.at<float>(imgRow, imgCol);  // Right
        }
    border /= (radius + 1) * (2 * radius + 1);
    return border;
}

// Get High Frequency Image
cv::Mat getHPF(cv::Mat oriImg, cv::Mat lpfImg, bool clipNeg) {
    // Convert to Float
    int typeOri = oriImg.type(), typeLpf = lpfImg.type();
    oriImg.convertTo(oriImg, CV_32F), lpfImg.convertTo(lpfImg, CV_32F);
    // High Frequency Image
    cv::Mat hFreq = cv::Mat::zeros(oriImg.rows, oriImg.cols, CV_32F);
    for (int row = 0; row < oriImg.rows; row++)
        for (int col = 0; col < oriImg.cols; col++) {
            float hVal = oriImg.at<float>(row, col) - lpfImg.at<float>(row, col);
            if (clipNeg) hVal = std::max(0.0f, hVal);
            hFreq.at<float>(row, col) = hVal;
        }
    return hFreq;
}

cv::Mat multiExpF(std::vector<cv::Mat> imgList, std::function<cv::Mat(cv::Mat)> func, std::string mode) {
    cv::Mat resImg = cv::Mat::zeros(imgList[0].rows, imgList[0].cols, CV_32F);
    for (int idx = 0; idx < imgList.size(); idx++) {
        cv::Mat expImg = func(imgList[idx]);
        expImg.convertTo(expImg, CV_32F);
        if (mode == "add") resImg += expImg;
        if (mode == "and")
            for (int row = 0; row < resImg.rows; row++)
                for (int col = 0; col < resImg.cols; col++)
                    resImg.at<float>(row, col) = std::min(resImg.at<float>(row, col), expImg.at<float>(row, col));
        if (mode == "or")
            for (int row = 0; row < resImg.rows; row++)
                for (int col = 0; col < resImg.cols; col++)
                    resImg.at<float>(row, col) = std::max(resImg.at<float>(row, col), expImg.at<float>(row, col));
    }
    return resImg;
}

}  // namespace filter

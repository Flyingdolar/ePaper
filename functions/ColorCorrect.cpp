#include "ColorCorrect.hpp"

namespace colorcorrect {
// Constructor: Set Image
void CCM_2D::initImg(const cv::Mat3f imgRGB, const cv::Mat3f gtRgb) {
    imgIn = imgRGB.clone(), imgCCM = imgRGB.clone(), imgGT = gtRgb.clone();
    varMat = cv::Mat::zeros(3, 2, CV_32F), ccmMat = cv::Mat::eye(3, 3, CV_32F);
    return;
}

// Constructor: Set Variable Matrix
void CCM_2D::initVar(const cv::Mat1f varVal) {
    varMat = varVal.clone(), updateCCM();
    return;
}

// Private Function: Update CCM
void CCM_2D::updateCCM() {
    ccmMat(0, 1) = varMat(0, 0), ccmMat(0, 2) = varMat(0, 1), ccmMat(0, 0) = 1.0f - varMat(0, 0) - varMat(0, 1);
    ccmMat(1, 0) = varMat(1, 0), ccmMat(1, 2) = varMat(1, 1), ccmMat(1, 1) = 1.0f - varMat(1, 0) - varMat(1, 1);
    ccmMat(2, 0) = varMat(2, 0), ccmMat(2, 1) = varMat(2, 1), ccmMat(2, 2) = 1.0f - varMat(2, 0) - varMat(2, 1);
    return;
}

/**
 * @brief Apply CCM to Image
 * @param img Input Image
 * @return cv::Mat3f Corrected Image with CCM
 */
cv::Mat3f CCM_2D::applyCCM(cv::Mat3f oriImg) {
    cv::Mat3f adjImg = oriImg.clone();
    for (int row = 0; row < oriImg.rows; row++)
        for (int col = 0; col < oriImg.cols; col++) {
            float oriR = oriImg(row, col)[0], oriG = oriImg(row, col)[1], oriB = oriImg(row, col)[2];
            adjImg(row, col)[0] = ccmMat(0, 0) * oriR + ccmMat(0, 1) * oriG + ccmMat(0, 2) * oriB;
            adjImg(row, col)[1] = ccmMat(1, 0) * oriR + ccmMat(1, 1) * oriG + ccmMat(1, 2) * oriB;
            adjImg(row, col)[2] = ccmMat(2, 0) * oriR + ccmMat(2, 1) * oriG + ccmMat(2, 2) * oriB;
        }
    return adjImg;
}

// Calculate Loss
double CCM_2D::operator()(const Eigen::VectorXd& x) {
    for (int row = 0; row < 3; row++)  // Set Variables
        for (int col = 0; col < 2; col++) varMat[row][col] = x(row * 2 + col);
    updateCCM();  // Update CCM
    return operator()();
}

// Calculate Loss
double CCM_2D::operator()() {
    int pixNum = imgIn.rows * imgIn.cols;
    double loss = 0.0;

    // Convert Image & Ground Truth to LAB Color Space
    imgCCM = applyCCM(imgIn);  // Apply CCM
    cv::Mat3f imgCCLab = colorconvert::cvtColor(colorconvert::cvtColor(imgCCM.clone(), colorconvert::RGB2XYZ), colorconvert::XYZ2Lab);
    cv::Mat3f gtLab = colorconvert::cvtColor(colorconvert::cvtColor(imgGT.clone(), colorconvert::RGB2XYZ), colorconvert::XYZ2Lab);

    // Calculate Loss
    for (int row = 0; row < imgCCM.rows; row++)
        for (int col = 0; col < imgCCM.cols; col++) {
            cv::Vec3f diff = imgCCLab(row, col) - gtLab(row, col);
            double pixLoss = 0.0;
            for (int ch = 0; ch < 3; ch++) pixLoss += std::pow(diff[ch] * 100, lossExp);
            loss += std::pow(pixLoss, 1.0 / lossExp);
        }
    return loss / pixNum;
}

/**
 * @brief Optimize CCM by PSO (Particle Swarm Optimization)
 * @param pcNum Particle Number
 * @param maxIter Maximum Iteration
 * @param minBnd Minimum Bound for Variables
 * @param maxBnd Maximum Bound for Variables
 * @param minPosDiff Stop Condition for Particle Position
 * @param minFuncDiff Stop Condition for Function Value
 * @param verbosity Verbosity (0: No Output, 1: Output Iteration, 2: Output Iteration & Function Value)
 */
void CCM_2D::optbyPSO(int pcNum, int maxIter, double minBnd, double maxBnd, double minPosDiff, double minFuncDiff, int verbosity) {
    pso::ParticleSwarmOptimization<double, CCM_2D> optimizer;

    // Set PSO Parameters
    optimizer.setObjective(*this);                // Objective Function
    optimizer.setMaxIterations(maxIter);          // Maximum Iteration
    optimizer.setMinParticleChange(minPosDiff);   // Stop Condition for Particle Position
    optimizer.setMinFunctionChange(minFuncDiff);  // Stop Condition for Function Value
    optimizer.setVerbosity(verbosity);            // 0: No Output, 1: Output Iteration, 2: Output Iteration & Function Value

    // PSO Optimization
    Eigen::MatrixXd bounds(2, 6);  // Set Bounds for Variables
    Eigen::VectorXd initX = Eigen::VectorXd::Zero(6);
    bounds.row(0).setConstant(minBnd), bounds.row(1).setConstant(maxBnd);
    auto result = optimizer.minimize(bounds, pcNum, initX);

    // Save Result
    for (int row = 0; row < 3; row++)
        for (int col = 0; col < 2; col++) varMat[row][col] = result.xval(row * 2 + col);
    updateCCM(), imgCCM = applyCCM(imgIn);
    return;
}

/**
 * @brief Optimize CCM by Brute Force
 * @param iter Iteration
 * @param minBnd Minimum Bound for Variables
 * @param maxBnd Maximum Bound for Variables
 * @param step Step Size
 * @param verbosity Verbosity (0: No Output, 1: Output Iteration, 2: Output Iteration & Function Value)
 * @return double Best Loss
 * @note This function is not recommended for high-dimensional optimization.
 */
void CCM_2D::optbyBF(int iters, double minBnd, double maxBnd, double step, int verbosity) {
    double bestLoss = std::numeric_limits<double>::max();
    Eigen::VectorXd bestVar = Eigen::VectorXd::Zero(6);

    Eigen::MatrixXd bounds(2, 6);  // Set Bounds for Variables
    bounds.row(0).setConstant(minBnd), bounds.row(1).setConstant(maxBnd);

    for (int iter = 0; iter < iters; iter++) {
        if (iter != 0) {
            for (int row = 0; row < 3; row++)
                for (int col = 0; col < 2; col++) {  // Update Bounds by Best Result
                    bounds(0, row * 2 + col) = bestVar(row * 2 + col) - step;
                    bounds(1, row * 2 + col) = bestVar(row * 2 + col) + step;
                }
            step /= 5.0;  // Decrease Step Size
        }
        int workNum = std::pow((bounds(1, 0) - bounds(0, 0)) / step + 1, 6), workCnt = 0;
        // Brute Force Optimization
        for (double var1 = bounds(0, 0); var1 <= bounds(1, 0); var1 += step)
            for (double var2 = bounds(0, 1); var2 <= bounds(1, 1); var2 += step)
                for (double var3 = bounds(0, 2); var3 <= bounds(1, 2); var3 += step)
                    for (double var4 = bounds(0, 3); var4 <= bounds(1, 3); var4 += step)
                        for (double var5 = bounds(0, 4); var5 <= bounds(1, 4); var5 += step)
                            for (double var6 = bounds(0, 5); var6 <= bounds(1, 5); var6 += step) {
                                Eigen::VectorXd x(6);
                                x << var1, var2, var3, var4, var5, var6;
                                double loss = operator()(x);
                                if (loss < bestLoss) bestLoss = loss, bestVar = x;
                                workCnt++;
                                // Verbosity
                                if (verbosity >= 2) saveData::showProgress("Brute Force <Iter " + std::to_string(iter) + ">", workCnt / (float)workNum, "Best Loss: " + std::to_string(bestLoss));
                            }
        // Verbosity
        if (verbosity >= 1) std::cout << "Brute Force <Iter " << iter << "> Best Loss: " << bestLoss << " ,Best Variables: " << bestVar.transpose() << std::endl;
    }
    // Save Result
    for (int row = 0; row < 3; row++)
        for (int col = 0; col < 2; col++) varMat[row][col] = bestVar(row * 2 + col);
    updateCCM(), imgCCM = applyCCM(imgIn);
    return;
}
}  // namespace colorcorrect

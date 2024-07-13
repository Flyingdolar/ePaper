#pragma once

#include "Functions.hpp"

namespace colorcorrect {
class CCM_2D {
   private:
    cv::Mat3f imgIn, imgCCM, imgGT;      // Image Input, Image with CCM, Ground Truth Image
    cv::Mat1f ccmMat = cv::Mat1f(3, 3);  // CCM Matrix
    cv::Mat1f varMat = cv::Mat1f(3, 2);  // Variable Matrix
    int lossExp = 2;                     // Loss Exponent

    void updateCCM();  // Update CCM

   public:
    // Constructor
    void initImg(const cv::Mat3f imgRGB, const cv::Mat3f gtRgb);  // Set Image & Ground Truth
    void initVar(const cv::Mat1f varVal);                         // Set Matrix Directly
    // Operator
    double operator()(const Eigen::VectorXd& x);  // Calculate Loss
    double operator()();                          // Calculate Loss
    // Optimize
    void optbyPSO(  // Optimize CCM by PSO (Particle Swarm Optimization)
        int pcNum, int maxIter, double minBnd, double maxBnd,
        double minPosDiff, double minFuncDiff, int verbosity);
    void optbyBF(  // Optimize CCM by Brute Force
        int iter, double minBnd, double maxBnd, double step, int verbosity);
    cv::Mat3f applyCCM(cv::Mat3f img);           // Apply CCM to Image
    cv::Mat1f getCCM() const { return ccmMat; }  // Get CCM Matrix
};

}  // namespace colorcorrect

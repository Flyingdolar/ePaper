#include "Functions.hpp"

void printMat(cv::Mat1f mat, std::string name) {
    std::cout << name << ": " << std::endl;
    for (int row = 0; row < mat.rows; row++) {
        for (int col = 0; col < mat.cols; col++) {
            std::cout << std::fixed << std::setprecision(2) << mat.at<float>(row, col) << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    // Create Test Random 0, 1 Matrix
    cv::Mat testMat = cv::Mat::zeros(8, 8, CV_32F);
    for (int row = 0; row < testMat.rows; row++)
        for (int col = 0; col < testMat.cols; col++) testMat.at<float>(row, col) = (rand() % 2 == 0) ? 0 : 1;

    // Print Test Matrix
    printMat(testMat, "Test Matrix");

    // Do Void & Clustering
    cv::Mat VCMat = halftone::voidCluster(testMat, cv::Vec2i(4, 4), 3, 1.0f);
    VCMat.convertTo(VCMat, CV_32F);

    // Print Void & Clustering Matrix
    printMat(VCMat, "Void & Clustering Matrix");

    return 0;
}

#include "SaveData.hpp"

namespace saveData {

// Define global variables
std::string defFolder = "", defFile = "";
int defCounter = 0;
bool isDefFolder = false, isDefFile = false, isDefCounter = false;
cv::Mat nanMap;

// Find NaN values in the image data, return the count of NaN values
// If there are NaN values, return the cv::Mat nanMap with NaN values marked as 255, other 0
int checkNAN(cv::Mat img) {
    int height = img.rows, width = img.cols, channel = img.channels();
    int nanCount = 0;

    nanMap = cv::Mat::zeros(height, width, CV_8U);
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++)
            if (channel > 1) {
                for (int ch = 0; ch < channel; ch++)
                    if (std::isnan(img.at<cv::Vec3f>(row, col)[ch])) {
                        nanMap.at<uchar>(row, col) = 255;
                        nanCount++;
                    } else if (std::isnan(img.at<float>(row, col))) {
                        nanMap.at<uchar>(row, col) = 255;
                        nanCount++;
                    }
            } else {
                if (std::isnan(img.at<float>(row, col))) {
                    nanMap.at<uchar>(row, col) = 255;
                    nanCount++;
                }
            }
    return nanCount;
}

void imgMat(cv::Mat img, std::string folderName, std::string fileName, int domain, bool useCounter) {
    cv::Mat saveImg = img.clone();
    int height = saveImg.rows, width = saveImg.cols, channel = saveImg.channels();

    // Setup save path
    folderName = folderName.empty() ? "." : folderName;
    std::string savePath = useCounter ? folderName + "/" + std::to_string(defCounter) + "_" + fileName : folderName + "/" + fileName;

    // Check for NaN values
    int nanCount = checkNAN(saveImg);
    if (nanCount > 0) {
        std::cerr << "Error [ " << fileName << " ]: Found " << nanCount << " NaN values." << std::endl;
        cv::imwrite(savePath + "_nanMap.png", nanMap);
    }

    saveImg *= 65535 / domain;
    for (int row = 0; row < height; row++)
        for (int col = 0; col < width; col++)
            if (channel > 1)
                for (int ch = 0; ch < channel; ch++)
                    saveImg.at<cv::Vec3f>(row, col)[ch] = std::clamp(saveImg.at<cv::Vec3f>(row, col)[ch], 0.0f, 65535.0f);
            else
                saveImg.at<float>(row, col) = std::clamp(saveImg.at<float>(row, col), 0.0f, 65535.0f);
    saveImg.convertTo(saveImg, CV_16U);
    cv::imwrite(savePath + ".png", saveImg);

    if (useCounter) defCounter++;
    return;
}

cJSON* readLog(std::string folderName, std::string fileName) {
    std::string filePath = folderName.empty() ? "." : folderName + "/" + fileName + ".json";
    std::ifstream ifs(filePath);
    cJSON* jsData = nullptr;

    // File exists and is not empty, read it
    if (ifs.good() && ifs.peek() != std::ifstream::traits_type::eof()) {
        std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        jsData = cJSON_Parse(content.c_str());
        if (!jsData) {  // Error parsing JSON
            std::cerr << "Error parsing JSON file." << std::endl;
            return nullptr;
        }
        return jsData;  // Success
    }

    // File does not exist or is empty, create a new JSON object
    jsData = cJSON_CreateObject();
    if (!jsData) {
        std::cerr << "Could not create JSON object." << std::endl;
        return nullptr;
    }
    return jsData;  // Success
}

void showProgress(std::string title, float progress, std::string desc) {
    // Calculate & Allocate the progress bar width
    struct winsize wSize;  // Get the console width (LINUX) ...for Windows, use GetConsoleScreenBufferInfo
    ioctl(STDOUT_FILENO, TIOCGWINSZ, &wSize);
    int consoleWidth = wSize.ws_col;
    int textLen = title.length() + desc.length() + 20;
    int barWidth = consoleWidth - textLen, progPos = barWidth * progress;
    float progRate = progress * 100.0;

    // Print the progress bar
    // Example: "[ Title |=====>        | 50.0% | Description ]"
    std::cout << "[ " << title << " |";
    for (int idx = 0; idx < barWidth; idx++)
        if (idx < progPos)
            std::cout << "=";  // Progress Bar
        else if (idx == progPos)
            std::cout << ">";  // Progress Indicator
        else
            std::cout << " ";  // Remaining Space
    std::cout << "| ";
    if (progRate < 10.0) std::cout << " ";
    if (progRate < 100.0) std::cout << " ";
    std::cout << std::fixed << std::setprecision(1) << progRate << "% | " << desc << " ]";
    if (progRate >= 100.0)
        std::cout << std::endl;
    else
        std::cout << "\r" << std::flush;  // Return to the beginning of the line
    return;
}

}  // namespace saveData

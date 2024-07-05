#pragma once

#ifndef SAVEDATA_HPP
#define SAVEDATA_HPP

#include <cjson/cJSON.h>

#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace saveData {

// Define global variables
extern std::string defFolder, defFile;  // Default folder and file name
extern int defCounter;                  // Default counter for image file name
extern bool isDefFolder, isDefFile;     // Default flag for folder and file name
extern bool isDefCounter;               // Default flag for counter

/**
 * @brief Set default values for saving image data.
 * @details You can set it customly also by setting the global variables.
 * @param folder Default folder to save the image. {defFolder = folder, isDefFolder = true}
 * @param file Default name of the image file. {defFile = file, isDefFile = true}
 * @param counter Default counter for the image file name. {defCounter = counter, isDefCounter = true}
 * @note If you set ""(empty string) or -1 for the counter, it will not be used and set Flag to false.
 */
inline void initVar(std::string folder = "", std::string file = "", int counter = -1) {
    // Set Flags
    isDefFolder = !folder.empty(), isDefFile = !file.empty();
    isDefCounter = counter >= 0;
    // Set Default Values
    defFolder = folder, defFile = file, defCounter = counter;
    // Create the Folder if not exists
    if (isDefFolder)
        if (system(("mkdir -p " + defFolder).c_str()) == -1) return;
    return;
}

/**
 * @brief Save image data to a file.
 *
 * @param img Input image data.
 * @param folderName Folder to save the image. (Empty for current directory)
 * @param fileName Name of the image file.
 * @param domain Domain of the image data.  (Default: 1)
 * @param useDefCounter Use defCounter for the image file name. (If not set, use global flag: isDefCounter)
 */
void imgMat(cv::Mat img, std::string folderName, std::string fileName, int domain = 1, bool useDefCounter = isDefCounter);
/**
 * @brief Save image data to a file.
 *
 * @param img Input image data.
 * @param fileName Name of the image file.
 * @param domain Domain of the image data.  (Default: 1)
 * @param useDefCounter Use defCounter for the image file prefix number. (If not set, use global flag: isDefCounter)
 * @param useDefFolder Use defFolder for the image folder path. (If not set, use global flag: isDefFolder)
 */
inline void imgMat(cv::Mat img, std::string fileName, int domain = 1, bool useDefCounter = isDefCounter, bool useDefFolder = isDefFolder) {
    if (useDefFolder)  // Mode 1: Save with Set Folder
        imgMat(img, defFolder, fileName, domain, useDefCounter);
    else  // Mode 2: Save at current directory
        imgMat(img, "", fileName, domain, useDefCounter);
    return;
}

/**
 * @brief Read Log Data from a file.
 * @details Read the cJSON data from a file in JSON format.
 * @param folderName Folder to read the log data. (Empty for using default folder)
 * @param fileName Name of the log file. (Empty for using default file name)
 * @return cJSON* cJSON data read from the file.
 */
cJSON* readLog(std::string folderName, std::string fileName);
inline cJSON* readLog(std::string fileName) { return readLog(defFolder, fileName); }

// =========================== CJSON Object Functions =========================== /
/**
 * @brief Create a cJSON object with a key-value pair.
 * @param key Key of the cJSON object.
 * @param value Value of the cJSON object.
 * @return cJSON* cJSON object with the key-value pair.
 */
template <typename valType>
inline cJSON* newObj(valType value) { return nullptr; }
template <>  // Specialization for std::string
inline cJSON* newObj(std::string value) { return cJSON_CreateString(value.c_str()); }
template <>  // Specialization for int
inline cJSON* newObj(int value) { return cJSON_CreateNumber(value); }
template <>  // Specialization for float
inline cJSON* newObj(float value) { return cJSON_CreateNumber(value); }
template <>  // Specialization for double
inline cJSON* newObj(double value) { return cJSON_CreateNumber(value); }
template <>  // Specialization for std::vector<int>
inline cJSON* newObj(std::vector<int> value) {
    cJSON* arr = cJSON_CreateArray();
    for (int idx = 0; idx < value.size(); idx++) cJSON_AddItemToArray(arr, cJSON_CreateNumber(value[idx]));
    return arr;
}
template <>  // Specialization for std::vector<float>
inline cJSON* newObj(std::vector<float> value) {
    cJSON* arr = cJSON_CreateArray();
    for (int idx = 0; idx < value.size(); idx++) cJSON_AddItemToArray(arr, cJSON_CreateNumber(value[idx]));
    return arr;
}
template <>  // Specialization for std::vector<double>
inline cJSON* newObj(std::vector<double> value) {
    cJSON* arr = cJSON_CreateArray();
    for (int idx = 0; idx < value.size(); idx++) cJSON_AddItemToArray(arr, cJSON_CreateNumber(value[idx]));
    return arr;
}
template <>  // Specialization for cv::Vec3f
inline cJSON* newObj(cv::Vec3f value) {
    cJSON* vec = cJSON_CreateArray();
    for (int ch = 0; ch < 3; ch++) cJSON_AddItemToArray(vec, cJSON_CreateNumber(value[ch]));
    return vec;
}
template <>  // Specialization for std::vector<cv::Vec3f>
inline cJSON* newObj(std::vector<cv::Vec3f> value) {
    cJSON* arr = cJSON_CreateArray();
    for (int idx = 0; idx < value.size(); idx++) cJSON_AddItemToArray(arr, newObj(value[idx]));
    return arr;
}
template <>  // Specialization for cv::Mat<float>
inline cJSON* newObj(cv::Mat1f value) {
    cJSON* mat = cJSON_CreateArray();
    for (int row = 0; row < value.rows; row++) {
        cJSON* rowArr = cJSON_CreateArray();
        for (int col = 0; col < value.cols; col++) {
            cv::Vec3f pix = value.at<cv::Vec3f>(row, col);
            cJSON* pixArr = newObj(pix);
            cJSON_AddItemToArray(rowArr, pixArr);
        }
        cJSON_AddItemToArray(mat, rowArr);
    }
    return mat;
}
template <>  // Specialization for cv::Mat<cv::Vec3f>
inline cJSON* newObj(cv::Mat3f value) {
    cJSON* mat = cJSON_CreateArray();
    for (int row = 0; row < value.rows; row++) {
        cJSON* rowArr = cJSON_CreateArray();
        for (int col = 0; col < value.cols; col++) {
            cv::Vec3f pix = value.at<cv::Vec3f>(row, col);
            cJSON* pixArr = newObj(pix);
            cJSON_AddItemToArray(rowArr, pixArr);
        }
        cJSON_AddItemToArray(mat, rowArr);
    }
    return mat;
}
// ================================================================================= /

/**
 * @brief Save Log Data to a file.
 * @details Save the cJSON data to a file in JSON format.
 * @param folderName Folder to save the log data. (Empty for using default folder)
 * @param fileName Name of the log file. (Empty for using default file name)
 * @param data cJSON data to save.
 */
template <typename valType>
void logData(std::string folderName, std::string fileName, std::string key, valType value) {
    const char* keyStr = key.c_str();
    std::string filePath = folderName.empty() ? "." : folderName + "/" + fileName + ".json";
    cJSON* fileData = readLog(folderName, fileName);

    // Add data as a new object to the fileData
    cJSON* oldData = cJSON_GetObjectItem(fileData, keyStr);
    cJSON* newData = newObj(value);
    // Replace the old data with the new data
    if (oldData) cJSON_DeleteItemFromObject(fileData, keyStr);
    cJSON_AddItemToObject(fileData, keyStr, newData);

    // Write the fileData to the file
    std::ofstream ofs(filePath);
    ofs << cJSON_Print(fileData);
    ofs.close(), cJSON_Delete(fileData);
    return;
}
template <typename valType>
inline void logData(std::string fileName, std::string key, valType value) { logData(defFolder, fileName, key, value); }
template <typename valType>
inline void logData(std::string key, valType value) { logData(defFolder, defFile, key, value); }

}  // namespace saveData
#endif  // SAVEDATA_HPP

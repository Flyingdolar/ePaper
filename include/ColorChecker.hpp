#pragma once

#include "Functions.hpp"

namespace colorchecker {

#define BLOCKS std::vector<std::pair<std::pair<int, int>, std::pair<int, int>>>
#define COLORS std::vector<cv::Vec3f>

// ColorChecker 24 blocks RGB value
const COLORS checker24 = {
    {115, 82, 68},    // 1: Dark Skin
    {194, 150, 130},  // 2: Light Skin
    {98, 122, 157},   // 3: Blue Sky
    {87, 108, 67},    // 4: Foliage
    {133, 128, 177},  // 5: Blue Flower
    {103, 189, 170},  // 6: Bluish Green
    {214, 126, 44},   // 7: Orange
    {80, 91, 166},    // 8: Purplish Blue
    {193, 90, 99},    // 9: Moderate Red
    {94, 60, 108},    // 10: Purple
    {157, 188, 64},   // 11: Yellow Green
    {224, 163, 46},   // 12: Orange Yellow
    {56, 61, 150},    // 13: Blue
    {70, 148, 73},    // 14: Green
    {175, 54, 60},    // 15: Red
    {231, 199, 31},   // 16: Yellow
    {187, 86, 149},   // 17: Magenta
    {8, 133, 161},    // 18: Cyan
    {243, 243, 242},  // 19: White
    {200, 200, 200},  // 20: Neutral 8
    {160, 160, 160},  // 21: Neutral 6.5
    {122, 122, 121},  // 22: Neutral 5
    {85, 85, 85},     // 23: Neutral 3.5
    {52, 52, 52},     // 24: Black
};

// ColorChecker 32 blocks RGB value
const COLORS checker32 = {
    {115, 82, 68},    // 1: Dark Skin
    {194, 150, 130},  // 2: Light Skin
    {98, 122, 157},   // 3: Blue Sky
    {87, 108, 67},    // 4: Foliage
    {133, 128, 177},  // 5: Blue Flower
    {103, 189, 170},  // 6: Bluish Green
    {214, 126, 44},   // 7: Orange
    {80, 91, 166},    // 8: Purplish Blue
    {193, 90, 99},    // 9: Moderate Red
    {94, 60, 108},    // 10: Purple
    {157, 188, 64},   // 11: Yellow Green
    {224, 163, 46},   // 12: Orange Yellow
    {56, 61, 150},    // 13: Blue
    {70, 148, 73},    // 14: Green
    {175, 54, 60},    // 15: Red
    {231, 199, 31},   // 16: Yellow
    {187, 86, 149},   // 17: Magenta
    {8, 133, 161},    // 18: Cyan
    {0, 0, 255},      // 19: Blue (Max)
    {0, 255, 0},      // 20: Green (Max)
    {255, 0, 0},      // 21: Red (Max)
    {255, 255, 0},    // 22: Yellow (Max)
    {255, 0, 255},    // 23: Magenta (Max)
    {0, 255, 255},    // 24: Cyan (Max)
    {255, 255, 255},  // 25: White (Max)
    {243, 243, 242},  // 26: White
    {200, 200, 200},  // 27: Neutral 8
    {160, 160, 160},  // 28: Neutral 6.5
    {122, 122, 121},  // 29: Neutral 5
    {85, 85, 85},     // 30: Neutral 3.5
    {52, 52, 52},     // 31: Black
    {0, 0, 0},        // 32: Black (Max)
};

// ColorChecker 24 blocks position in image
const BLOCKS colorPos = {
    {{594, 971}, {654, 1031}},  // 1: Dark Skin
    {{594, 887}, {654, 947}},   // 2: Light Skin
    {{594, 803}, {654, 863}},   // 3: Blue Sky
    {{594, 719}, {654, 779}},   // 4: Foliage
    {{594, 635}, {654, 695}},   // 5: Blue Flower
    {{594, 551}, {654, 611}},   // 6: Bluish Green

    {{677, 971}, {737, 1031}},  // 7: Orange
    {{677, 887}, {737, 947}},   // 8: Purplish Blue
    {{677, 803}, {737, 863}},   // 9: Moderate Red
    {{677, 719}, {737, 779}},   // 10: Purple
    {{677, 635}, {737, 695}},   // 11: Yellow Green
    {{677, 551}, {737, 611}},   // 12: Orange Yellow

    {{760, 971}, {820, 1031}},  // 13: Blue
    {{760, 887}, {820, 947}},   // 14: Green
    {{760, 803}, {820, 863}},   // 15: Red
    {{760, 719}, {820, 779}},   // 16: Yellow
    {{760, 635}, {820, 695}},   // 17: Magenta
    {{760, 551}, {820, 611}},   // 18: Cyan

    {{843, 971}, {903, 1031}},  // 19: White
    {{843, 887}, {903, 947}},   // 20: Neutral 8
    {{843, 803}, {903, 863}},   // 21: Neutral 6.5
    {{843, 719}, {903, 779}},   // 22: Neutral 5
    {{843, 635}, {903, 695}},   // 23: Neutral 3.5
    {{843, 551}, {903, 611}},   // 24: Black
};

// GrayCard position in image
const BLOCKS grayPos = {
    {{320, 625}, {684, 1208}},  // 1: GrayCard
};

}  // namespace colorchecker

//
// Created by chengli on 19-4-23.
//

#include "detect.h"


cv::Mat BaseDetect::binary_thresh_mat(cv::Mat &src, const int (*color_dict)[3][2], int color_index) {
    cv::MatConstIterator_<cv::Vec3b> it_in = src.begin<cv::Vec3b>();
    cv::MatConstIterator_<cv::Vec3b> itend_in = src.end<cv::Vec3b>();
    cv::Mat binary_mat(src.size[0], src.size[1], CV_8UC1);
    cv::MatIterator_<uchar> it_bg_out = binary_mat.begin<uchar>();
    while (it_in != itend_in) {
        if (this->in_color_range(*it_in, color_dict[color_index])) {
            (*it_bg_out) = 255;
        } else {
            (*it_bg_out) = 0;
        }
        it_in++;
        it_bg_out++;
    }
    return binary_mat;
}

bool BaseDetect::in_color_range(cv::Vec3b bgr, const int (*color_range)[2]) {
    return color_range[2][1] >= bgr[2] && color_range[2][0] <= bgr[2] &&
           color_range[1][1] >= bgr[1] && color_range[1][0] <= bgr[1] &&
           color_range[0][1] >= bgr[0] && color_range[0][0] <= bgr[0];
}

float BaseDetect::similar_percent(cv::Mat &src, cv::Mat &pattern) {
    cv::MatConstIterator_<uchar> it_in = src.begin<uchar>();
    cv::MatConstIterator_<uchar> itend_in = src.end<uchar>();
    cv::MatIterator_<uchar> it_pattern_in = pattern.begin<uchar>();
    if (src.size[0] == pattern.size[0] && src.size[1] == pattern.size[1]) {
        int all_value = 0;
        int same_value = 0;
        while (it_in != itend_in) {
            if ((*it_pattern_in) >= 190 && (*it_in) >= 190) {
                same_value++;
            }
            if (*it_in >= 190 || *it_pattern_in >= 190) {
                all_value++;
            }
            it_in++;
            it_pattern_in++;
        }
        return same_value * 1.0f / (all_value + 0.001f);
    } else {
        return 0;
    }
}

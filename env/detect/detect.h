//
// Created by chengli on 19-4-23.
//

#ifndef PY_C_TEST_BASE_AGENT_H
#define PY_C_TEST_BASE_AGENT_H

#include <iostream>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

using std::string;
using std::vector;
using std::to_string;
using std::cout;
using std::endl;
using std::sort;
using cv::Mat;
using cv::Rect;
using cv::Point;
using cv::RotatedRect;

class BaseDetect {
private:
    bool in_color_range(cv::Vec3b bgr, const int (*color_range)[2]);

protected:

    bool debug = true;

//    cv::Mat binary_thresh_mat(cv::Mat &src, const int (*color_dict)[3][2], int color_index);

    float similar_percent(cv::Mat &src, cv::Mat &pattern);

public:
    cv::Mat diff_mat(cv::Mat &src, cv::Mat &pattern, int thresh=50);

    static cv::Mat noise_filter(cv::Mat &src);


    cv::Mat binary_thresh_mat(cv::Mat &src, const int (*color_dict)[3][2], int color_index);
    static cv::Mat exract_attention(cv::Mat &src, cv::Mat &mask);
};


#endif //PY_C_TEST_BASE_AGENT_H

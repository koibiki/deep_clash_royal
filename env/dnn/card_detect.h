//
// Created by holaverse on 19-4-29.
//

#ifndef CLASH_ROYAL_AGENT_CARD_DETECT_H
#define CLASH_ROYAL_AGENT_CARD_DETECT_H

#include <iostream>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"

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
using cv::Size;

struct CardResult {
    int card_type = 0;
    int available = 0;
    float prob = 0.0;
};

class CardDetect {

private:
    cv::dnn::Net net;

    int pre_type[4];

    bool debug = false;

    int argmax(Mat &src);


public:

    CardDetect();

    void reset();

    bool load_model(string mode_path);

    CardResult predict(Mat &src);

};


#endif //CLASH_ROYAL_AGENT_CARD_DETECT_H

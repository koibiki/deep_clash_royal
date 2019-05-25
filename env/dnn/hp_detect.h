//
// Created by chengli on 2019/5/24.
//

#ifndef ENV_HP_DETECT_H
#define ENV_HP_DETECT_H


#include "../detect/detect.h"
#include "opencv2/dnn.hpp"

struct HpResult {
    float opp_result[3] = {0, 0, 0};
    float mine_result[3] = {0, 0, 0};
};

class HpDetect : protected BaseDetect {

private:
    int COLOR_DICT[2][3][2] = {
            // red
            {{170, 230}, {170, 230}, {200, 255}},
            // blue
            {{200, 255}, {170, 235}, {140, 220}},
    };

    // opp_throne opp_left opp_right
    Rect opp_hp_rects[3] = {{277, 14,  50, 15},
                            {108, 132, 40, 15},
                            {392, 132, 40, 15}};
    Rect mine_hp_rects[3] = {{277, 694, 50, 15},
                             {108, 589, 40, 15},
                             {392, 589, 40, 15}};

    cv::dnn::Net net;

    bool debug = false;

    // 默认均为9级国王塔
    float opp_pre_hp[3] = {4008.f, 2534.f, 2534.f};
    float mine_pre_hp[3] = {4008.f, 2534.f, 2534.f};

    Mat *opp_pre_mats;
    Mat *mine_pre_mats;

    Rect has_num(Mat &src);

    int predict(Mat &src);

    int argmax(Mat &src);

    bool load_model(string mode_path);

    Mat process_img(Mat &src, int color_index);

public:

    HpDetect();

    ~HpDetect();

    HpResult detect_hp(Mat &src);

    void reset();

    void detect_one_size(Mat &src, Rect rects[3], float *pre_hp, int color_index, float *result);
};


#endif //ENV_HP_DETECT_H

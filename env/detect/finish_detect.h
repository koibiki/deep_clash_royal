//
// Created by chengli on 19-4-23.
//

#ifndef PY_C_TEST_BLUE_BUTTON_DETECT_H
#define PY_C_TEST_BLUE_BUTTON_DETECT_H


#include "detect.h"

struct FinishResult {
    bool is_finish = false;
    int battle_result = -1;
};

class FinishDetect : protected BaseDetect {

private:
    int COLOR_DICT[5][3][2] = {
            // blue button upon finish page
            {{220, 255}, {150, 200}, {50,  110}},
            // white text color upon finish page when battle_result
            {{240, 255}, {240, 255}, {240, 255}},
            // grey text color upon finish page when full chest
            {{200, 230}, {200, 230}, {200, 230}},

            {{240, 255}, {230, 255}, {80,  255}},
            {{230, 255}, {190, 230}, {200, 255}}
    };

    Mat process_img(Mat &src, int color_index);

    bool has_finish_button(Mat &src);

    bool is_win_finish(Mat &src);

    bool has_winner(Mat &src, int color_index);


public:
    FinishDetect();

    FinishResult detect_finish(cv::Mat &src, int frame_index);

};

#endif //PY_C_TEST_BLUE_BUTTON_DETECT_H

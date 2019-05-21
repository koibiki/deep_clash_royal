//
// Created by holaverse on 19-4-23.
//

#ifndef PY_C_TEST_EXCUTE_H
#define PY_C_TEST_EXCUTE_H

#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include "detect/running_detect.h"
#include "detect/menu_detect.h"
#include "detect/finish_detect.h"
#include "detect/start_detect.h"
#include "detect/button_detect.h"
#include "dnn/elixir_detect.h"
#include "dnn/card_detect.h"


static const int MENU_STATE = 0;

static const int RUNNING_STATE = 1;

static const int FINISH_STATE = 2;

static const int ERROR_STATE = 3;

typedef struct PyMat {
    uchar *frame_data;
    int height;
    int width;
    int channel;
} py_mat;

typedef struct Result {
    int game_state = MENU_STATE;
    int frame_state = ERROR_STATE;
    int index = -1;
    bool is_grey = false;
    int purple_loc[2] = {0, 0};
    int yellow_loc[2] = {0, 0};
    int opp_crown = 0;
    int mine_crown = 0;
    int card_type[4] = {0, 0, 0, 0};
    int available[4] = {0, 0, 0, 0};
    float prob[4] = {0.0, 0.0, 0.0, 0.0};
    int battle_result = -1;
    int frame_index = 0;
    int time;
    int remain_elixir = 5;
    float milli;
} result;

/**
 * 仅用于检测 960 x 540 的图片
 * */

class ClashRoyalAgent {
private:
    RunningDetect runningDetect;

    FinishDetect finishDetect;

    ElixirDetect elixirDetect;

    CardDetect cardDetect;

    Menu menu;

    ButtonDetect buttonDetect;

    int currentGameState = MENU_STATE;

    int gameId;
    int frame_index = -5;

    int frame_w = 540;
    int frame_h = 960;
    int start_w = int(0.23 * frame_w);
    int w_gap = int(0.022 * frame_w);

    int clip_w = frame_w / 6;

    int start_h = frame_h * 5 / 6;
    int clip_h = frame_h * 8 / 85;

    Rect rects[4] = {Rect(start_w, start_h, clip_w, clip_h),
                     Rect(start_w + clip_w + w_gap, start_h, clip_w, clip_h),
                     Rect(start_w + 2 * clip_w + 2 * w_gap, start_h, clip_w, clip_h),
                     Rect(start_w + 3 * clip_w + 3 * w_gap, start_h, clip_w, clip_h)};

    int pre_type[4];

    int start_time;

    int finish_count = 0;

    bool debug = false;

    cv::Mat transfer_mat(py_mat mat);

public:

    ClashRoyalAgent();

    void init_agent(int gameId);

    result detect_frame(py_mat mat, result r);

    result detect_mat(Mat mat, result r);
};


#endif //PY_C_TEST_EXCUTE_H

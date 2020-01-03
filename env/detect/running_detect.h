//
// Created by chengli on 19-4-25.
//

#ifndef CLASH_ROYAL_AGENT_RUNNING_DETECT_H
#define CLASH_ROYAL_AGENT_RUNNING_DETECT_H


#include "detect.h"

struct RunningResult {
    bool isRunning = false;
    int opp_crown = 0;
    int mine_crown = 0;
    double milli = 0.0;
};

class RunningDetect : public BaseDetect {
private:
    int COLOR_DICT[3][3][2] = {
            {{35,  85},  {35,  85},  {195, 255}},
            {{225, 255}, {175, 215}, {85,  115}},
            {{50, 115}, {50, 115}, {220, 255}},
    };

    Rect opp_pattern_rect = {15, 55, 24, 30};
    Rect mine_pattern_rect = {15, 90, 24, 30};

    string opp_patterns_root = "./crown_num_pattern/opp/";
    string mine_patterns_root = "./crown_num_pattern/mine/";

    int pre_opp_crown = 0;
    int pre_mine_crown = 0;
    bool debug = true;

    Mat process_img(Mat &src, int color_index);

    Rect get_crown_num_rect(Mat &src, bool is_opp);

    int get_most_possible_num(Mat &src, bool is_opp);

public:

    RunningDetect();

    RunningResult detect_running(Mat &src , int frame_index);

    bool has_split_line(Mat &src);
};


#endif //CLASH_ROYAL_AGENT_RUNNING_DETECT_H

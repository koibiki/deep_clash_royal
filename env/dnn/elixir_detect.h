//
// Created by holaverse on 19-4-28.
//

#ifndef CLASH_ROYAL_AGENT_ELIXIR_DETECT_H
#define CLASH_ROYAL_AGENT_ELIXIR_DETECT_H


#include "../detect/detect.h"
#include "opencv2/dnn.hpp"

class ElixirDetect : protected BaseDetect {
private:
    int COLOR_DICT[1][3][2] = {
            {{240, 255}, {240, 255}, {240, 255}},
    };

    Rect num_rect = {141, 912, 40, 27};

    cv::dnn::Net net;

    bool debug = false;

    int pre_elixir = 0;

    int predict(Mat &src);

    int argmax(Mat &src);

    bool load_model(string mode_path);

public:

    ElixirDetect();

    int detect_elixir(Mat &src);

    void reset();

};


#endif //CLASH_ROYAL_AGENT_ELIXIR_DETECT_H

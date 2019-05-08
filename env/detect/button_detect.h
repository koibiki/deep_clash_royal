//
// Created by holaverse on 19-5-9.
//

#ifndef ENV_BUTTON_DETECT_H
#define ENV_BUTTON_DETECT_H

#include "detect.h"


struct ButtonResult {
    int purple_button_loc[2] = {0, 0};
    int yellow_button_loc[2] = {0, 0};
};

class ButtonDetect : protected BaseDetect {

private:
    int COLOR_DICT[2][3][2] = {
            {{240, 255}, {95,  135}, {170, 210}},
            {{40,  100}, {165, 210}, {230, 255}},
    };

    Rect check_button(Mat &src, Rect &button_rect, int color_index);

    Mat process_img(Mat &src, int color_index);

public:

    ButtonDetect();

    ButtonResult detect_button(Mat &src, int frame_index);

};


#endif //ENV_BUTTON_DETECT_H

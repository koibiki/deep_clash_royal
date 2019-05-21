//
// Created by holaverse on 19-3-26.
//

#ifndef SLIDE_TAP_MENU_H
#define SLIDE_TAP_MENU_H

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <map>
#include <string>

using namespace cv;
using namespace std;


struct MenuResult {
    bool inGame = false;
    bool isGrey = false;
    int index = -1;
    Rect rect;
    double milli = 0.0;
};

class Menu {
public:
    Menu();

    MenuResult detect_tap_location(Mat &mat, int frame_index);

private:
    bool debug = false;

    float length_thresh = 0.05f;

    cv::Mat clip_mat(Mat &src);

    bool is_clock_color(Vec3b bgr, const int thresh[3][2]);

    void thresh_img(Mat &src, Mat &bg_out, int index);

    bool is_in_game(vector<Rect> &rects, int width, int color_index);

    int get_tap_index(vector<Rect> &rects, int width, bool isGrey);

    void detect_tap_by_color(Mat &src, vector<Rect> &rects, int width, int height, int color_index);

    int get_rect_height(Mat &src, int index, int color_index);

};

#endif //SLIDE_TAP_MENU_H

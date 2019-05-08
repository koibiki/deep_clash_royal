//
// Created by holaverse on 19-5-9.
//

#include "button_detect.h"

ButtonDetect::ButtonDetect() {
    cout << "init ButtonDetect." << endl;
}

ButtonResult ButtonDetect::detect_button(Mat &src, int frame_index) {

    Mat resize;
    cv::resize(src, resize, cv::Size(), 0.3, 0.3);

    int width = resize.size[1];
    int height = resize.size[0];

    float scale = 1080 * 1.0f / width;

    Rect purple_rec(width / 3, height * 8 / 10, width / 6, height / 10);
    Rect yellow_rec(width * 2 / 3, height / 10, width / 6, height * 8 / 10);

    ButtonResult result{};

    const Rect &purple_button = check_button(resize, purple_rec, 0);
    const Rect &yellow_button = check_button(resize, yellow_rec, 1);
    if (!purple_button.empty()) {
        result.purple_button_loc[0] = (purple_button.x + purple_button.width / 2 + width / 3) * scale;
        result.purple_button_loc[1] = (purple_button.y + purple_button.height / 2 + height * 8 / 10) * scale;
    }
    if (!yellow_button.empty()) {
        result.yellow_button_loc[0] = (yellow_button.x + yellow_button.width / 2 + width * 2 / 3) * scale;
        result.yellow_button_loc[1] = (yellow_button.y + yellow_button.height / 2 + height / 10) * scale;
    }
    return result;
}

Mat ButtonDetect::process_img(Mat &src, int color_index) {
    Mat binary_mat = this->binary_thresh_mat(src, COLOR_DICT, color_index);
    Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(binary_mat, binary_mat, element);
    return binary_mat;
}


Rect ButtonDetect::check_button(Mat &src, Rect &button_rect, int color_index) {

    Mat button_mat = src(button_rect);

    const Mat &binary = this->process_img(button_mat, color_index);

    int width = button_mat.cols;
    int height = button_mat.rows;

    vector<Rect> possible_rects;

    vector<vector<Point> > contours;

    findContours(binary, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contours.size(); i++) {
        vector<Point> points = contours[i];
        RotatedRect rect = minAreaRect(points);

        Rect possible_rect = rect.boundingRect();
        if (possible_rect.x * 1.0f / width < 0.2 &&
            (possible_rect.x + possible_rect.width) * 1.0f / width > 0.97) {
            if (color_index == 0) {
                if (possible_rect.y * 1.0f / height < 0.5 &&
                    (possible_rect.y + possible_rect.height) * 1.0f / height > 0.7) {
                    possible_rects.push_back(possible_rect);
                }
            } else {
                possible_rects.push_back(possible_rect);
            }
        }
    }

    sort(possible_rects.begin(), possible_rects.end(),
         [](Rect r1, Rect r2) { return r1.width * r1.height > r2.height * r2.width; });

    if (possible_rects.empty()) {
        return Rect(0, 0, 0, 0);
    } else {
        return possible_rects[0];
    }
}





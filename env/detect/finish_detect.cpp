//
// Created by chengli on 19-4-23.
//

#include "finish_detect.h"


FinishResult FinishDetect::detect_finish(cv::Mat &src, int frame_index) {
    cout << "c++ detect finish page:" << frame_index << endl;
    Mat resize_src;
    cv::resize(src, resize_src, cv::Size(540 / 2, 960 / 2));
    bool finishPage = has_finish_button(resize_src);


    FinishResult result = FinishResult();
    result.is_finish = finishPage;
    if (finishPage) {

        int w = resize_src.size[1];
        int h = resize_src.size[0];
        Rect mine(w / 3, int(0.38 * h), w / 3, int(0.04 * h));
        Rect opp(w / 3, int(0.09 * h), w / 3, int(0.04 * h));

        Mat mine_mat = resize_src(mine);
        Mat opp_mat = resize_src(opp);

        bool win = has_winner(mine_mat, 3);
        bool lose = has_winner(opp_mat, 4);

        if (win && !lose) {
            result.battle_win = true;
        } else if (!win && lose) {
            result.battle_win = false;
        } else {
            result.battle_win = is_win_finish(resize_src);
        }

    }
    return result;
}


Mat FinishDetect::process_img(Mat &src, int color_index) {
    Mat binary_mat = this->binary_thresh_mat(src, COLOR_DICT, color_index);
    Mat element0 = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(binary_mat, binary_mat, element0);
    return binary_mat;
}

bool FinishDetect::has_finish_button(Mat &src) {

    int width = src.size[1];
    int height = src.size[0];

    int start_w = width / 3;

    int clip_w = width / 3;
    int start_h = height * 4 / 5;
    int clip_h = height * 2 / 15;

    Rect clip_rect(start_w, start_h, clip_w, clip_h);

    Mat clip_mat = src(clip_rect);

    Mat binary_mat = process_img(clip_mat, 0);

    vector <Rect> possible_rects;

    vector <vector<Point>> contours;

    findContours(binary_mat, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contours.size(); i++) {
        vector <Point> points = contours[i];
        RotatedRect rect = minAreaRect(points);

        Rect possible_rect = rect.boundingRect();
        if (possible_rect.x * 1.0f / clip_w > 0.1 &&
            (possible_rect.x + possible_rect.width) * 1.0f / clip_w < 0.9 &&
            possible_rect.width * 1.0f / clip_w > 0.6 &&
            possible_rect.y * 1.0f / clip_h < 0.5 &&
            (possible_rect.y + possible_rect.height) * 1.0f / clip_h < 0.9 &&
            possible_rect.height * 1.0f / clip_h > 0.3) {

            possible_rects.push_back(possible_rect);

        }
    }
    return possible_rects.size() == 1;

}

bool FinishDetect::has_winner(Mat &src, int color_index) {

    Mat resize;

    cv::resize(src, resize, cv::Size(), 0.3, 0.3);

    int w = src.size[1];
    int h = src.size[0];


    Mat binary = process_img(src, color_index);

    vector <Rect> possible_rects;

    vector <vector<Point>> contours;

    findContours(binary, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contours.size(); i++) {
        vector <Point> points = contours[i];
        RotatedRect rect = minAreaRect(points);

        Rect possible_rect = rect.boundingRect();
        if (possible_rect.x * 1.0f / w < 0.3 &&
            (possible_rect.x + possible_rect.width) * 1.0f / w > 0.7 &&
            possible_rect.width * 1.0f / w > 0.6 &&
            possible_rect.y * 1.0f / h < 0.5 &&
            (possible_rect.y + possible_rect.height) * 1.0f / h > 0.75 &&
            possible_rect.height * 1.0f / h > 0.4) {

            possible_rects.push_back(possible_rect);

        }
    }

    return possible_rects.size() == 1;
}

bool FinishDetect::is_win_finish(Mat &src) {

    int width = src.size[1];
    int height = src.size[0];

    int start_w = width / 5;

    int clip_w = width * 3 / 5;
    int start_h = height * 2 / 3;
    int clip_h = height / 6;

    Rect chest_rect(start_w, start_h, clip_w / 2, clip_h);
    Rect golden_rect(start_w + clip_w / 2, start_h, clip_w / 2, clip_h);

    Mat chest_mat = src(chest_rect);
    Mat golden_mat = src(golden_rect);

    Mat golden_binary = process_img(golden_mat, 1);

    vector <Rect> possible_rects;

    vector <vector<Point>> contours;

    findContours(golden_binary, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contours.size(); i++) {
        vector <Point> points = contours[i];
        RotatedRect rect = minAreaRect(points);

        Rect possible_rect = rect.boundingRect();
        if (possible_rect.x * 2.0f / clip_w > 0.25 &&
            (possible_rect.x + possible_rect.width) * 2.0f / clip_w < 0.75 &&
            possible_rect.width * 2.0f / clip_w > 0.25 &&
            possible_rect.y * 1.0f / clip_h > 0.5 &&
            (possible_rect.y + possible_rect.height) * 1.0f / clip_h > 0.75 &&
            possible_rect.height * 1.0f / clip_h < 0.4) {

            possible_rects.push_back(possible_rect);

        }
    }

    return possible_rects.size() == 1;
}

FinishDetect::FinishDetect() {
    cout << "init FinishDetect." << endl;
}


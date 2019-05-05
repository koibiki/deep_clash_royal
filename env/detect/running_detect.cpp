//
// Created by chengli on 19-4-25.
//

#include "running_detect.h"


RunningDetect::RunningDetect() {
    cout << "init RunningDetect" << endl;
}

RunningResult RunningDetect::detect_running(Mat &src, int frame_index) {
    cout << "c++ detect running:" << frame_index << endl;
    int width = src.size[1];
    int height = src.size[0];

    int start_w = width * 9 / 10;

    int clip_w = width / 10;
    int start_h = height / 4;
    int clip_h = height / 3;


    char * ss = "ss";



    Rect opp_clip_rect(start_w, start_h, clip_w, clip_h / 2);
    Rect mine_clip_rect(start_w, start_h + clip_h / 2, clip_w, clip_h / 2);

    Mat opp_clip_mat = src(opp_clip_rect);
    Mat mine_clip_mat = src(mine_clip_rect);
    Mat opp_mat = process_img(opp_clip_mat, 0);
    Mat mine_mat = process_img(mine_clip_mat, 1);

    Rect opp_rect = get_crown_num_rect(opp_mat, true);
    Rect mine_rect = get_crown_num_rect(mine_mat, false);
    RunningResult result = RunningResult();

//    cv::imshow("opp", opp_mat);
//    cv::imshow("min", mine_mat);
//    cv::imshow("op", opp_mat(opp_pattern_rect));
//    cv::imshow("mi", mine_mat(mine_pattern_rect));
//    cv::waitKey(1);

//    cv::imwrite("../crown_num/opp_" + to_string(frame_index) + ".jpg", opp_mat(opp_pattern_rect));
//    cv::imwrite("../crown_num/mine_" + to_string(frame_index) + ".jpg", mine_mat(mine_pattern_rect));

    if (opp_rect.width > 0 && mine_rect.width > 0 && opp_rect.height > 0 && mine_rect.height > 0) {
        Mat opp_num = opp_mat(opp_pattern_rect);
        Mat mine_num = mine_mat(mine_pattern_rect);
        int opp_crown = get_most_possible_num(opp_num, true);
        int mine_crown = get_most_possible_num(mine_num, false);
        result.isRunning = true;
        result.opp_crown = opp_crown == -1 ? pre_opp_crown : opp_crown;
        result.mine_crown = mine_crown == -1 ? pre_mine_crown : mine_crown;

    }
    return result;
}

Mat RunningDetect::process_img(Mat &src, int color_index) {
    Mat binary_mat = this->binary_thresh_mat(src, COLOR_DICT, color_index);
    Mat element0 = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(binary_mat, binary_mat, element0);
    Mat element1 = getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::erode(binary_mat, binary_mat, element1);
    return binary_mat;
}

Rect RunningDetect::get_crown_num_rect(Mat &src, bool is_opp) {

    int width = src.cols;
    int height = src.rows;

    vector<Rect> possible_rects;

    vector<vector<Point> > contours;

    findContours(src, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contours.size(); i++) {
        vector<Point> points = contours[i];
        RotatedRect rect = minAreaRect(points);

        Rect possible_rect = rect.boundingRect();
        if (possible_rect.x * 1.0f / width > 0.2 &&
            (possible_rect.x + possible_rect.width) * 1.0f / width < 0.8 &&
            std::abs((possible_rect.x + possible_rect.width * 1.0f / 2) - width * 1.0f / 2) / width < 0.2 &&
            possible_rect.height * 1.0f / height > 0.1) {

            if (is_opp) {
                if (possible_rect.y * 1.0f / height > 0.25 &&
                    (possible_rect.y + possible_rect.height) * 1.0f / height < 0.8) {
                    possible_rects.push_back(possible_rect);
                }
            } else {
                if (possible_rect.y * 1.0f / height > 0.3) {
                    possible_rects.push_back(possible_rect);
                }
            }
        }
    }

    sort(possible_rects.begin(), possible_rects.end(),
         [](Rect r1, Rect r2) { return r1.width * r1.height > r2.height * r2.width; });

    if (possible_rects.empty()) {
        return {0, 0, 0, 0};
    } else {
        return possible_rects[0];
    }
}

int RunningDetect::get_most_possible_num(Mat &src, bool is_opp) {
    float max_similar_value = -1;
    int possible_num = -1;

    for (int i = 0; i < 4; i++) {
        string pattern_path = (is_opp ? opp_patterns_root : mine_patterns_root) + to_string(i) + ".jpg";
        Mat pattern = cv::imread(pattern_path, cv::IMREAD_GRAYSCALE);

        float similar_value = similar_percent(src, pattern);
        if (max_similar_value < similar_value) {
            max_similar_value = similar_value;
            possible_num = i < 2 ? i : 2;
        }
    }
    //cout << "value:" << possible_num << "     similar value:" << max_similar_value << endl;
    if (max_similar_value >= 0.75) {
        return possible_num;
    } else {
        return -1;
    }
}




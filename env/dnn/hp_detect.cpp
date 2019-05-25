//
// Created by chengli on 2019/5/24.
//

#include "hp_detect.h"

HpDetect::HpDetect() {
    cout << "init HpDetect." << endl;
    this->opp_pre_mats = new Mat[3];
    this->mine_pre_mats = new Mat[3];
}

void HpDetect::reset() {
    opp_pre_hp[0] = 4008.f;
    opp_pre_hp[1] = 2534.f;
    opp_pre_hp[2] = 2534.f;
    mine_pre_hp[0] = 4008.f;
    mine_pre_hp[1] = 2534.f;
    mine_pre_hp[2] = 2534.f;
    opp_pre_mats[0].data = nullptr;
    opp_pre_mats[1].data = nullptr;
    opp_pre_mats[2].data = nullptr;
    mine_pre_mats[0].data = nullptr;
    mine_pre_mats[1].data = nullptr;
    mine_pre_mats[2].data = nullptr;

}

HpResult HpDetect::detect_hp(Mat &src) {
    auto *opp_result = new float[3];
    auto *mine_result = new float[3];

    HpResult hpResult;
    detect_one_size(src, mine_hp_rects, mine_pre_hp, 1, mine_result);
    detect_one_size(src, opp_hp_rects, opp_pre_hp, 0, opp_result);

    hpResult.opp_result[0] = opp_result[0];
    hpResult.opp_result[1] = opp_result[1];
    hpResult.opp_result[2] = opp_result[2];

    hpResult.mine_result[0] = mine_result[0];
    hpResult.mine_result[1] = mine_result[1];
    hpResult.mine_result[2] = mine_result[2];

    delete[] opp_result;
    delete[] mine_result;
    return hpResult;
}

bool HpDetect::load_model(string mode_path) {
    return false;
}

Mat HpDetect::process_img(Mat &src, int color_index) {
    Mat binary_mat = this->binary_thresh_mat(src, COLOR_DICT, color_index);
    if (debug) {
        cv::imshow("oo", binary_mat);
    }
    Mat element0 = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(binary_mat, binary_mat, element0);
    return binary_mat;
}

Rect HpDetect::has_num(Mat &src) {
    int w = src.size[1];
    int h = src.size[0];

    vector<Rect> possible_rects;

    vector<vector<Point>> contours;

    findContours(src, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contours.size(); i++) {
        vector<Point> points = contours[i];
        RotatedRect rect = minAreaRect(points);

        Rect possible_rect = rect.boundingRect();
        if (possible_rect.x * 1.0f / w < 0.15 &&
            (possible_rect.x + possible_rect.width) * 1.0f / w > 0.3 &&
            possible_rect.width * 1.0f / w > 0.3 &&
            possible_rect.y * 1.0f / h < 0.15 &&
            (possible_rect.y + possible_rect.height) * 1.0f / h > 0.9 &&
            possible_rect.height * 1.0f / h > 0.85) {
            possible_rects.push_back(possible_rect);
        }
    }

    sort(possible_rects.begin(), possible_rects.end(),
         [](Rect r1, Rect r2) { return r1.width * r1.height > r2.height * r2.width; });

    if (possible_rects.empty()) {
        return {};
    } else {
        return {0, 0, possible_rects[0].x + possible_rects[0].width + 5, possible_rects[0].height};
    }
}

int HpDetect::predict(Mat &src) {
    return 0;
}

int HpDetect::argmax(Mat &src) {
    return 0;
}

void HpDetect::detect_one_size(Mat &src, Rect *rects, float *pre_hp, int color_index, float *result) {
    for (int i = 0; i < 3; i++) {

        Mat mat = src(rects[i]);

        Mat binary_mat = this->binary_thresh_mat(mat, COLOR_DICT, color_index);
        if (debug) {
            cv::imshow("oo", binary_mat);
        }
        Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        Mat dilate;
        cv::dilate(binary_mat, dilate, element);

        if (debug) {
            cv::imshow("ss", mat);
            cv::waitKey(1);
        }
        const Rect &rect = has_num(dilate);
        if (!rect.empty()) {
            Mat pre;
            if (color_index == 0) {
                pre = opp_pre_mats[i];
            } else {
                pre = mine_pre_mats[i];
            }
            if (pre.empty()) {
                result[i] = 1.f;
            } else {
                float percent = similar_percent(pre, binary_mat);
                if (percent < 0.9) {
                    result[i] = 1.f;
                } else {
                    result[i] = 0.f;
                }
            }
            if (color_index == 0) {
                opp_pre_mats[i].data = nullptr;
                opp_pre_mats[i] = binary_mat;
            } else {
                mine_pre_mats[i].data = nullptr;
                mine_pre_mats[i] = binary_mat;
            }
//                int num = predict(mat);
//                if (num < pre_hp[i]) {
//                    result[i] = pre_hp[i] - num;
//                    pre_hp[i] = num;
//                } else {
//                    result[i] = 0;
//                }
        } else {
//                if (pre_hp[i] < 50) {
//                    pre_hp[i] = 0;
//                }
            result[i] = 0.f;
        }

    }
}

HpDetect::~HpDetect() {
    delete[]opp_pre_mats;
    delete[]mine_pre_mats;
}


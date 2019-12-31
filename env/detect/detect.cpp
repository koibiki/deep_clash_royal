//
// Created by chengli on 19-4-23.
//

#include "detect.h"


cv::Mat BaseDetect::binary_thresh_mat(cv::Mat &src, const int (*color_dict)[3][2], int color_index) {
    cv::MatConstIterator_<cv::Vec3b> it_in = src.begin<cv::Vec3b>();
    cv::MatConstIterator_<cv::Vec3b> itend_in = src.end<cv::Vec3b>();
    cv::Mat binary_mat(src.size[0], src.size[1], CV_8UC1);
    cv::MatIterator_<uchar> it_bg_out = binary_mat.begin<uchar>();
    while (it_in != itend_in) {
        if (this->in_color_range(*it_in, color_dict[color_index])) {
            (*it_bg_out) = 255;
        } else {
            (*it_bg_out) = 0;
        }
        it_in++;
        it_bg_out++;
    }
    return binary_mat;
}


cv::Mat BaseDetect::diff_mat(cv::Mat &src, cv::Mat &pattern, int thresh) {
    cv::Mat binary_mat(src.size[0], src.size[1], CV_8UC1, cv::Scalar(0));
    int nr = src.rows;
    int nl = src.cols;
    int channel = src.channels();
    for (int k = 0; k < nr; k++) {
        const uchar *srcData = src.ptr<uchar>(k);
        auto *patternData = pattern.ptr<uchar>(k);
        auto *binaryData = binary_mat.ptr<uchar>(k);

        for (int i = 0; i < nl; i++) {
            int start_index = i * channel;
            int diff_value = abs(srcData[start_index] - patternData[start_index])
                             + abs(srcData[start_index + 1] - patternData[start_index + 1]);
            //+ abs(srcData[start_index + 2] - patternData[start_index + 2]);
//            cout<<" diff_value"<< diff_value <<endl;
            if (diff_value > thresh) {
                binaryData[i] = 255;
            }
        }
    }
    return binary_mat;
}


bool BaseDetect::in_color_range(cv::Vec3b bgr, const int (*color_range)[2]) {
    return color_range[2][1] >= bgr[2] && color_range[2][0] <= bgr[2] &&
           color_range[1][1] >= bgr[1] && color_range[1][0] <= bgr[1] &&
           color_range[0][1] >= bgr[0] && color_range[0][0] <= bgr[0];
}

float BaseDetect::similar_percent(cv::Mat &src, cv::Mat &pattern) {
    cv::MatConstIterator_<uchar> it_in = src.begin<uchar>();
    cv::MatConstIterator_<uchar> itend_in = src.end<uchar>();
    cv::MatIterator_<uchar> it_pattern_in = pattern.begin<uchar>();
    if (src.size[0] == pattern.size[0] && src.size[1] == pattern.size[1]) {
        int all_value = 0;
        int same_value = 0;
        while (it_in != itend_in) {
            if ((*it_pattern_in) >= 190 && (*it_in) >= 190) {
                same_value++;
            }
            if (*it_in >= 190 || *it_pattern_in >= 190) {
                all_value++;
            }
            it_in++;
            it_pattern_in++;
        }
        return same_value * 1.0f / (all_value + 0.001f);
    } else {
        return 0;
    }
}

cv::Mat BaseDetect::noise_filter(cv::Mat &src) {
    vector<vector<Point>> contours;
    findContours(src, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
    cv::Mat binary_mat(src.size[0], src.size[1], CV_8UC1, cv::Scalar(0));
    int i = 0;
    for (i = 0; i < contours.size(); i++) {
        vector<Point> points = contours[i];
        double area = cv::contourArea(points);
        if (area > 20) {
            drawContours(binary_mat, contours, (int) i, cv::Scalar(255), -1);
        }
        cout << "area:" << area << endl;
    }

    Mat element0 = getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::dilate(binary_mat, binary_mat, element0);
    return binary_mat;
}

cv::Mat BaseDetect::exract_attention(cv::Mat &src, cv::Mat &mask) {
    mask = noise_filter(mask);
    Mat dst;
    src.copyTo(dst, mask);
    return dst;
}

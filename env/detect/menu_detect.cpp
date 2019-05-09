//
// Created by holaverse on 19-3-26.
//

#include "menu_detect.h"


static int CHECK_NUM = 0;
static int RECT_HEIGHT = -1;

static const int BACK_THRESH[3][3][2]{
        // normal
        {{135, 185}, {100, 145}, {75,  100}},
        {{65,  135}, {165, 240}, {190, 255}},
        {{40,  80},  {40,  70},  {25,  55}}
};


struct MenuResult Menu::detect_tap_location(Mat &mat, int frame_index) {

    cout << "c++ detect menu:" << frame_index << endl;

    cv::TickMeter tm;
    tm.start();
    MenuResult result;
    if (mat.empty())
        return {};

    int width = mat.size[1];
    int height = mat.size[0];

    cv::Mat clipMat = this->clip_mat(mat);

    Mat scale_mat;
    resize(clipMat, scale_mat, Size(0, 0), 0.5, 0.5);
    int scale_width = scale_mat.size[1];
    int scale_height = scale_mat.size[0];

    vector<Rect> possible_rects;

    bool inGame = false;
    int colorIndex = 0;
    for (int i = 0; i < 3; i++) {
        colorIndex = i;
        detect_tap_by_color(scale_mat, possible_rects, scale_width, scale_height, i);
        inGame = is_in_game(possible_rects, scale_width, i);
        if (inGame) {
            break;
        }
    }
    tm.stop();

    result.inGame = inGame;
    if (inGame) {
        result.isGrey = colorIndex >= 2;
        result.index = get_tap_index(possible_rects, scale_width, result.isGrey);
        if (result.index != -1) {
            if (result.isGrey) {
                if (RECT_HEIGHT != -1) {
                    result.rect = Rect(width * result.index / 6, height - RECT_HEIGHT, width / 3, RECT_HEIGHT);
                }
            } else {
                int rect_height = get_rect_height(mat, result.index, colorIndex);
                if (rect_height != -1) {
                    RECT_HEIGHT = rect_height;
                    result.rect = Rect(width * result.index / 6, height - rect_height, width / 3, rect_height);
                } else {
                    result.index = -1;
                }
            }
        }
    } else {
        CHECK_NUM = 0;
        RECT_HEIGHT = -1;
    }
    result.milli = tm.getTimeMilli();

    if (debug) {
        cv::waitKey(0);
    }

    return result;
}


cv::Mat Menu::clip_mat(Mat &src) {
    int width = src.size[1];
    int height = src.size[0];

    int start_h = height * 29 / 30;

    Rect rect(0, start_h, width, height - start_h - 5);

    Mat mat;
    mat = src(rect);
    return mat;
}

bool Menu::is_clock_color(Vec3b bgr, const int (*thresh)[2]) {
    return thresh[2][1] >= bgr[2] && thresh[2][0] <= bgr[2] &&
           thresh[1][1] >= bgr[1] && thresh[1][0] <= bgr[1] &&
           thresh[0][1] >= bgr[0] && thresh[0][0] <= bgr[0];
}

void Menu::thresh_img(Mat &src, Mat &bg_out, int index) {
    MatConstIterator_<Vec3b> it_in = src.begin<Vec3b>();
    MatConstIterator_<Vec3b> itend_in = src.end<Vec3b>();
    MatIterator_<uchar> it_bg_out = bg_out.begin<uchar>();
    while (it_in != itend_in) {
        if (index == 1) {
            if (this->is_clock_color(*it_in, BACK_THRESH[index]) ||
                this->is_clock_color(*it_in, BACK_THRESH[0])) {
                (*it_bg_out) = 255;
            } else {
                (*it_bg_out) = 0;
            }
        } else {
            if (this->is_clock_color(*it_in, BACK_THRESH[index])) {
                (*it_bg_out) = 255;
            } else {
                (*it_bg_out) = 0;
            }
        }
        it_in++;
        it_bg_out++;
    }
}

bool Menu::is_in_game(vector<Rect> &rects, int width, int color_index) {
    if (!rects.empty()) {
        if (debug) {
            cout << "rect size:" << rects.size() << "  x:" << rects[0].x << "  w:" << rects[0].width << endl;
        }

        float rect_width_percent = rects[0].width * 3.0f / width;

        if (rects[0].x * 1.0f / width <= 0.15 || (rects[0].x + rects[0].width) * 1.0f / width >= 0.95) {
            if (debug) {
                cout << "is near edge:" << rects[0].x * 1.0f / width << endl;
            }
            if (color_index == 1) {
                return rect_width_percent >= 0.45 && rect_width_percent <= 2;
            } else {
                return rect_width_percent >= 0.45 && rect_width_percent <= 1.2;
            }
        } else {
            if (color_index == 1) {
                if (debug) {
                    cout << "rect_width_percent:" << rect_width_percent << endl;
                }
                return rect_width_percent >= 0.92 && rect_width_percent <= 2;
            } else {
                return rect_width_percent >= 0.92 && rect_width_percent <= 1.2;
            }
        }
    }
    return false;
}

int Menu::get_tap_index(vector<Rect> &rects, int width, bool isGrey) {
    if (rects.size() == 1) {
        int start = rects[0].x;
        int end = rects[0].x + rects[0].width;
        float start_percent = start * 1.0f / width;
        float end_percent = end * 1.0f / width;


        if (isGrey && (rects[0].width > width * 1.0f / 6) && abs(end_percent - 4.0 / 6) <= length_thresh) {
            return 2;
        }

        if (abs(start_percent - 0) <= length_thresh && abs(end_percent - 2.0 / 6) <= length_thresh) {
            return 0;
        } else if (abs(start_percent - 1.0f / 6) <= length_thresh && abs(end_percent - 3.0 / 6) <= length_thresh) {
            return 1;
        } else if (abs(start_percent - 2.0f / 6) <= length_thresh && abs(end_percent - 4.0 / 6) <= length_thresh) {
            return 2;
        } else if (abs(start_percent - 3.0f / 6) <= length_thresh && abs(end_percent - 5.0 / 6) <= length_thresh) {
            return 3;
        } else if (abs(start_percent - 4.0f / 6) <= length_thresh && abs(end_percent - 1) <= length_thresh) {
            return 4;
        }
    }
    return -1;
}

void Menu::detect_tap_by_color(Mat &mat, vector<Rect> &possible_rects, int width, int height, int color_index) {
    possible_rects.clear();
    Mat bg_thresh_img(height, width, CV_8UC1);

    this->thresh_img(mat, bg_thresh_img, color_index);

    Mat element = getStructuringElement(MORPH_RECT, Size(8, 8));
    cv::dilate(bg_thresh_img, bg_thresh_img, element);

    vector<vector<Point> > contours; // Vector for storing contours

    findContours(bg_thresh_img, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

    if (debug) {
        imshow("thresh" + to_string(color_index), bg_thresh_img);
    }


    for (int i = 0; i < contours.size(); i++) {
        vector<Point> points = contours[i];
        RotatedRect rect = minAreaRect(points);

        Rect possible_rect = rect.boundingRect();

        if (possible_rect.x * 1.0f / width <= 0.04 || (possible_rect.x + possible_rect.width) * 1.0f / width >= 0.96) {
            if (possible_rect.
                    width >= width * 5 / 30
                && possible_rect.height >= height * 7 / 8
                && possible_rect.width <= width * 20 / 30) {
                possible_rects.push_back(possible_rect);
            }
        } else {
            if (possible_rect.width >= width * 9 / 30
                && possible_rect.height >= height * 7 / 8) {
                if (color_index == 1) {
                    if (possible_rect.width <= width * 17 / 30) {
                        possible_rects.push_back(possible_rect);
                    }
                } else {
                    if (possible_rect.width <= width * 13 / 30) {
                        possible_rects.push_back(possible_rect);
                    }
                }
            }

        }

    }


    if (possible_rects.size() == 2) {
        possible_rects.clear();
    }

    if (debug) {
        Scalar color;
        color[0] = 0;
        color[1] = 0;
        color[2] = 255;

        for (int i = 0; i < possible_rects.size(); i++) {
            rectangle(mat, possible_rects[i], color, 2);
        }
        imshow("poss rect " + to_string(color_index), mat);
    }
}

int Menu::get_rect_height(Mat &src, int index, int color_index) {
    int rect_height = -1;
    if (RECT_HEIGHT == -1) {
        int width = src.size[1];
        int height = src.size[0];

        int start_h = height * 25 / 30;

        Rect rect(width * index / 6 + 5, start_h, width / 6 / 4, height - start_h);

        Mat out = src(rect);

        Mat thresh_img(out.size[0], out.size[1], CV_8UC1);

        this->thresh_img(out, thresh_img, color_index);


        vector<vector<Point> > contours; // Vector for storing contours

        findContours(thresh_img, contours, RETR_CCOMP, CHAIN_APPROX_SIMPLE);

        if (debug) {
            imshow("thresh rect", thresh_img);
        }

        vector<Rect> rects;

        for (int i = 0; i < contours.size(); i++) {
            vector<Point> points = contours[i];
            RotatedRect rect = minAreaRect(points);

            Rect possible_rect = rect.boundingRect();

            rects.push_back(possible_rect);
        }

        if (!rects.empty()) {
            sort(rects.begin(), rects.end(), [](Rect r1, Rect r2) {
                return r1.width * r1.height > r2.width * r2.height;
            });

            rect_height = rects[0].height;
        }
    } else {
        rect_height = RECT_HEIGHT;
    }
    return rect_height;

}

Menu::Menu() {
    cout << "init Menu." << endl;

}


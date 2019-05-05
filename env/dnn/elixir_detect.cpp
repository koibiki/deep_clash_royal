//
// Created by holaverse on 19-4-28.
//

#include "elixir_detect.h"

ElixirDetect::ElixirDetect() {
    cout << "init ElixirDetect." << endl;
    this->load_model("./asset/num_net.pb");
}

bool ElixirDetect::load_model(string mode_path) {
    this->net = cv::dnn::readNetFromTensorflow(mode_path);
    if (this->net.empty()) {
        cout << "Elixir net load fail." << endl;
    } else {
        cout << "Elixir net load success." << endl;
    }
}


int ElixirDetect::detect_elixir(Mat &src) {
    Mat clip_mat = src(num_rect);

    int width = clip_mat.cols;
    int height = clip_mat.rows;

    Mat binary_mat = this->binary_thresh_mat(clip_mat, COLOR_DICT, 0);


    Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(1, 1));
    cv::dilate(binary_mat, binary_mat, element);

    vector<Rect> possible_rects;

    vector<vector<Point> > contours;

    findContours(binary_mat, contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    for (int i = 0; i < contours.size(); i++) {
        vector<Point> points = contours[i];
        RotatedRect rect = minAreaRect(points);

        float d = rect.size.area() / (width * height);
        if (d > 0.05) {
            Rect possible_rect = rect.boundingRect();
            if (possible_rect.height * 1.0f / height > 0.4 &&
                possible_rect.y * 1.0f / height < 0.3 &&
                (possible_rect.x + possible_rect.width) * 1.0f / width < 0.9 &&
                (possible_rect.y + possible_rect.height) * 1.0f / height > 0.75) {
                possible_rects.push_back(possible_rect);
            }
        }
    }

//    cv::imshow("s", clip_mat);
//    cv::imshow("ss", binary_mat);
//    cv::waitKey(0);

    if (possible_rects.empty()) {
        cout << " 没有找到圣水位置" << endl;
        return this->pre_elixir;
    } else {
        int i = predict(clip_mat);
        return i;
    }

}

int ElixirDetect::predict(Mat &src) {
    Mat inputBlob = cv::dnn::blobFromImage(src, 1. / 255, cv::Size(32, 32), cv::Scalar(), false, false);
    net.setInput(inputBlob, "input");//set the network input, "data" is the name of the input layer

    vector<cv::String> blobNames;
    blobNames.emplace_back("num_net/pred");

    vector<Mat> outputs;
    net.forward(outputs, blobNames);
    return argmax(outputs[0]);

}

int ElixirDetect::argmax(Mat &src) {
    auto *data = (float *) src.data;
    int max_index = 0;
    float max_value = -99;
    for (int c = 0; c < src.cols; c++) {
        if (data[c] > max_value) {
            max_index = c;
            max_value = data[c];
        }
    }
    if (max_index > 0.95) {
        return max_index;
    } else {
        return this->pre_elixir;
    }
}

void ElixirDetect::reset() {
    this->pre_elixir = 0;
}


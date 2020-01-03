#include <iostream>
#include "agent.h"
#include "detect/running_detect.h"
#include "dnn/card_detect.h"

#include<opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main() {

    int COLOR_DICT[2][3][2] = {
            {{0,   34},  {0,  34},  {142, 178}},
            {{120, 224}, {80, 154}, {0,   50}}
    };

    std::cout << "Hello, World!" << std::endl;
    
    
    CardDetect cardDetect;

    Mat mat = cv::imread("./84.jpg");
    cv::resize(mat, mat, cv::Size(63, 64));

    CardResult predict = cardDetect.predict(mat);

    cout << "predict:" << predict.card_type << endl;


    RunningDetect runningDetect;

    VideoCapture capture("./record1.mp4");//获取视频
    if (!capture.isOpened())
        return -1;
    double rate = capture.get(CAP_PROP_FPS);
    int delay = 10000 / rate;
    Mat framepro, frame, dframe;
    bool flag = false;
    
    while (capture.read(frame)) {
        if (false == flag) {
            cv::resize(frame, frame, cv::Size(540, 960));
            framepro = frame.clone();
            flag = true;
        } else {
            cv::resize(frame, frame, cv::Size(540, 960));
            // Mat diff_mat = runningDetect.diff_mat(frame, framepro);

            runningDetect.has_split_line(frame);

//            Mat fine_mat = runningDetect.noise_filter(diff_mat);
//
//            Mat fine = runningDetect.exract_attention(frame, fine_mat);

//            threshold(dframe, dframe, 125, 255, THRESH_BINARY);//阈值分割
            // imshow("image", frame);
            // imshow("test", diff_mat);
//            imshow("fine_mat", fine_mat);
//            imshow("fine", fine);

            waitKey(0);
        }
    }


    return 0;
}
#include <iostream>
#include "agent.h"
#include "detect/running_detect.h"
#include "dnn/card_detect.h"

using namespace cv;
using namespace std;

int main() {
    std::cout << "Hello, World!" << std::endl;

    RunningDetect runningDetect;

    FinishDetect finishDetect;

    ElixirDetect elixirDetect;

    Mat img = cv::imread("./147.jpg");
    cv::resize(img, img, cv::Size(540, 960));
    finishDetect.detect_finish(img, 0);
    int result = elixirDetect.detect_elixir(img);


    VideoCapture cap;

    int skip_frames = 0;

    string root = ".";

    const string video_path = root + "/s1.mp4";
    cap.open(video_path);

    if (!cap.isOpened()) {
        cerr << "Unable to open video capture." << endl;
        return -1;
    }

    if (skip_frames > 0) {
        cap.set(CAP_PROP_POS_FRAMES, skip_frames);
    }


    ClashRoyalAgent clashRoyalAgent;

    int gameId = 10080;
    clashRoyalAgent.init_agent(0);
    Mat pattern;
    while (true) {
        Mat im = cv::imread("./401.jpg");

        cap >> im;

        if (im.data == nullptr) {
            break;
        }
        cv::resize(im, im, cv::Size(540, 960));
        cv::imshow("or", im);
        cv::waitKey(10);
//        elixirDetect.detect_elixir(im, 0);

//        runningDetect.detect_running(im, 0);
        Result r{};
        const Result &re = clashRoyalAgent.detect_mat(im, r);
//
        if (re.frame_state == ERROR_STATE) {
            cout << "error" << endl;
        } else {
            if (re.frame_state == MENU_STATE) {
                cout << "id:" << gameId << "  in hall:" << re.index << endl;
            } else if (re.frame_state == RUNNING_STATE) {
                cout << "id:" << gameId << "  running:" << re.frame_index << "  time:" << re.time << endl;
            } else if (re.frame_state == FINISH_STATE) {
                cout << "id:" << gameId << "  is_finish:" << re.win << endl;
                clashRoyalAgent.init_agent(++gameId);
            }
        }


    }

    return 0;
}
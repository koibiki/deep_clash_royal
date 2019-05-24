#include <iostream>
#include "detect/running_detect.h"
#include "dnn/card_detect.h"
#include "detect/button_detect.h"
#include "detect/menu_detect.h"
#include "dnn/elixir_detect.h"
#include "detect/finish_detect.h"
#include "dnn/hp_detect.h"
#include "agent.h"

using namespace cv;
using namespace std;

int main() {
    std::cout << "Hello, World!" << std::endl;

    RunningDetect runningDetect;

    FinishDetect finishDetect;

    HpDetect hpDetect;

    ElixirDetect elixirDetect;

    Menu menu;

    ButtonDetect buttonDetect;


    ClashRoyalAgent clashRoyalAgent;

    int gameId = 10080;
    clashRoyalAgent.init_agent(0);

    Mat img = cv::imread("G:\\PyCharmProjects\\deep_clash_royal\\env\\84.jpg");
    cv::resize(img, img, cv::Size(540, 960));

    hpDetect.detect_hp(img);
    


    for (int i = 0; i < 650; i++) {

        Mat img = cv::imread("D:\\gym_data\\clash_royal\\836200159\\running\\" + to_string(i) + ".jpg");
        cv::resize(img, img, cv::Size(540, 960));

        Result r{};
        const Result &re = clashRoyalAgent.detect_mat(img, r);
        runningDetect.detect_running(img, i);
    }


    return 0;
}
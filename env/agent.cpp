//
// Created by holaverse on 19-4-23.
//

#include "agent.h"

ClashRoyalAgent::ClashRoyalAgent() {
    std::cout << "c++ init ClashRoyalAgent." << std::endl;
}

result ClashRoyalAgent::detect_frame(py_mat mat, result r) {
    cv::TickMeter tm;
    tm.start();
    Mat src = transfer_mat(mat);
    result detect_result = detect_mat(src, r);
    tm.stop();
    detect_result.milli = static_cast<float>(tm.getTimeMilli());
    return detect_result;
}


result ClashRoyalAgent::detect_mat(Mat src, result r) {
    if (currentGameState == MENU_STATE) {
        MenuResult menuResult = menu.detect_tap_location(src, frame_index);
        if (menuResult.inGame) {
            r.frame_state = MENU_STATE;
            r.index = menuResult.index;
            r.is_grey = menuResult.isGrey;
            if (r.index == 3 && !r.is_grey) {
                const ButtonResult &buttonResult = buttonDetect.detect_button(src, frame_index);
                r.purple_loc[0] = buttonResult.purple_button_loc[0];
                r.purple_loc[1] = buttonResult.purple_button_loc[1];
                r.yellow_loc[0] = buttonResult.yellow_button_loc[0];
                r.yellow_loc[1] = buttonResult.yellow_button_loc[1];
            }
        } else {
            RunningResult runningResult = runningDetect.detect_running(src, frame_index);
            if (runningResult.isRunning) {
                currentGameState = RUNNING_STATE;
                start_time = time(0);
                r.frame_state = RUNNING_STATE;
                r.opp_crown = runningResult.opp_crown;
                r.mine_crown = runningResult.mine_crown;
                r.frame_index = frame_index++;
                r.time = 0;
                for (int i = 0; i < 4; i++) {
                    Mat card_mat = src(rects[i]);
                    const CardResult &cardResult = this->cardDetect.predict(card_mat);
                    r.available[i] = cardResult.available;
                    if (cardResult.prob < 0.95) {
                        r.card_type[i] = pre_type[i];
                    } else {
                        r.card_type[i] = cardResult.card_type;
                        pre_type[i] = cardResult.card_type;
                    }
                    r.prob[i] = cardResult.prob;
                }

                r.remain_elixir = elixirDetect.detect_elixir(src);

            } else {
                FinishResult finishResult = finishDetect.detect_finish(src, frame_index);
                if (finishResult.is_finish) {
                    finish_count++;
                }
                if (finishResult.is_finish && finish_count > 5) {
                    currentGameState = MENU_STATE;
                    r.frame_state = FINISH_STATE;
                    r.battle_result = finishResult.battle_result;
                    r.frame_index = frame_index++;

                } else {
                    r.frame_state = ERROR_STATE;
                }
            }
        }
    } else if (currentGameState == RUNNING_STATE) {
        RunningResult runningResult = runningDetect.detect_running(src, frame_index);
        if (runningResult.isRunning) {
            r.frame_state = RUNNING_STATE;
            r.opp_crown = runningResult.opp_crown;
            r.mine_crown = runningResult.mine_crown;
            r.frame_index = frame_index++;
            r.time = time(0) - start_time;
            for (int i = 0; i < 4; i++) {
                Mat card_mat = src(rects[i]);
                const CardResult &cardResult = this->cardDetect.predict(card_mat);
                r.available[i] = cardResult.available;
                if (cardResult.prob < 0.95) {
                    r.card_type[i] = pre_type[i];
                } else {
                    r.card_type[i] = cardResult.card_type;
                    pre_type[i] = cardResult.card_type;
                }
                r.prob[i] = cardResult.prob;
            }
            r.remain_elixir = elixirDetect.detect_elixir(src);
        } else {
            FinishResult finishResult = finishDetect.detect_finish(src, frame_index);
            if (finishResult.is_finish) {
                finish_count++;
            }
            if (finishResult.is_finish && finish_count > 5) {
                currentGameState = MENU_STATE;
                r.frame_state = FINISH_STATE;
                r.battle_result = finishResult.battle_result;
                r.frame_index = frame_index++;
            } else {
                r.frame_state = ERROR_STATE;
            }
        }
    }
    r.game_state = currentGameState;
    return r;
}

cv::Mat ClashRoyalAgent::transfer_mat(py_mat mat) {
    int type = CV_8UC1;
    if (mat.channel == 3) {
        type = CV_8UC3;
    }
    cv::Mat image(mat.height, mat.width, type);
    for (int row = 0; row < mat.height; row++) {
        uchar *pxvec = image.ptr<uchar>(row);
        for (int col = 0; col < mat.width; col++) {
            for (int c = 0; c < mat.channel; c++) {
                pxvec[col * mat.channel + c] = mat.frame_data[row * mat.width * mat.channel + mat.channel * col + c];
            }
        }
    }
    return image;
}

void ClashRoyalAgent::init_agent(int gameId) {
    this->frame_index = 0;
    this->currentGameState = MENU_STATE;
    this->gameId = gameId;
    this->cardDetect.reset();
    this->elixirDetect.reset();
    for (int i = 0; i < 4; i++) {
        this->pre_type[i] = 0;
    }
    this->finish_count = 0;
}






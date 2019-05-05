//
// Created by holaverse on 19-4-29.
//

#ifndef CLASH_ROYAL_AGENT_CARD_DETECT_H
#define CLASH_ROYAL_AGENT_CARD_DETECT_H

#include <iostream>
#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/dnn.hpp"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"

using std::string;
using std::vector;
using std::to_string;
using std::cout;
using std::endl;
using std::sort;
using cv::Mat;
using cv::Rect;
using cv::Point;
using cv::RotatedRect;
using cv::Size;

struct CardResult {
    string card_name = "empty";
    int card_type = 0;
    int available = 0;
    float prob = 0.0;
};

class CardDetect {

private:
    const char *card_dict_json =
            R"({"0": "empty", "1": "Furnace", "2": "GoblinBarrel", "3": "DarkPrince", "4": "Prince",
            "5": "RoyalHogs", "6": "Giant", "7": "Arrows", "8": "FireSpirit", "9": "Bomber",
            "10": "IceGolem", "11": "X-box", "12": "BarbarianHut", "13": "Witch",
            "14": "Knight", "15": "Hunter", "16": "Poison", "17": "GoblinHut",
            "18": "P.E.K.K.A", "19": "BattleRam", "20": "GiantSnowball", "21": "Musketeer",
            "22": "Princess", "23": "Archers", "24": "DartGoblin", "25": "InfernoDragon",
            "26": "MegaKnight", "27": "ThreeMusketeer", "28": "IceWizard", "29": "SkeletonArmy",
            "30": "HogRider", "31": "Golem", "32": "Fireball", "33": "Valkyrie",
            "34": "Zap", "35": "Guards", "36": "IceSpirit", "37": "Tornado",
            "38": "BarbarianBarrel", "39": "Rage", "40": "miniP.E.K.K.A", "41": "Miner",
            "42": "Wizard", "43": "GiantSkeleton", "44": "BabyDragon", "45": "ElectroDragon",
            "46": "InfernoTower", "47": "Ballon", "48": "Minions", "49": "MinionHorde"})";
    rapidjson::Document char_dict;
    cv::dnn::Net net;

    int pre_type[4];

    bool debug = false;

    int argmax(Mat &src);

    void load_dict();

public:

    CardDetect();

    void reset();

    bool load_model(string mode_path);

    CardResult predict(Mat &src);

};


#endif //CLASH_ROYAL_AGENT_CARD_DETECT_H

//
// Created by holaverse on 19-4-23.
//

#include "agent.h"

extern "C"
{
ClashRoyalAgent agent;

// 传递结构体到 c++ 层 并返回结构体
result detect_frame(py_mat pymat, result t) {
    return agent.detect_frame(pymat, t);
}

void init_game(int gameId){
    agent.init_agent(gameId);
}

}

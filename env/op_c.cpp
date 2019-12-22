//
// Created by holaverse on 19-4-23.
//

#include "agent.h"

extern "C"
{
ClashRoyalAgent host(0);

ClashRoyalAgent guest(1);

// 传递结构体到 c++ 层 并返回结构体
result detect_frame(py_mat pymat, result t, int agent_id) {
    if (agent_id == 0) {
        return host.detect_frame(pymat, t);
    }else{
        return guest.detect_frame(pymat, t);
    }
}

void init_game(int gameId, int agent_id){
    if(agent_id == 0){
        host.init_agent(gameId);
    }else{
        guest.init_agent(gameId);
    }
}

}

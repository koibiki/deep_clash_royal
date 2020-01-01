import numpy as np

CARD_TYPE = 93

card_dict = {0: "empty", 1: "Prince", 2: "Bomber", 3: "Arrows", 4: "Witch", 5: "miniP.E.K.K.A", 6: "Zap",
             7: "BabyDragon", 8: "Musketeer", 9: "Fireball", 10: "SkeletonArmy", 11: "Minions", 12: "Valkyrie"}

elixir_dict = {"empty": 0.0, "Prince": 0.5, "Bomber": 0.3, "Arrows": 0.3, "Witch": 0.5, "miniP.E.K.K.A": 0.4,
               "Zap": 0.2, "BabyDragon": 0.4, "Musketeer": 0.4, "Fireball": 0.4, "SkeletonArmy": 0.3, "Minions": 0.3,
               "Valkyrie": 0.4,}


def parse_frame_state(result):
    remain_elixir = result.remain_elixir / 10
    double_elixir = 1. if result.time > 60 * 2 - 1 else 0.
    dead_im = 1. if result.time > 60 * 3 - 1 else 0.

    env_state = [remain_elixir, double_elixir, dead_im]

    card_type = np.array(result.card_type)
    card_available = np.array(result.available)
    card_type = [int(card_type[0]), int(card_type[1]), int(card_type[2]), int(card_type[3]), ]
    card_property = [int(card_available[0]) * 1., elixir_dict[card_dict[card_type[0]]],
                     int(card_available[1]) * 1., elixir_dict[card_dict[card_type[1]]],
                     int(card_available[2]) * 1., elixir_dict[card_dict[card_type[2]]],
                     int(card_available[3]) * 1., elixir_dict[card_dict[card_type[3]]]]
    return env_state, card_type, card_property


def parse_card_type_and_property(card_state):
    card_state_array = np.array(card_state)
    if len(card_state_array.shape) == 1:
        card_type = [card_state[0], card_state[3], card_state[6], card_state[9]]
        card_property = card_state[1:3] + card_state[4:6] + card_state[7:9] + card_state[10:]
    elif len(card_state_array.shape) == 2:
        card0 = card_state_array[:, 0].reshape(-1, 1)
        card1 = card_state_array[:, 3].reshape(-1, 1)
        card2 = card_state_array[:, 6].reshape(-1, 1)
        card3 = card_state_array[:, 9].reshape(-1, 1)
        card_type = np.concatenate([card0, card1, card2, card3], axis=-1).astype(np.int)
        card_property = np.concatenate([card_state[:, 1:3],
                                        card_state[:, 4:6],
                                        card_state[:, 7:9],
                                        card_state[:, 10:]], axis=-1).astype(np.float)
    else:
        raise Exception("error card_state shape:{}".format(card_state.shape))
    return card_type, card_property


def calu_available_card(card_type, card_property):
    card_state = np.array(card_type)
    if len(card_state.shape) == 1:
        available_card = [0 for _ in range(94)]
        available_card[card_type[0]] = 0 if card_property[0] < 0.1 else 1
        available_card[card_type[1]] = 0 if card_property[2] < 0.1 else 1
        available_card[card_type[2]] = 0 if card_property[4] < 0.1 else 1
        available_card[card_type[3]] = 0 if card_property[6] < 0.1 else 1
        available_card[0] = 1
        return available_card
    elif len(card_state.shape) == 2:
        available_cards = []
        for i, c in enumerate(card_type):
            available_card = [0 for _ in range(94)]
            available_card = np.asarray([available_card])
            available_card[:, c[0]] = 0 if card_property[i][0] < 0.1 else 1
            available_card[:, c[1]] = 0 if card_property[i][2] < 0.1 else 1
            available_card[:, c[2]] = 0 if card_property[i][4] < 0.1 else 1
            available_card[:, c[3]] = 0 if card_property[i][6] < 0.1 else 1
            available_card[:, 0] = 1
            available_cards.append(available_card)
        available_cards = np.concatenate(available_cards, axis=0)
        return available_cards
    else:
        raise Exception("error card_state shape:{}".format(card_state.shape))


def parse_running_state(state):
    card_type_state = parse_card_state(state[:4].astype(np.int32), state[4:8].astype(np.int32))
    state_vector = np.concatenate([card_type_state, state[8:]], axis=0)
    return state_vector


def parse_card_state(card_type, available):
    card_type_state = np.zeros(1 + 92 * 3, dtype=np.float32)
    card_type_state[0] = 1
    for i in range(4):
        if card_type[i] != 0:
            card_type_state[card_type[i]] = available[i]
            card_type_state[92 + 1 + card_type[i] - 1] = 1
            card_type_state[92 * 2 + 1 + card_type[i] - 1] = elixir_dict[card_dict[card_type[i]]]
    return card_type_state

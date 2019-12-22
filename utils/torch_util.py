import torch
import numpy as np


def gen_torch_tensor(img, env_state, card_type, card_property, device):
    img = torch.from_numpy(img).float().to(device)
    env_state = torch.from_numpy(np.array(env_state).astype(np.float)).float().to(device)
    card_type = torch.from_numpy(np.array(card_type).astype(np.int)).long().to(device)
    card_property = torch.from_numpy(np.array(card_property).astype(np.float)).float().to(device)
    return img, env_state, card_type, card_property

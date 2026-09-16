import torch
import torch.nn as nn

from app.config import (
    NUM_ACTION_TYPES,
    MOVE_TARGET_DIM, BATTLE_TARGET_DIM,
    HIDDEN1, HIDDEN2, HEAD_HIDDEN,
    CONTEXT_OFFSET,
    CTX_DT_OFF, CTX_DT_LEN,
    CTX_REC_OFF, CTX_REC_LEN,
    CTX_ARMY_OFF, CTX_ARMY_LEN,
    CTX_MOV_OFF, CTX_MOV_LEN,
    CTX_BTGT_OFF, CTX_BTGT_LEN,
    CTX_BSEL_OFF, CTX_BSEL_LEN,
    CTX_BALLOC_OFF, CTX_BALLOC_LEN,
    CTX_BRET_OFF, CTX_BRET_LEN,
)


class GraphConquestNN(nn.Module):
    """
    Multi-head network v9: context-free trunk with per-head context shortcut.

    Input is the full encoding [1148]; the trunk consumes only the context-free
    core x[:, :924], so the value head (trunk only) is a pure state value usable
    for TD differences. Policy heads receive their context via the shortcuts.

    state_core[924] → Dense(1024, ReLU) → Dense(256, ReLU) = trunk
    Each head: concat(trunk[256], decision_type[7], own_context[N]) → Dense(64, ReLU) → head

    Categorical heads (logits, masked softmax at use site):
      action_type[5], move_target[16], battle_target[17]
    """

    def __init__(self):
        super().__init__()
        self.dense1 = nn.Linear(CONTEXT_OFFSET, HIDDEN1)
        self.dense2 = nn.Linear(HIDDEN1, HIDDEN2)

        T = HIDDEN2  # 256
        D = CTX_DT_LEN  # 7

        self.value_hidden = nn.Linear(T, HEAD_HIDDEN)
        self.value_head = nn.Linear(HEAD_HIDDEN, 1)

        self.action_type_hidden = nn.Linear(T + D + CTX_ARMY_LEN, HEAD_HIDDEN)
        self.action_type_head = nn.Linear(HEAD_HIDDEN, NUM_ACTION_TYPES)

        self.split_fraction_hidden = nn.Linear(T + D + CTX_ARMY_LEN, HEAD_HIDDEN)
        self.split_fraction_head = nn.Linear(HEAD_HIDDEN, 1)

        self.disband_fraction_hidden = nn.Linear(T + D + CTX_ARMY_LEN, HEAD_HIDDEN)
        self.disband_fraction_head = nn.Linear(HEAD_HIDDEN, 1)

        self.recruit_fraction_hidden = nn.Linear(T + D + CTX_REC_LEN, HEAD_HIDDEN)
        self.recruit_fraction_head = nn.Linear(HEAD_HIDDEN, 1)

        self.move_target_hidden = nn.Linear(T + D + CTX_MOV_LEN, HEAD_HIDDEN)
        self.move_target_head = nn.Linear(HEAD_HIDDEN, MOVE_TARGET_DIM)

        self.battle_target_hidden = nn.Linear(T + D + CTX_BTGT_LEN, HEAD_HIDDEN)
        self.battle_target_head = nn.Linear(HEAD_HIDDEN, BATTLE_TARGET_DIM)

        self.battle_select_hidden = nn.Linear(T + D + CTX_BSEL_LEN, HEAD_HIDDEN)
        self.battle_select_head = nn.Linear(HEAD_HIDDEN, 1)

        self.kill_fraction_hidden = nn.Linear(T + D + CTX_BALLOC_LEN, HEAD_HIDDEN)
        self.kill_fraction_head = nn.Linear(HEAD_HIDDEN, 1)

        self.battle_retreat_hidden = nn.Linear(T + D + CTX_BRET_LEN, HEAD_HIDDEN)
        self.battle_retreat_head = nn.Linear(HEAD_HIDDEN, 1)

    def forward(self, x):
        h = torch.relu(self.dense1(x[:, :CONTEXT_OFFSET]))
        h = torch.relu(self.dense2(h))

        # Extract context segments
        B = CONTEXT_OFFSET
        dt = x[:, B + CTX_DT_OFF:B + CTX_DT_OFF + CTX_DT_LEN]
        rec = x[:, B + CTX_REC_OFF:B + CTX_REC_OFF + CTX_REC_LEN]
        army = x[:, B + CTX_ARMY_OFF:B + CTX_ARMY_OFF + CTX_ARMY_LEN]
        mov = x[:, B + CTX_MOV_OFF:B + CTX_MOV_OFF + CTX_MOV_LEN]
        btgt = x[:, B + CTX_BTGT_OFF:B + CTX_BTGT_OFF + CTX_BTGT_LEN]
        bsel = x[:, B + CTX_BSEL_OFF:B + CTX_BSEL_OFF + CTX_BSEL_LEN]
        balloc = x[:, B + CTX_BALLOC_OFF:B + CTX_BALLOC_OFF + CTX_BALLOC_LEN]
        bret = x[:, B + CTX_BRET_OFF:B + CTX_BRET_OFF + CTX_BRET_LEN]

        value = torch.sigmoid(self.value_head(torch.relu(self.value_hidden(h))))
        action_type = self.action_type_head(torch.relu(self.action_type_hidden(
            torch.cat([h, dt, army], dim=1))))
        split_frac = torch.sigmoid(self.split_fraction_head(torch.relu(self.split_fraction_hidden(
            torch.cat([h, dt, army], dim=1)))))
        disband_frac = torch.sigmoid(self.disband_fraction_head(torch.relu(self.disband_fraction_hidden(
            torch.cat([h, dt, army], dim=1)))))
        recruit_frac = torch.sigmoid(self.recruit_fraction_head(torch.relu(self.recruit_fraction_hidden(
            torch.cat([h, dt, rec], dim=1)))))
        move_target = self.move_target_head(torch.relu(self.move_target_hidden(
            torch.cat([h, dt, mov], dim=1))))
        battle_target = self.battle_target_head(torch.relu(self.battle_target_hidden(
            torch.cat([h, dt, btgt], dim=1))))
        battle_select = torch.sigmoid(self.battle_select_head(torch.relu(self.battle_select_hidden(
            torch.cat([h, dt, bsel], dim=1)))))
        kill_frac = torch.sigmoid(self.kill_fraction_head(torch.relu(self.kill_fraction_hidden(
            torch.cat([h, dt, balloc], dim=1)))))
        battle_retreat = torch.sigmoid(self.battle_retreat_head(torch.relu(self.battle_retreat_hidden(
            torch.cat([h, dt, bret], dim=1)))))

        return (
            value, action_type, split_frac, disband_frac, recruit_frac,
            move_target, battle_target, battle_select,
            kill_frac, battle_retreat,
        )

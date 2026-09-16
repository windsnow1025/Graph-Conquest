# Constants matching TypeScript definitions (v9, 10 heads + context shortcut,
# context-free trunk consuming state[0:CONTEXT_OFFSET],
# categorical move_target[16] and battle_target[17] heads).
# Must stay in sync with:
#   src/AI/nn/StateEncoder.ts
#   src/AI/nn/NNModel.ts
#   training/src/SampleTypes.ts

STATE_SIZE = 1148
NUM_NODES = 16
NUM_ACTION_TYPES = 5  # EXIT, MERGE, MOVE, SPLIT, DISBAND

MOVE_TARGET_DIM = NUM_NODES        # 16 destination nodes
BATTLE_TARGET_DIM = NUM_NODES + 1  # 16 nodes + stop

# Context: input[924:1148], 224 features
CONTEXT_OFFSET = 924   # 5 + 18 + 21 + 880
CONTEXT_SIZE = 224     # 7 + 20 + 28 + 39 + 16 + 51 + 46 + 17

# Context segment offsets (relative to CONTEXT_OFFSET)
CTX_DT_OFF = 0;       CTX_DT_LEN = 7
CTX_REC_OFF = 7;      CTX_REC_LEN = 20
CTX_ARMY_OFF = 27;    CTX_ARMY_LEN = 28
CTX_MOV_OFF = 55;     CTX_MOV_LEN = 39
CTX_BTGT_OFF = 94;    CTX_BTGT_LEN = 16
CTX_BSEL_OFF = 110;   CTX_BSEL_LEN = 51
CTX_BALLOC_OFF = 161; CTX_BALLOC_LEN = 46
CTX_BRET_OFF = 207;   CTX_BRET_LEN = 17

# Per-sample binary record layout (all float32):
#   state[1148] + value(1) + policyWeight(1)
#   + actionTypeTarget(1) + actionTypeMask[5]
#   + splitFraction(1) + splitMask(1)
#   + disbandFraction(1) + disbandMask(1)
#   + recruitFraction(1) + recruitMask(1)
#   + moveTargetIdx(1) + moveMask[16]
#   + battleTargetIdx(1) + battleTargetMask[17]
#   + battleSelect(1) + battleSelectMask(1)
#   + killFraction(1) + killFracMask(1)
#   + battleRetreat(1) + retreatMask(1)
# Total: 1148 + 2 + 6 + 6 + 17 + 18 + 6 = 1203

OFF_STATE = 0
OFF_VALUE = STATE_SIZE                                    # 1148
OFF_POLICY_WEIGHT = OFF_VALUE + 1                         # 1149
OFF_ACTION_TYPE = OFF_POLICY_WEIGHT + 1                   # 1150
OFF_ACTION_MASK = OFF_ACTION_TYPE + 1                     # 1151
OFF_SPLIT_FRAC = OFF_ACTION_MASK + NUM_ACTION_TYPES       # 1156
OFF_SPLIT_MASK = OFF_SPLIT_FRAC + 1                       # 1157
OFF_DISBAND_FRAC = OFF_SPLIT_MASK + 1                     # 1158
OFF_DISBAND_MASK = OFF_DISBAND_FRAC + 1                   # 1159
OFF_RECRUIT_FRAC = OFF_DISBAND_MASK + 1                   # 1160
OFF_RECRUIT_MASK = OFF_RECRUIT_FRAC + 1                   # 1161
OFF_MOVE_TARGET = OFF_RECRUIT_MASK + 1                    # 1162
OFF_MOVE_MASK = OFF_MOVE_TARGET + 1                       # 1163 (16 floats)
OFF_BATTLE_TARGET = OFF_MOVE_MASK + MOVE_TARGET_DIM       # 1179
OFF_BATTLE_TARGET_MASK = OFF_BATTLE_TARGET + 1            # 1180 (17 floats)
OFF_BATTLE_SELECT = OFF_BATTLE_TARGET_MASK + BATTLE_TARGET_DIM  # 1197
OFF_BATTLE_SELECT_MASK = OFF_BATTLE_SELECT + 1            # 1198
OFF_KILL_FRAC = OFF_BATTLE_SELECT_MASK + 1                # 1199
OFF_KILL_FRAC_MASK = OFF_KILL_FRAC + 1                    # 1200
OFF_RETREAT = OFF_KILL_FRAC_MASK + 1                      # 1201
OFF_RETREAT_MASK = OFF_RETREAT + 1                        # 1202

SAMPLE_FLOATS = OFF_RETREAT_MASK + 1                      # 1203

# Hidden layer sizes
HIDDEN1 = 1024
HIDDEN2 = 256
HEAD_HIDDEN = 64

# Constants matching TypeScript definitions (v10, 11 heads + context shortcut,
# context-free trunk consuming state[0:CONTEXT_OFFSET], categorical
# action_type[3], move_target[16] and battle_target[17] heads, per-group
# fraction heads move_fraction[4], disband_fraction[6], commit_fraction[3]).
# Must stay in sync with:
#   src/AI/nn/StateEncoder.ts
#   src/AI/nn/NNModel.ts
#   training/src/SampleTypes.ts

STATE_SIZE = 1360
NUM_NODES = 16
NUM_ACTION_TYPES = 3  # EXIT, MOVE, DISBAND
NUM_MOVE_GROUPS = 4
NUM_DISBAND_GROUPS = 6
NUM_COMMIT_GROUPS = 3

MOVE_TARGET_DIM = NUM_NODES        # 16 destination nodes
BATTLE_TARGET_DIM = NUM_NODES + 1  # 16 nodes + stop

# Context: input[1068:1360], 292 features
CONTEXT_OFFSET = 1068  # 5 + 18 + 21 + 1024
CONTEXT_SIZE = 292     # 8 + 20 + 36 + 43 + 47 + 16 + 59 + 46 + 17

# Context segment offsets (relative to CONTEXT_OFFSET)
CTX_DT_OFF = 0;       CTX_DT_LEN = 8
CTX_REC_OFF = 8;      CTX_REC_LEN = 20
CTX_ARMY_OFF = 28;    CTX_ARMY_LEN = 36
CTX_MOV_OFF = 64;     CTX_MOV_LEN = 43
CTX_MCNT_OFF = 107;   CTX_MCNT_LEN = 47
CTX_BTGT_OFF = 154;   CTX_BTGT_LEN = 16
CTX_BSEL_OFF = 170;   CTX_BSEL_LEN = 59
CTX_BALLOC_OFF = 229; CTX_BALLOC_LEN = 46
CTX_BRET_OFF = 275;   CTX_BRET_LEN = 17

# Per-sample binary record layout (all float32):
#   state[1360] + value(1) + policyWeight(1)
#   + actionTypeTarget(1) + actionTypeMask[3]
#   + moveFraction[4] + moveFractionMask[4]
#   + disbandFraction[6] + disbandMask[6]
#   + recruitFraction(1) + recruitMask(1)
#   + moveTargetIdx(1) + moveMask[16]
#   + battleTargetIdx(1) + battleTargetMask[17]
#   + battleSelect(1) + battleSelectMask(1)
#   + commitFraction[3] + commitMask[3]
#   + killFraction(1) + killFracMask(1)
#   + battleRetreat(1) + retreatMask(1)
# Total: 1360 + 2 + 4 + 8 + 12 + 2 + 17 + 18 + 2 + 6 + 4 = 1435

OFF_STATE = 0
OFF_VALUE = STATE_SIZE                                    # 1360
OFF_POLICY_WEIGHT = OFF_VALUE + 1                         # 1361
OFF_ACTION_TYPE = OFF_POLICY_WEIGHT + 1                   # 1362
OFF_ACTION_MASK = OFF_ACTION_TYPE + 1                     # 1363 (3 floats)
OFF_MOVE_FRAC = OFF_ACTION_MASK + NUM_ACTION_TYPES        # 1366 (4 floats)
OFF_MOVE_FRAC_MASK = OFF_MOVE_FRAC + NUM_MOVE_GROUPS      # 1370 (4 floats)
OFF_DISBAND_FRAC = OFF_MOVE_FRAC_MASK + NUM_MOVE_GROUPS   # 1374 (6 floats)
OFF_DISBAND_MASK = OFF_DISBAND_FRAC + NUM_DISBAND_GROUPS  # 1380 (6 floats)
OFF_RECRUIT_FRAC = OFF_DISBAND_MASK + NUM_DISBAND_GROUPS  # 1386
OFF_RECRUIT_MASK = OFF_RECRUIT_FRAC + 1                   # 1387
OFF_MOVE_TARGET = OFF_RECRUIT_MASK + 1                    # 1388
OFF_MOVE_MASK = OFF_MOVE_TARGET + 1                       # 1389 (16 floats)
OFF_BATTLE_TARGET = OFF_MOVE_MASK + MOVE_TARGET_DIM       # 1405
OFF_BATTLE_TARGET_MASK = OFF_BATTLE_TARGET + 1            # 1406 (17 floats)
OFF_BATTLE_SELECT = OFF_BATTLE_TARGET_MASK + BATTLE_TARGET_DIM  # 1423
OFF_BATTLE_SELECT_MASK = OFF_BATTLE_SELECT + 1            # 1424
OFF_COMMIT_FRAC = OFF_BATTLE_SELECT_MASK + 1              # 1425 (3 floats)
OFF_COMMIT_MASK = OFF_COMMIT_FRAC + NUM_COMMIT_GROUPS     # 1428 (3 floats)
OFF_KILL_FRAC = OFF_COMMIT_MASK + NUM_COMMIT_GROUPS       # 1431
OFF_KILL_FRAC_MASK = OFF_KILL_FRAC + 1                    # 1432
OFF_RETREAT = OFF_KILL_FRAC_MASK + 1                      # 1433
OFF_RETREAT_MASK = OFF_RETREAT + 1                        # 1434

SAMPLE_FLOATS = OFF_RETREAT_MASK + 1                      # 1435

# Hidden layer sizes
HIDDEN1 = 1024
HIDDEN2 = 256
HEAD_HIDDEN = 64

import numpy as np
import torch
import torch.nn.functional as F

from app.config import (
    STATE_SIZE, NUM_ACTION_TYPES,
    MOVE_TARGET_DIM, BATTLE_TARGET_DIM,
    OFF_STATE, OFF_VALUE, OFF_POLICY_WEIGHT,
    OFF_ACTION_TYPE, OFF_ACTION_MASK,
    OFF_SPLIT_FRAC, OFF_SPLIT_MASK,
    OFF_DISBAND_FRAC, OFF_DISBAND_MASK,
    OFF_RECRUIT_FRAC, OFF_RECRUIT_MASK,
    OFF_MOVE_TARGET, OFF_MOVE_MASK,
    OFF_BATTLE_TARGET, OFF_BATTLE_TARGET_MASK,
    OFF_BATTLE_SELECT, OFF_BATTLE_SELECT_MASK,
    OFF_KILL_FRAC, OFF_KILL_FRAC_MASK,
    OFF_RETREAT, OFF_RETREAT_MASK,
)

HEAD_NAMES = [
    "val", "act", "spl", "dis", "rec",
    "mov", "btgt", "bsel", "kfr", "ret",
]
NUM_HEADS = 10


def _active_mean(values, mask):
    count = mask.sum()
    if count == 0:
        return values.sum() * 0
    return (values * mask).sum() / count


def _bce_masked(pred, target, mask):
    """Binary cross-entropy, masked."""
    eps = 1e-7
    p = pred.squeeze(1).clamp(eps, 1 - eps)  # [B,1] -> [B]
    bce = -(target * p.log() + (1 - target) * (1 - p).log())
    return _active_mean(bce, mask)


def _mse_masked(pred, target, mask):
    """MSE, masked."""
    mse = (pred.squeeze(1) - target) ** 2
    return _active_mean(mse, mask)


# Defense in depth: policy weights reaching this trainer must be NON-NEGATIVE.
# Negative weights have been introduced and removed several times in this
# project's history and collapsed the policy every time. A negative weight
# flips CE/BCE into a push-away objective that is unbounded below (loss -> -inf
# as the action's probability -> 0) and whose gradient never saturates, so in
# offline epochs it out-competes the saturating attraction terms and crushes
# every action the policy actually takes. The TS side clamps weights at the
# source (SelfPlay.recordsToSamples); this cap only bounds the damage if a
# negative weight ever slips through again.
PUSH_AWAY_CLAMP = 6.0


def _weighted_active_mean(values, mask, weight):
    """Weighted mean over active samples: sum(loss * w) / sum(w).

    Normalizing by the weight sum (not the sample count) makes the objective
    invariant to near-zero-weight samples: they neither dilute the step size
    (as count normalization did) nor need a magnitude threshold at export.
    Weights are expected non-negative (see the note above PUSH_AWAY_CLAMP).
    """
    total_w = (mask * weight).sum()
    if total_w <= 1e-8:
        return values.sum() * 0
    return (values * mask * weight).sum() / total_w


def _clamp_push_away(losses, weight):
    return torch.where(weight < 0, losses.clamp(max=PUSH_AWAY_CLAMP), losses)


def _bce_weighted(pred, target, mask, weight):
    """Binary cross-entropy, masked and weighted."""
    eps = 1e-7
    p = pred.squeeze(1).clamp(eps, 1 - eps)  # [B,1] -> [B]
    bce = -(target * p.log() + (1 - target) * (1 - p).log())
    return _weighted_active_mean(_clamp_push_away(bce, weight), mask, weight)


def _mse_weighted(pred, target, mask, weight):
    """MSE, masked and weighted."""
    mse = (pred.squeeze(1) - target) ** 2
    return _weighted_active_mean(mse, mask, weight)


def _ce_masked_weighted(logits, target, legal_mask, active, weight):
    """Categorical cross-entropy toward target class with illegal options masked out."""
    masked_logits = logits + (1 - legal_mask) * (-1e9)
    log_probs = F.log_softmax(masked_logits, dim=1)
    safe_target = target.clamp(0, logits.shape[1] - 1)
    ce = F.nll_loss(log_probs, safe_target, reduction="none")
    return _weighted_active_mean(_clamp_push_away(ce, weight), active, weight)


def _compute_losses(model, batch):
    state = batch[:, OFF_STATE:OFF_STATE + STATE_SIZE]
    value_target = batch[:, OFF_VALUE]
    value_mask = (value_target >= 0).float()  # value < 0 means no label (NaN encoded as -1)
    policy_weight = batch[:, OFF_POLICY_WEIGHT]

    action_type = batch[:, OFF_ACTION_TYPE].long()
    action_mask = batch[:, OFF_ACTION_MASK:OFF_ACTION_MASK + NUM_ACTION_TYPES]
    action_active = (action_type >= 0).float()

    split_frac = batch[:, OFF_SPLIT_FRAC]
    split_mask = batch[:, OFF_SPLIT_MASK]
    disband_frac = batch[:, OFF_DISBAND_FRAC]
    disband_mask = batch[:, OFF_DISBAND_MASK]
    recruit_frac = batch[:, OFF_RECRUIT_FRAC]
    recruit_mask = batch[:, OFF_RECRUIT_MASK]

    move_target = batch[:, OFF_MOVE_TARGET].long()
    move_mask = batch[:, OFF_MOVE_MASK:OFF_MOVE_MASK + MOVE_TARGET_DIM]
    move_active = (move_target >= 0).float()
    battle_target = batch[:, OFF_BATTLE_TARGET].long()
    bt_mask = batch[:, OFF_BATTLE_TARGET_MASK:OFF_BATTLE_TARGET_MASK + BATTLE_TARGET_DIM]
    bt_active = (battle_target >= 0).float()
    battle_select = batch[:, OFF_BATTLE_SELECT]
    bs_mask = batch[:, OFF_BATTLE_SELECT_MASK]
    kill_frac = batch[:, OFF_KILL_FRAC]
    kf_mask = batch[:, OFF_KILL_FRAC_MASK]
    retreat = batch[:, OFF_RETREAT]
    ret_mask = batch[:, OFF_RETREAT_MASK]

    (pred_value, pred_action, pred_split, pred_disband, pred_recruit,
     pred_move, pred_btarget, pred_bselect,
     pred_kfrac, pred_retreat) = model(state)

    pred_value = pred_value.squeeze(1)

    # 1. Value head (MSE) — always learn, no policy weight
    v_loss = _active_mean((pred_value - value_target) ** 2, value_mask)

    # 2. Action type (masked cross-entropy, weighted)
    a_loss = _ce_masked_weighted(pred_action, action_type, action_mask, action_active, policy_weight)

    # 3-4. Split/disband fraction (MSE, weighted)
    sf_loss = _mse_weighted(pred_split, split_frac, split_mask, policy_weight)
    df_loss = _mse_weighted(pred_disband, disband_frac, disband_mask, policy_weight)

    # 5. Recruit fraction (MSE, weighted)
    rf_loss = _mse_weighted(pred_recruit, recruit_frac, recruit_mask, policy_weight)

    # 6-7. Categorical heads (masked cross-entropy, weighted)
    mv_loss = _ce_masked_weighted(pred_move, move_target, move_mask, move_active, policy_weight)
    bt_loss = _ce_masked_weighted(pred_btarget, battle_target, bt_mask, bt_active, policy_weight)

    # 8-10. Binary/regression heads (BCE/MSE, weighted)
    bs_loss = _bce_weighted(pred_bselect, battle_select, bs_mask, policy_weight)
    kf_loss = _mse_weighted(pred_kfrac, kill_frac, kf_mask, policy_weight)
    rt_loss = _bce_weighted(pred_retreat, retreat, ret_mask, policy_weight)

    return [v_loss, a_loss, sf_loss, df_loss, rf_loss,
            mv_loss, bt_loss, bs_loss, kf_loss, rt_loss]


def eval_loss(model, data, batch_size, device):
    model.eval()
    n = data.shape[0]
    totals = [0.0] * NUM_HEADS
    num_batches = 0
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.from_numpy(data[i:min(i + batch_size, n)]).to(device)
            losses = _compute_losses(model, batch)
            for j in range(NUM_HEADS):
                totals[j] += losses[j].item()
            num_batches += 1
    d = max(num_batches, 1)
    head_losses = [t / d for t in totals]
    return sum(head_losses), head_losses


def train_epoch(model, optimizer, data, batch_size, device):
    model.train()
    n = data.shape[0]
    perm = np.random.permutation(n)
    totals = [0.0] * NUM_HEADS
    num_batches = 0

    for i in range(0, n, batch_size):
        idx = perm[i:min(i + batch_size, n)]
        batch = torch.from_numpy(data[idx]).to(device)
        losses = _compute_losses(model, batch)
        loss = sum(losses)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for j in range(NUM_HEADS):
            totals[j] += losses[j].item()
        num_batches += 1

    d = max(num_batches, 1)
    head_losses = [t / d for t in totals]
    return sum(head_losses), head_losses

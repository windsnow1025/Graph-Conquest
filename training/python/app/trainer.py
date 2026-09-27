import numpy as np
import torch
import torch.nn.functional as F

from app.config import (
    STATE_SIZE, NUM_ACTION_TYPES,
    NUM_MOVE_GROUPS, NUM_DISBAND_GROUPS, NUM_COMMIT_GROUPS, FRACTION_BINS,
    MOVE_TARGET_DIM, BATTLE_TARGET_DIM,
    OFF_STATE, OFF_VALUE, OFF_POLICY_WEIGHT,
    OFF_ACTION_TYPE, OFF_ACTION_MASK,
    OFF_MOVE_FRAC, OFF_MOVE_FRAC_MASK,
    OFF_DISBAND_FRAC, OFF_DISBAND_MASK,
    OFF_RECRUIT_FRAC, OFF_RECRUIT_MASK,
    OFF_MOVE_TARGET, OFF_MOVE_MASK,
    OFF_BATTLE_TARGET, OFF_BATTLE_TARGET_MASK,
    OFF_BATTLE_SELECT, OFF_BATTLE_SELECT_MASK,
    OFF_COMMIT_FRAC, OFF_COMMIT_MASK,
    OFF_KILL_FRAC, OFF_KILL_FRAC_MASK,
    OFF_RETREAT, OFF_RETREAT_MASK,
)

HEAD_NAMES = [
    "val", "act", "mfr", "dfr", "rec",
    "mov", "btgt", "bsel", "cfr", "kfr", "ret",
]
PPO_HEAD_NAMES = ["val", "pol", "bsel", "ret", "clip"]


def _active_mean(values, mask):
    count = mask.sum()
    if count == 0:
        return values.sum() * 0
    return (values * mask).sum() / count


# The imitation objective and the BCE heads of the PPO objective take
# NON-NEGATIVE weights only. A negative weight flips CE/BCE into a push-away
# objective that is unbounded below (loss -> -inf as the action's probability
# -> 0) and whose gradient never saturates, so in offline epochs it out-competes
# the saturating attraction terms and crushes every action the policy actually
# takes; this project saw that collapse every time negative weights entered a
# CE/BCE loss. Signed advantages enter only the clipped surrogate of the PPO
# objective, whose push-away stops at the clip. This cap bounds the damage if
# a negative weight ever reaches a CE/BCE loss again.
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


def _ce_levels_weighted(logits, target_frac, mask, weight):
    """Cross-entropy of a fraction head toward the level nearest each target fraction.

    logits [B, G*FRACTION_BINS] hold FRACTION_BINS logits per group, target_frac
    and mask are [B, G] (a single-fraction head is G = 1), weight is per sample.
    """
    groups = target_frac.shape[1]
    log_probs = F.log_softmax(logits.view(-1, groups, FRACTION_BINS), dim=2)
    levels = torch.round(target_frac * (FRACTION_BINS - 1)).long().clamp(0, FRACTION_BINS - 1)
    ce = -log_probs.gather(2, levels.unsqueeze(2)).squeeze(2)
    w = weight.unsqueeze(1)
    return _weighted_active_mean(_clamp_push_away(ce, w), mask, w)


def _ce_masked_weighted(logits, target, legal_mask, active, weight):
    """Categorical cross-entropy toward target class with illegal options masked out."""
    masked_logits = logits + (1 - legal_mask) * (-1e9)
    log_probs = F.log_softmax(masked_logits, dim=1)
    safe_target = target.clamp(0, logits.shape[1] - 1)
    ce = F.nll_loss(log_probs, safe_target, reduction="none")
    return _weighted_active_mean(_clamp_push_away(ce, weight), active, weight)


def _compute_losses(model, batch, action_class_weight):
    state = batch[:, OFF_STATE:OFF_STATE + STATE_SIZE]
    value_target = batch[:, OFF_VALUE]
    value_mask = (value_target >= 0).float()  # value < 0 means no label (NaN encoded as -1)
    policy_weight = batch[:, OFF_POLICY_WEIGHT]

    action_type = batch[:, OFF_ACTION_TYPE].long()
    action_mask = batch[:, OFF_ACTION_MASK:OFF_ACTION_MASK + NUM_ACTION_TYPES]
    action_active = (action_type >= 0).float()

    move_frac = batch[:, OFF_MOVE_FRAC:OFF_MOVE_FRAC + NUM_MOVE_GROUPS]
    move_frac_mask = batch[:, OFF_MOVE_FRAC_MASK:OFF_MOVE_FRAC_MASK + NUM_MOVE_GROUPS]
    disband_frac = batch[:, OFF_DISBAND_FRAC:OFF_DISBAND_FRAC + NUM_DISBAND_GROUPS]
    disband_mask = batch[:, OFF_DISBAND_MASK:OFF_DISBAND_MASK + NUM_DISBAND_GROUPS]
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
    commit_frac = batch[:, OFF_COMMIT_FRAC:OFF_COMMIT_FRAC + NUM_COMMIT_GROUPS]
    commit_mask = batch[:, OFF_COMMIT_MASK:OFF_COMMIT_MASK + NUM_COMMIT_GROUPS]
    kill_frac = batch[:, OFF_KILL_FRAC]
    kf_mask = batch[:, OFF_KILL_FRAC_MASK]
    retreat = batch[:, OFF_RETREAT]
    ret_mask = batch[:, OFF_RETREAT_MASK]

    (pred_value, pred_action, pred_move_frac, pred_disband, pred_recruit,
     pred_move, pred_btarget, pred_bselect, pred_commit,
     pred_kfrac, pred_retreat) = model(state)

    pred_value = pred_value.squeeze(1)

    # 1. Value head (MSE) — always learn, no policy weight
    v_loss = _active_mean((pred_value - value_target) ** 2, value_mask)

    # 2. Action type (masked cross-entropy, weighted; class-balanced when weights are given)
    action_weight = policy_weight
    if action_class_weight is not None:
        action_weight = policy_weight * action_class_weight[action_type.clamp(min=0)]
    a_loss = _ce_masked_weighted(pred_action, action_type, action_mask, action_active, action_weight)

    # 3-4. Move/disband levels per group (cross-entropy per group, weighted)
    mf_loss = _ce_levels_weighted(pred_move_frac, move_frac, move_frac_mask, policy_weight)
    df_loss = _ce_levels_weighted(pred_disband, disband_frac, disband_mask, policy_weight)

    # 5. Recruit level (cross-entropy, weighted)
    rf_loss = _ce_levels_weighted(pred_recruit, recruit_frac.unsqueeze(1), recruit_mask.unsqueeze(1), policy_weight)

    # 6-7. Categorical heads (masked cross-entropy, weighted)
    mv_loss = _ce_masked_weighted(pred_move, move_target, move_mask, move_active, policy_weight)
    bt_loss = _ce_masked_weighted(pred_btarget, battle_target, bt_mask, bt_active, policy_weight)

    # 8-11. Battle select (BCE), commit levels (cross-entropy per group), kill level (cross-entropy), retreat (BCE)
    bs_loss = _bce_weighted(pred_bselect, battle_select, bs_mask, policy_weight)
    cf_loss = _ce_levels_weighted(pred_commit, commit_frac, commit_mask, policy_weight)
    kf_loss = _ce_levels_weighted(pred_kfrac, kill_frac.unsqueeze(1), kf_mask.unsqueeze(1), policy_weight)
    rt_loss = _bce_weighted(pred_retreat, retreat, ret_mask, policy_weight)

    return [v_loss, a_loss, mf_loss, df_loss, rf_loss,
            mv_loss, bt_loss, bs_loss, cf_loss, kf_loss, rt_loss]


# ─── PPO objective (reinforcement phases) ───

def _policy_logp(pred, batch):
    """Log-probability of each sample's recorded action under the prediction, summed
    over the sampled heads (action type, move target, battle target, fraction
    levels per group), and the mask of samples with any such head active."""
    (_, pred_action, pred_move_frac, pred_disband, pred_recruit,
     pred_move, pred_btarget, _, pred_commit, pred_kfrac, _) = pred
    logp = torch.zeros(batch.shape[0], device=batch.device)
    active = torch.zeros_like(logp)

    categorical = (
        (pred_action, batch[:, OFF_ACTION_TYPE].long(), batch[:, OFF_ACTION_MASK:OFF_ACTION_MASK + NUM_ACTION_TYPES]),
        (pred_move, batch[:, OFF_MOVE_TARGET].long(), batch[:, OFF_MOVE_MASK:OFF_MOVE_MASK + MOVE_TARGET_DIM]),
        (pred_btarget, batch[:, OFF_BATTLE_TARGET].long(), batch[:, OFF_BATTLE_TARGET_MASK:OFF_BATTLE_TARGET_MASK + BATTLE_TARGET_DIM]),
    )
    for logits, target, legal_mask in categorical:
        act = (target >= 0).float()
        masked_logits = logits + (1 - legal_mask) * (-1e9)
        lp = F.log_softmax(masked_logits, dim=1).gather(1, target.clamp(min=0).unsqueeze(1)).squeeze(1)
        logp = logp + lp * act
        active = torch.maximum(active, act)

    fractions = (
        (pred_move_frac, batch[:, OFF_MOVE_FRAC:OFF_MOVE_FRAC + NUM_MOVE_GROUPS], batch[:, OFF_MOVE_FRAC_MASK:OFF_MOVE_FRAC_MASK + NUM_MOVE_GROUPS]),
        (pred_disband, batch[:, OFF_DISBAND_FRAC:OFF_DISBAND_FRAC + NUM_DISBAND_GROUPS], batch[:, OFF_DISBAND_MASK:OFF_DISBAND_MASK + NUM_DISBAND_GROUPS]),
        (pred_commit, batch[:, OFF_COMMIT_FRAC:OFF_COMMIT_FRAC + NUM_COMMIT_GROUPS], batch[:, OFF_COMMIT_MASK:OFF_COMMIT_MASK + NUM_COMMIT_GROUPS]),
        (pred_recruit, batch[:, OFF_RECRUIT_FRAC].unsqueeze(1), batch[:, OFF_RECRUIT_MASK].unsqueeze(1)),
        (pred_kfrac, batch[:, OFF_KILL_FRAC].unsqueeze(1), batch[:, OFF_KILL_FRAC_MASK].unsqueeze(1)),
    )
    for logits, frac, mask in fractions:
        groups = frac.shape[1]
        log_probs = F.log_softmax(logits.view(-1, groups, FRACTION_BINS), dim=2)
        levels = torch.round(frac * (FRACTION_BINS - 1)).long().clamp(0, FRACTION_BINS - 1)
        lp = (log_probs.gather(2, levels.unsqueeze(2)).squeeze(2) * mask).sum(dim=1)
        logp = logp + lp
        active = torch.maximum(active, (mask.sum(dim=1) > 0).float())

    return logp, active


class PPOContext:
    """Per-sample behavior log-probabilities under the starting weights (the
    dataset's generating model), the advantage standardization, and the value
    loss settings: its coefficient (the trunk is shared, the value fit competes
    with the surrogate) and whether it leaves the trunk to the policy."""

    def __init__(self, old_logp, active, adv_mean, adv_std, clip, value_coef, detach_value):
        self.old_logp = old_logp
        self.active = active
        self.adv_mean = adv_mean
        self.adv_std = adv_std
        self.clip = clip
        self.value_coef = value_coef
        self.detach_value = detach_value


def prepare_ppo(model, data, batch_size, device, clip, value_coef, detach_value):
    model.eval()
    n = data.shape[0]
    old_logp = np.zeros(n, dtype=np.float32)
    active = np.zeros(n, dtype=np.float32)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.from_numpy(data[i:min(i + batch_size, n)]).to(device)
            lp, act = _policy_logp(model(batch[:, OFF_STATE:OFF_STATE + STATE_SIZE]), batch)
            old_logp[i:i + len(lp)] = lp.cpu().numpy()
            active[i:i + len(act)] = act.cpu().numpy()
    adv = data[:, OFF_POLICY_WEIGHT][active > 0]
    adv_mean = float(adv.mean()) if len(adv) > 0 else 0.0
    adv_std = float(adv.std()) if len(adv) > 1 else 1.0
    print(f"ppo: {int(active.sum())} policy samples, advantage mean {adv_mean:.4f} std {adv_std:.4f}, clip {clip}, "
          f"value coef {value_coef}{', value head detached from the trunk' if detach_value else ''}", flush=True)
    return PPOContext(old_logp, active, adv_mean, max(adv_std, 1e-6), clip, value_coef, detach_value)


def _compute_ppo_losses(model, batch, idx, ppo):
    """[value, policy surrogate, battle select, retreat, clip fraction]; the last
    is a diagnostic, not a loss."""
    state = batch[:, OFF_STATE:OFF_STATE + STATE_SIZE]
    value_target = batch[:, OFF_VALUE]
    value_mask = (value_target >= 0).float()
    adv = (batch[:, OFF_POLICY_WEIGHT] - ppo.adv_mean) / ppo.adv_std
    old_logp = torch.from_numpy(ppo.old_logp[idx]).to(batch.device)

    pred = model(state, detach_value=ppo.detach_value)
    pred_value = pred[0].squeeze(1)
    v_loss = _active_mean((pred_value - value_target) ** 2, value_mask) * ppo.value_coef

    logp, active = _policy_logp(pred, batch)
    ratio = torch.exp(logp - old_logp)
    clipped = ratio.clamp(1 - ppo.clip, 1 + ppo.clip)
    surrogate = torch.minimum(ratio * adv, clipped * adv)
    pol_loss = -_active_mean(surrogate, active)
    clip_frac = _active_mean(((ratio - 1).abs() > ppo.clip).float(), active).detach()

    # The battle select and retreat heads are decoded by threshold, not sampled:
    # they keep the weighted BCE on above-average turns
    positive = adv.clamp(min=0)
    bs_loss = _bce_weighted(pred[7], batch[:, OFF_BATTLE_SELECT], batch[:, OFF_BATTLE_SELECT_MASK], positive)
    rt_loss = _bce_weighted(pred[10], batch[:, OFF_RETREAT], batch[:, OFF_RETREAT_MASK], positive)

    return [v_loss, pol_loss, bs_loss, rt_loss, clip_frac]


# ─── Epochs ───

def _losses(model, batch, idx, action_class_weight, ppo):
    if ppo is None:
        return _compute_losses(model, batch, action_class_weight)
    return _compute_ppo_losses(model, batch, idx, ppo)


def head_names(ppo):
    return PPO_HEAD_NAMES if ppo is not None else HEAD_NAMES


def _objective(losses, ppo):
    # The clip fraction is reported, not optimized
    return sum(losses[:4]) if ppo is not None else sum(losses)


def eval_loss(model, data, batch_size, device, action_class_weight, ppo):
    model.eval()
    n = data.shape[0]
    names = head_names(ppo)
    totals = [0.0] * len(names)
    num_batches = 0
    with torch.no_grad():
        for i in range(0, n, batch_size):
            idx = np.arange(i, min(i + batch_size, n))
            batch = torch.from_numpy(data[idx]).to(device)
            losses = _losses(model, batch, idx, action_class_weight, ppo)
            for j in range(len(names)):
                totals[j] += losses[j].item()
            num_batches += 1
    d = max(num_batches, 1)
    head_losses = [t / d for t in totals]
    return _objective(head_losses, ppo), head_losses


def train_epoch(model, optimizer, data, batch_size, device, action_class_weight, ppo):
    model.train()
    n = data.shape[0]
    perm = np.random.permutation(n)
    names = head_names(ppo)
    totals = [0.0] * len(names)
    num_batches = 0

    for i in range(0, n, batch_size):
        idx = perm[i:min(i + batch_size, n)]
        batch = torch.from_numpy(data[idx]).to(device)
        losses = _losses(model, batch, idx, action_class_weight, ppo)
        loss = _objective(losses, ppo)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        for j in range(len(names)):
            totals[j] += losses[j].item()
        num_batches += 1

    d = max(num_batches, 1)
    head_losses = [t / d for t in totals]
    return _objective(head_losses, ppo), head_losses

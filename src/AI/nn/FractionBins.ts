/**
 * Fraction levels: every fraction head (move, disband, commit, recruit, kill)
 * is categorical over FRACTION_BINS evenly spaced levels in [0, 1], one
 * softmax per unit group, so a level can be sampled for exploration and the
 * reinforcement phases train the head with cross-entropy on the level that
 * was played. Labels are stored as fractions; the trainer and the executor
 * map a fraction to its nearest level.
 */
export const FRACTION_BINS = 5; // 0, 0.25, 0.5, 0.75, 1

export function binToFraction(bin: number): number {
  return bin / (FRACTION_BINS - 1);
}

from blackjack_utils import get_score
import csv
DRAW_REWARD=0
LOSS_UNDER_12_REWARD=-2
def calculate_custom_reward(payoff, state,reward_config):
    hand = state['raw_obs'].get('player0 hand')
    if hand is None: return payoff

    score = get_score(hand)
    
    # 辞書から値を取得（なければデフォルト値）
    def get_val(key, default):
        return reward_config.get(key, default)

    # === 勝った場合 ===
    if payoff > 0:
        if score == 21:
            return get_val('win_21', 1.0)
        return get_val('win_normal', 1.0)

    # === 負けた場合 ===
    elif payoff < 0:
        if score > 21:
            return get_val('loss_burst', -1.0)
        elif score >= 17:
            return get_val('loss_17_plus', -0.25)
        elif score>=12:
            return get_val('loss_under_17', -1.0)
        else:
            return LOSS_UNDER_12_REWARD

    # === 引き分け ===
    return DRAW_REWARD
import csv
import os


# ★変更: 指定されたファイルパスから設定を読み込む
def load_reward_config(filepath):
    reward_config = {} # 初期化
    
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found. Using default values.")
        return

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # personality列はもう無いので、keyとvalueだけ取得
                key = row['key']
                val = float(row['value'])
                reward_config[key] = val
                
        print(f"Reward config loaded from {filepath}")
        print(f"Config: {reward_config}") # 確認用出力
        
    except Exception as e:
        print(f"Error loading config: {e}")
    return reward_config

def get_score(hand):
    score = 0
    aces = 0
    for card in hand:
        rank = card[1:]
        if rank == 'A':
            aces += 1
            score += 11
        elif rank in ['T', 'J', 'Q', 'K']:
            score += 10
        else:
            score += int(rank)
    while score > 21 and aces > 0:
        score -= 10
        aces -= 1
    return score

def decode_card(card_str):
    suit_map = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}
    suit = suit_map.get(card_str[0], card_str[0])
    rank = card_str[1:]
    return f"{suit}{rank}"

def print_hand(cards):
    return ", ".join([decode_card(c) for c in cards])

def get_action_name(action_id):
    return "Hit" if action_id == 0 else "Stand"
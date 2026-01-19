import rlcard
from rlcard.agents import DQNAgent
import torch
import os
import glob
import numpy as np

# --- GUIエラー回避用 ---
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from blackjack_utils import get_score

# ---------------------------------------------------------
# 設定
# ---------------------------------------------------------
# 学習済みモデルが保存されている場所
SAVE_DIR = 'experiments/blackjack_custom_reward'

# シミュレーション回数 (多いほど正確な表になります)
SIMULATION_GAMES = 50000 

# ---------------------------------------------------------
# 1. 環境とエージェントの初期化 (共通)
# ---------------------------------------------------------
env = rlcard.make('blackjack')
# 学習時と同じネットワーク構造
agent = DQNAgent(
    num_actions=env.num_actions,
    state_shape=env.state_shape[0],
    mlp_layers=[128, 128],
    device=torch.device("cpu")
)

# ---------------------------------------------------------
# 2. モデルファイルの検索
# ---------------------------------------------------------
search_pattern = os.path.join(SAVE_DIR, 'model_*.pth')
model_files = glob.glob(search_pattern)
model_files.sort()

if not model_files:
    print(f"Error: No model files found in {SAVE_DIR}")
    exit()

print(f"Found {len(model_files)} models. Starting visualization...\n")

# ---------------------------------------------------------
# 3. 描画用関数の定義
# ---------------------------------------------------------
def plot_strategy(matrix, title, filename, personality):
    # 行: プレイヤー (21〜12), 列: ディーラー (2〜A)
    player_range = range(21, 11, -1)
    dealer_range = range(2, 12)
    
    plt.figure(figsize=(10, 8))
    
    # 0(Hit):赤, 1(Stand):青
    cmap = sns.color_palette(["#ff9999", "#66b3ff"]) 
    
    ax = sns.heatmap(matrix, annot=True, fmt="d", cmap=cmap, cbar=False,
                     xticklabels=[str(i) if i<11 else 'A' for i in dealer_range],
                     yticklabels=[str(i) for i in player_range],
                     linewidths=.5, linecolor='gray')
    
    plt.title(f"{title} ({personality})", fontsize=16)
    plt.xlabel("Dealer's Up Card", fontsize=12)
    plt.ylabel("Player's Sum", fontsize=12)
    
    plt.text(0, -0.5, "0 = Hit (Red), 1 = Stand (Blue)", fontsize=10, color='black')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# ---------------------------------------------------------
# 4. モデルごとにループ処理
# ---------------------------------------------------------
for model_path in model_files:
    # ファイル名から性格名を取得
    file_name = os.path.basename(model_path)
    personality_name = file_name.replace('model_', '').replace('.pth', '')
    
    print(f"Processing: {personality_name} ...")

    # モデルのロード
    try:
        checkpoint = torch.load(model_path)
        agent.q_estimator.qnet.load_state_dict(checkpoint)
    except Exception as e:
        print(f"  Error loading {file_name}: {e}")
        continue

    # --- シミュレーション (データ収集) ---
    policy = {} # (player_sum, dealer_card, is_soft) -> action

    for _ in range(SIMULATION_GAMES):
        state, player_id = env.reset()
        while not env.is_over():
            obs = state['raw_obs']
            p_hand = obs['player0 hand']
            d_hand = obs['dealer hand']
            
            p_score = get_score(p_hand)
            
            # ディーラーカードの数値化
            d_card_str = d_hand[0][1:] 
            if d_card_str == 'A': d_score = 11
            elif d_card_str in ['T', 'J', 'Q', 'K']: d_score = 10
            else: d_score = int(d_card_str)

            # ソフトハンド判定
            has_ace = any(c.endswith('A') for c in p_hand)
            is_soft = False
            if has_ace:
                raw_sum = 0
                for c in p_hand:
                    r = c[1:]
                    if r in ['T', 'J', 'Q', 'K']: raw_sum += 10
                    elif r == 'A': raw_sum += 1
                    else: raw_sum += int(r)
                if raw_sum <= 11:
                    is_soft = True

            # AIの判断
            action, _ = agent.eval_step(state)
            
            if p_score <= 21:
                policy[(p_score, d_score, is_soft)] = action

            state, next_player_id = env.step(action, player_id)

    # --- マトリックス作成 ---
    player_range = range(21, 11, -1)
    dealer_range = range(2, 12)
    
    hard_matrix = []
    soft_matrix = []

    for p_val in player_range:
        hard_row = []
        soft_row = []
        for d_val in dealer_range:
            # データ不足時は Stand(1) をデフォルトとする
            act_h = policy.get((p_val, d_val, False), 1)
            hard_row.append(act_h)
            
            act_s = policy.get((p_val, d_val, True), 1)
            soft_row.append(act_s)
            
        hard_matrix.append(hard_row)
        soft_matrix.append(soft_row)

    # --- 保存 ---
    # ファイル名に性格名を含める
    hard_filename = f"result/strategy_hard_{personality_name}.png"
    soft_filename = f"result/strategy_soft_{personality_name}.png"
    
    plot_strategy(hard_matrix, "Hard Hand", hard_filename, personality_name)
    plot_strategy(soft_matrix, "Soft Hand", soft_filename, personality_name)
    
    print(f"  -> Saved {hard_filename} & {soft_filename}")

print("\nAll visualizations completed! 📊")
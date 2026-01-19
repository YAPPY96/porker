import rlcard
from rlcard.agents import DQNAgent
import torch
import os
import glob
import numpy as np
from PIL import Image

# --- GUIエラー回避用設定 ---
import matplotlib
matplotlib.use('Agg') # 画面表示せず描画するモード
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from blackjack_utils import get_score

# ---------------------------------------------------------
# 設定
# ---------------------------------------------------------
SAVE_DIR = 'experiments/blackjack_custom_reward'
OUTPUT_DIR = 'replays' # GIFの保存先
GAMES_TO_RECORD = 1 # 各性格につき何ゲーム録画するか

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ---------------------------------------------------------
# 描画用ヘルパー関数
# ---------------------------------------------------------
def draw_card(ax, x, y, card_str):
    """カード1枚を描画する関数"""
    # カードの枠
    rect = patches.Rectangle((x, y), 0.8, 1.2, linewidth=1, edgecolor='black', facecolor='white', zorder=2)
    ax.add_patch(rect)
    
    # スートと数字の変換
    if card_str == 'BACK':
        # 裏面
        pattern = patches.Rectangle((x+0.1, y+0.1), 0.6, 1.0, facecolor='firebrick', zorder=3)
        ax.add_patch(pattern)
        return

    suit_map = {'S': '♠', 'H': '♥', 'D': '♦', 'C': '♣'}
    color_map = {'S': 'black', 'H': 'red', 'D': 'red', 'C': 'black'}
    
    suit_char = card_str[0]
    rank_char = card_str[1:]
    
    suit = suit_map.get(suit_char, suit_char)
    color = color_map.get(suit_char, 'black')
    
    # 中央の文字
    ax.text(x + 0.4, y + 0.6, f"{rank_char}\n{suit}", fontsize=15, 
            ha='center', va='center', color=color, zorder=4)
    # 左上の文字
    ax.text(x + 0.1, y + 1.0, rank_char, fontsize=8, color=color, zorder=4)

def create_frame(player_hand, dealer_hand, action_text, result_text, score, personality):
    """現在の盤面を画像として生成する"""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor('#006400') # カジノっぽい緑色の背景
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 6)
    ax.axis('off') # 軸を消す

    # タイトル
    ax.text(3, 5.5, f"Agent: {personality}", fontsize=16, ha='center', color='white', fontweight='bold')
    
    # --- ディーラーの描画 (上段) ---
    ax.text(0.5, 4.5, "Dealer", fontsize=12, color='white')
    for i, card in enumerate(dealer_hand):
        draw_card(ax, 1.5 + i * 1.0, 3.5, card)
    
    # --- プレイヤーの描画 (下段) ---
    ax.text(0.5, 1.5, f"Player\nScore: {score}", fontsize=12, color='white')
    for i, card in enumerate(player_hand):
        draw_card(ax, 1.5 + i * 1.0, 0.5, card)

    # --- アクション/結果の表示 ---
    if result_text:
        # 結果が出ている場合
        box_color = 'gold' if "WIN" in result_text else 'gray'
        ax.text(3, 2.5, result_text, fontsize=24, color='blue', ha='center', 
                bbox=dict(facecolor=box_color, alpha=0.8))
    elif action_text:
        # 行動中の場合
        ax.text(3, 2.5, f"Action: {action_text}", fontsize=18, color='yellow', ha='center', fontweight='bold')

    # メモリ上の画像データに変換
    fig.canvas.draw()
    
    # 新しい書き方: buffer_rgba() を使って配列として取得
    # 自動的に (高さ, 幅, 4) の RGBA 配列になります
    image_array = np.asarray(fig.canvas.buffer_rgba())
    
    # PILは RGBA 配列をそのまま画像に変換できます
    plt.close(fig)
    return Image.fromarray(image_array)

# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------
# モデルを探す
model_files = glob.glob(os.path.join(SAVE_DIR, 'model_*.pth'))
model_files.sort()

env = rlcard.make('blackjack')
agent = DQNAgent(num_actions=env.num_actions, state_shape=env.state_shape[0], mlp_layers=[128, 128], device=torch.device("cpu"))

print(f"Generating replays for {len(model_files)} models...\n")

for model_path in model_files:
    personality = os.path.basename(model_path).replace('model_', '').replace('.pth', '')
    print(f"Creating replay for: {personality}")

    # モデルロード
    agent.q_estimator.qnet.load_state_dict(torch.load(model_path))
    
    frames = []
    
    for _ in range(GAMES_TO_RECORD):
        state, player_id = env.reset()
        
        # ゲーム開始時の状態
        raw_obs = state['raw_obs']
        p_hand = raw_obs['player0 hand']
        d_hand = raw_obs['dealer hand'] # ここでは1枚しか見えてない想定
        
        # ディーラーの手札表示ロジック（最初は1枚＋裏面）
        display_d_hand = [d_hand[0], 'BACK'] 
        
        # フレーム1: 配られた直後
        frames.append(create_frame(p_hand, display_d_hand, "Thinking...", None, get_score(p_hand), personality))
        
        done = False
        while not env.is_over():
            action, _ = agent.eval_step(state)
            act_str = "Hit" if action == 0 else "Stand"
            
            # フレーム2: 決断
            frames.append(create_frame(p_hand, display_d_hand, act_str, None, get_score(p_hand), personality))
            
            state, next_player_id = env.step(action, player_id)
            
            # 状態更新
            raw_obs = state['raw_obs']
            p_hand = raw_obs['player0 hand']
            
            # Hitした場合、カードが増えた状態を表示
            if action == 0 and not env.is_over():
                frames.append(create_frame(p_hand, display_d_hand, "Hit!", None, get_score(p_hand), personality))

        # --- 結果表示 ---
        raw_obs = state['raw_obs']
        p_hand = raw_obs['player0 hand']
        d_hand = raw_obs['dealer hand'] # 全て公開
        
        payoffs = env.get_payoffs()
        score = payoffs[player_id]
        
        res_text = "WIN 🏆" if score > 0 else ("LOSE 💀" if score < 0 else "DRAW 🤝")
        
        # フレーム3: 最終結果（ディーラーの手札オープン）
        # 最後の余韻のために同じフレームを数枚追加
        end_frame = create_frame(p_hand, d_hand, None, res_text, get_score(p_hand), personality)
        for _ in range(5):
            frames.append(end_frame)

    # GIF保存
    gif_path = os.path.join(OUTPUT_DIR, f'replay_{personality}.gif')
    # duration=800 は 0.8秒ごとにコマ送り
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], optimize=False, duration=800, loop=0)
    print(f"  -> Saved: {gif_path}")

print(f"\nAll replays saved in '{OUTPUT_DIR}' folder! 🎥")
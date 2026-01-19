import rlcard
from rlcard.agents import DQNAgent
import torch
import os
import glob
from blackjack_utils import get_score, print_hand, decode_card, get_action_name

# ---------------------------------------------------------
# 設定
# ---------------------------------------------------------
SAVE_DIR = 'experiments/blackjack_custom_reward' # モデルがある場所
LOG_DIR = 'logs'                                 # ログ保存先
GAMES_PER_MODEL = 5                              # 記録するゲーム数

# ログ保存用フォルダを作成
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# ---------------------------------------------------------
# 1. 準備
# ---------------------------------------------------------
env = rlcard.make('blackjack')
agent = DQNAgent(
    num_actions=env.num_actions,
    state_shape=env.state_shape[0],
    mlp_layers=[128, 128],
    device=torch.device("cpu")
)
env.set_agents([agent])

# モデルファイルを探す
model_files = glob.glob(os.path.join(SAVE_DIR, 'model_*.pth'))
model_files.sort()

if not model_files:
    print(f"Error: No models found in {SAVE_DIR}")
    exit()

print(f"Found {len(model_files)} models. Saving logs to '{LOG_DIR}/'...\n")

# ---------------------------------------------------------
# 2. モデルごとにログ保存しながら実行
# ---------------------------------------------------------
for model_path in model_files:
    # ファイル名から性格名を取得
    personality = os.path.basename(model_path).replace('model_', '').replace('.pth', '')
    
    # 保存するログファイルのパス
    log_file_path = os.path.join(LOG_DIR, f"log_{personality}.txt")
    
    print(f"Processing {personality}... (Saving to {log_file_path})")

    # モデルのロード
    try:
        agent.q_estimator.qnet.load_state_dict(torch.load(model_path))
    except Exception as e:
        print(f"  Load Error: {e}")
        continue

    # ファイルを開いて書き込む準備
    # encoding='utf-8' にすることで絵文字（🏆など）の文字化けを防ぎます
    with open(log_file_path, 'w', encoding='utf-8') as f:
        
        # 画面出力とファイル書き込みを同時に行うヘルパー関数
        def log(text):
            # print(text) # 画面にも出したい場合はコメントアウトを外す
            f.write(text + "\n")

        # ヘッダー書き込み
        log("="*60)
        log(f" 【 エージェント性格: {personality.upper()} 】")
        log("="*60)

        # ゲーム実行ループ
        for i in range(GAMES_PER_MODEL):
            log(f"\n--- Game {i+1} ---")
            state, player_id = env.reset()
            
            step_count = 1
            while not env.is_over():
                raw_obs = state['raw_obs']
                p_hand = raw_obs['player0 hand']
                d_hand = raw_obs['dealer hand']
                
                p_score = get_score(p_hand)
                d_up_card = decode_card(d_hand[0]) if d_hand else "?"

                # AIの決断
                action, _ = agent.eval_step(state)
                act_str = get_action_name(action)

                # ログ記録
                log(f"  Step {step_count}:")
                log(f"    Player: {print_hand(p_hand)} (Score: {p_score})")
                log(f"    Dealer: {d_up_card} (Hidden)")
                log(f"    -> Action: {act_str}")

                state, next_player_id = env.step(action, player_id)
                step_count += 1

            # --- 最終結果 ---
            final_obs = state['raw_obs']
            p_final = final_obs['player0 hand']
            d_final = final_obs['dealer hand']
            
            p_final_score = get_score(p_final)
            d_final_score = get_score(d_final)
            
            payoffs = env.get_payoffs()
            score = payoffs[player_id]

            if score > 0:
                result = "WIN 🏆"
            elif score < 0:
                result = "LOSE 💀"
            else:
                result = "DRAW 🤝"

            log(f"  [Result] {result}")
            log(f"    Player Final: {print_hand(p_final)} (Score: {p_final_score})")
            log(f"    Dealer Final: {print_hand(d_final)} (Score: {d_final_score})")

print("\nAll logs saved successfully! Check the 'logs' folder.")
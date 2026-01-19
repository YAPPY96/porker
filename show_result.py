import rlcard
from rlcard.agents import DQNAgent
import torch
import os
import glob
from blackjack_utils import get_score, print_hand, decode_card, get_action_name

# ---------------------------------------------------------
# 設定
# ---------------------------------------------------------
SAVE_DIR = 'experiments/blackjack_custom_reward'
NUM_GAMES = 1000 # テストするゲーム数
SHOW_LOGS = False # Trueなら1戦ごとのログを表示、Falseなら結果だけ表示(推奨)

# ---------------------------------------------------------
# 1. 環境とエージェントの準備 (共通)
# ---------------------------------------------------------
env = rlcard.make('blackjack')

# 学習時と同じネットワーク構造 [128, 128] に合わせる
agent = DQNAgent(
    num_actions=env.num_actions,
    state_shape=env.state_shape[0],
    mlp_layers=[128, 128], 
    device=torch.device("cpu")
)
env.set_agents([agent])

# ---------------------------------------------------------
# 2. モデルファイルを探す
# ---------------------------------------------------------
# experimentsフォルダ内の model_*.pth をすべて取得
model_files = glob.glob(os.path.join(SAVE_DIR, 'model_*.pth'))
model_files.sort()

if not model_files:
    print(f"Error: Model files not found in {SAVE_DIR}")
    exit()

# 結果比較用のリスト
summary_results = []

# ---------------------------------------------------------
# 3. モデルごとにテスト実行
# ---------------------------------------------------------
print(f"\nFound {len(model_files)} models. Starting evaluation...\n")

for model_path in model_files:
    # ファイル名から性格名を取得 (例: model_aggressive.pth -> aggressive)
    personality_name = os.path.basename(model_path).replace('model_', '').replace('.pth', '')
    
    print(f"==========================================")
    print(f" Testing Model: {personality_name}")
    print(f"==========================================")

    # モデルのロード
    try:
        checkpoint = torch.load(model_path)
        agent.q_estimator.qnet.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Error loading {personality_name}: {e}")
        continue

    # --- ゲーム実行 ---
    total_win = 0
    total_draw = 0
    total_lose = 0

    for i in range(NUM_GAMES):
        state, player_id = env.reset()
        
        step_count = 1
        # ログ表示がONの場合のみ詳細を表示
        if SHOW_LOGS:
            print(f"--- Game {i+1} ---")

        while not env.is_over():
            action, _ = agent.eval_step(state)
            
            # ログ表示
            if SHOW_LOGS:
                raw_obs = state['raw_obs']
                p_hand = raw_obs['player0 hand']
                d_hand = raw_obs['dealer hand']
                print(f"  Hand: {print_hand(p_hand)} ({get_score(p_hand)}) | AI: {get_action_name(action)}")

            state, next_player_id = env.step(action, player_id)
            step_count += 1

        # 結果判定
        payoffs = env.get_payoffs()
        result_score = payoffs[player_id]
        
        if result_score > 0:
            total_win += 1
            outcome = "WIN"
        elif result_score < 0:
            total_lose += 1
            outcome = "LOSE"
        else:
            total_draw += 1
            outcome = "DRAW"

        if SHOW_LOGS:
            final_obs = state['raw_obs']
            print(f"  Result: {outcome} (Player: {get_score(final_obs['player0 hand'])}, Dealer: {get_score(final_obs['dealer hand'])})\n")

    # --- 勝率計算 ---
    # 引き分けを除いた勝率 (Win / (Win + Lose))
    decisive_games = NUM_GAMES - total_draw
    if decisive_games > 0:
        win_rate = total_win / decisive_games
    else:
        win_rate = 0.0
    
    print(f"  Results ({NUM_GAMES} games):")
    print(f"    WIN : {total_win}")
    print(f"    LOSE: {total_lose}")
    print(f"    DRAW: {total_draw}")
    print(f"    Win Rate (excl. draws): {win_rate:.2%}")
    print("\n")
    
    # 結果を保存
    summary_results.append({
        'name': personality_name,
        'rate': win_rate,
        'win': total_win,
        'lose': total_lose,
        'draw': total_draw
    })

# ---------------------------------------------------------
# 4. 最終ランキング表示
# ---------------------------------------------------------
# 勝率が高い順にソート
summary_results.sort(key=lambda x: x['rate'], reverse=True)

print("##########################################")
print(" FINAL RANKING (Win Rate excl. draws)")
print("##########################################")
print(f"{'Rank':<5} {'Personality':<15} {'Rate':<10} {'W-L-D':<10}")
print("-" * 45)

for rank, res in enumerate(summary_results, 1):
    print(f"{rank:<5} {res['name']:<15} {res['rate']:.2%}     {res['win']}-{res['lose']}-{res['draw']}")
print("##########################################")
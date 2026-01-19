import rlcard
from rlcard.agents import DQNAgent
from rlcard.utils import tournament
import torch
import os
import csv
from blackjack_utils import load_reward_config
from custom_reward import  calculate_custom_reward


def train_and_save(config_path,target_personality):
    # 1. 対応するCSVファイルを読み込む
    reward_config=load_reward_config(config_path)

    save_dir = 'experiments/blackjack_custom_reward'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    log_path = os.path.join(save_dir, f'performance_{target_personality}.csv')
    model_save_name = f'model_{target_personality}.pth'

    # 2. 環境設定
    env = rlcard.make('blackjack', config={'seed': 42})
    eval_env = rlcard.make('blackjack', config={'seed': 42})

    # 3. エージェント設定
    agent = DQNAgent(
        num_actions=env.num_actions,
        state_shape=env.state_shape[0],
        mlp_layers=[128, 128], 
        device=torch.device("cpu")
    )

    env.set_agents([agent])
    eval_env.set_agents([agent])

    # 4. 学習ループ
    num_episodes = 50000 
    evaluate_every = 500
    print(f"Start training ({target_personality}) using {config_path}...")

    for episode in range(num_episodes):
        state, player_id = env.reset()
        trajectory = [] 

        while not env.is_over():
            action = agent.step(state)
            next_state, next_player_id = env.step(action, player_id)
            trajectory.append((state, action))
            state = next_state

        payoffs = env.get_payoffs()
        original_payoff = payoffs[player_id]
        
        # ★変更: 引数から personality を削除 (ロード済みデータを使うため)
        custom_reward = calculate_custom_reward(original_payoff, state,reward_config)

        for i, (s, a) in enumerate(trajectory):
            done = (i == len(trajectory) - 1)
            next_s = trajectory[i+1][0] if not done else state
            agent.feed((s, a, custom_reward, next_s, done))

        if episode % evaluate_every == 0:
            result = tournament(eval_env, 100)[0]
            print(f'Episode: {episode}, Win Rate: {result:.4f}')
            
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode, result])

    final_save_path = os.path.join(save_dir, model_save_name)
    torch.save(agent.q_estimator.qnet.state_dict(), final_save_path)
    print(f"Training finished. Model saved to {final_save_path}")
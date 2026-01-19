import rlcard
from rlcard.agents import DQNAgent
from rlcard.utils import tournament
import torch
import os
import csv
import glob
from train_and_save import train_and_save

# --- 設定 ---
# configファイルが入っているフォルダ
CONFIG_DIR = 'personality' 
# 保存先フォルダ
SAVE_DIR = 'experiments/blackjack_custom_reward'

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ---------------------------------------------------------
# 1. personalityフォルダから設定ファイルを全取得
# ---------------------------------------------------------
search_pattern = os.path.join(CONFIG_DIR, 'config_*.csv')
config_files = glob.glob(search_pattern)
config_files.sort()

if not config_files:
    print(f"エラー: '{CONFIG_DIR}' フォルダに config_*.csv が見つかりません。")
    exit()

# ---------------------------------------------------------
# 2. ファイルごとにループして学習実行
# ---------------------------------------------------------
for config_path in config_files:
    
    # ファイル名から性格名を取得 (例: personality/config_aggressive.csv -> aggressive)
    file_name = os.path.basename(config_path)
    target_personality = file_name.replace('config_', '').replace('.csv', '')
    
    print(f"\n==================================================")
    print(f" START TRAINING: {target_personality}")
    print(f" Config File   : {config_path}")
    print(f"==================================================")

    train_and_save(config_path,target_personality)
import os
import pickle
from datasets import load_dataset, Dataset, concatenate_datasets
import numpy as np
from tqdm import tqdm

categories = np.array([1550, 1750, 1950])

def process_pgns(dataset, categories):
    current_sequence = ""
    max_length = 768  
    for row in dataset["train"]:
        game = row['transcript']
        white_elo = row['WhiteElo']
        cat = np.digitize(white_elo, categories, right=False)
        result = row['Result']
        full_game = f";{cat}#{result}#{game}"
        if len(current_sequence) <= max_length:
            current_sequence += full_game
        else:
            yield {"transcript": current_sequence[:max_length]}
            current_sequence = ""

dataset1 = load_dataset("adamkarvonen/chess_games", data_files="lichess_9gb.zip")
dataset1 = Dataset.from_generator(lambda : process_pgns(dataset1))

dataset2 = load_dataset("adamkarvonen/chess_games", data_files="lichess_6gb.zip")
dataset2 = Dataset.from_generator(lambda : process_pgns(dataset2))

dataset = concatenate_datasets([dataset1, dataset2])
dataset = dataset.shuffle(seed=42)

all_chars = set()
for i, d in tqdm(enumerate(dataset["transcript"])):
    for c in d:
        all_chars.add(c)
    if i>=1000000:
        break

all_chars = sorted(list(all_chars))
all_chars.append("[PAD]")
vocab_size = len(all_chars)

stoi = { ch:i for i,ch in enumerate(all_chars) }
itos = { i:ch for i,ch in enumerate(all_chars) }

meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

dataset = dataset.train_test_split(test_size=0.002)
dataset.to_parquet('chess_games.parquet')
import pickle
from datasets import load_dataset, Dataset, concatenate_datasets
import numpy as np
from tqdm import tqdm

def process_pgns(dataset):
    current_sequence = ""
    max_length = 768  
    for row in dataset["train"]:
        game = row['transcript']
        result = row['Result']
        full_game = f";{result}#{game}"
        if len(current_sequence) <= max_length:
            current_sequence += full_game
        else:
            yield {"transcript": current_sequence[:max_length]}
            current_sequence = ""

dataset = load_dataset("adamkarvonen/chess_games", data_files="stockfish_dataset.zip")
dataset = Dataset.from_generator(lambda : process_pgns(dataset))

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
with open('meta.pkl', 'wb') as f:
    pickle.dump(meta, f)

dataset = dataset.train_test_split(test_size=0.002)
dataset.to_parquet('chess_games_stockfish_balanced.parquet')
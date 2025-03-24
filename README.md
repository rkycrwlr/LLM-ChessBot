# LLM-ChessBot

This repository contains the code for training a language model on chess games and using it to generate chess moves. The model is trained on a dataset of chess games from Lichess, and the code includes scripts for preparing the data, training the model, and generating moves. The model is based on the [NanoGPT](https://github.com/karpathy/nanoGPT) implementation of Andrej Karpathy.

## Preparing the Data

The data preparation script `src/prepare_data.py` downloads the Stockfish chess games dataset, processes the games to create sequences of moves, and saves the processed data to a Parquet file. The script also creates a vocabulary of all the unique characters in the dataset and saves it to a pickle file.

## Training the Model

The model training script `src/training.py` loads the processed data and trains a nanoGPT model on the data.

```python
python src/training.py --config config.json --dataset chess_games.parquet --meta meta.pkl
```

## Download the model weights

```bash
mkdir -p models/stockfish-16 && wget -O models/stockfish-16/model.safetensors https://nextcloud.rky.ovh/public.php/dav/files/mxX5WF75amkLAPy/?accept=zip
```

## Testing the model against Stockfish

The `chess_fight.ipynb` notebook contains the code for testing the model against Stockfish. You can install the stcokfish engine using the following command: `sudo apt install stockfish`. Don't forget to give the path to stockfish in the notebook.

## Running experiments

The `experiments` folder contains the code for running multiple experiments. You can evaluate the ELO of the LLM chessbot or interpret the internal representations of the model for ELO classification and board position extraction.

import os
import json 
import argparse

from transformers import get_scheduler

import torch
from torch.utils.data.dataloader import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW

from datasets import Dataset
from accelerate import Accelerator

from model import GPT
from utils import CfgNode as CN

from tqdm import tqdm
import pickle


parser = argparse.ArgumentParser(description='Train GPT')
parser.add_argument('--conf', default='config.json')
parser.add_argument('--meta', default='meta.pkl')
parser.add_argument('--dataset', default='chess_games.parquet')
args = parser.parse_args()

C = CN()

with open(args.conf) as f:
    config = json.load(f)

with open(args.meta, "rb") as f:
    meta = pickle.load(f)

C.n_layer = config["n_layer"]
C.n_embd = config["n_embd"]
C.n_head = config["n_head"]
C.embd_pdrop = config["embd_pdrop"]
C.resid_pdrop = config["resid_pdrop"]
C.attn_pdrop = config["attn_pdrop"]
C.block_size = config["block_size"]
C.vocab_size = meta["vocab_size"]

itos = meta["itos"]
stoi = meta["stoi"]

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

def tokenize(element, context_length):
    input_batch = []
    for elt in element["transcript"]:
        try:
            encoded_txt = torch.tensor(encode(elt), dtype=torch.int64)[:context_length]
        except:
            print(f"Error with {elt}")
            continue
        out = torch.ones(context_length, dtype=torch.int64) * stoi["[PAD]"]
        out[:len(encoded_txt)] = encoded_txt
        input_batch.append(out)

    return {"input_ids": input_batch}

def get_grouped_params(model, weight_decay, no_decay=["bias", "LayerNorm.weight"]):
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {"params": params_with_wd, "weight_decay": weight_decay},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]

def ce_loss(labels, logits, loss_function=CrossEntropyLoss()):
    labels = labels[...,1:].contiguous()
    shift_logits = logits[...,:-1,:].contiguous()
    return loss_function(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

def evaluate():
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            x = batch['input_ids']
            if type(x) == list:
                x = torch.stack(x, dim=1)
            outputs, _ = model(x)
            loss = ce_loss(x, outputs)

        losses.append(loss)
    loss = torch.mean(torch.stack(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")
    return loss.item(), perplexity.item()

def log(text, log="log.txt"):
    print(text)
    with open(log, 'a+') as f:
        f.write(text + '\n')

dataset = Dataset.from_parquet(args.dataset)

tokenized_dataset = dataset.map(
    lambda x : tokenize(x, C.block_size),
    batched=True,
    remove_columns=dataset["train"].column_names,
    num_proc=16
)
tokenized_dataset.set_format(type='torch')

model = GPT(C)

train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=config['batch_size'], num_workers=1)
eval_dataloader = DataLoader(tokenized_dataset["test"], batch_size=config['batch_size'])

weight_decay = config['weight_decay']
optimizer = AdamW(get_grouped_params(model, weight_decay), lr=config['learning_rate'])

accelerator = Accelerator(mixed_precision="fp16")

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(model, optimizer, train_dataloader, eval_dataloader)

num_train_epochs = 1
num_training_steps = len(train_dataloader) * num_train_epochs

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=1,
    num_training_steps=num_training_steps
)

gradient_accumulation_steps = config['gradient_accumulation_steps']
eval_steps = config['eval_steps']
save_steps = config['save_steps']
samples_per_step = config['batch_size']
output_dir = config['output_dir']

model.train()

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)


for epoch in range(num_train_epochs):
    for step, batch in tqdm(enumerate(train_dataloader, start=1), total=len(train_dataloader), desc="Training"):
        with accelerator.autocast():
            x = batch['input_ids']
            if type(x) == list:
                x = torch.stack(x, dim=1)

            logits, _ = model(x)
            loss = ce_loss(x, logits)
            loss = loss / gradient_accumulation_steps
            accelerator.backward(loss)

            if step % config["log_step"] == 0:
                log(f"Train : Step {step}: Loss {loss.item():.4f}")

            if step % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if step % (eval_steps * gradient_accumulation_steps) == 0:
                loss, perplexity = evaluate()
                log(f"Eval : Step {step}: Loss {loss:.4f}, Perplexity {perplexity:.2f}")
                model.train()
                accelerator.wait_for_everyone()

            if (step % (save_steps * gradient_accumulation_steps) == 0):
                accelerator.wait_for_everyone()
                accelerator.save_model(model, f'{output_dir}/{step}.pt')

accelerator.wait_for_everyone()
accelerator.save_model(model, f'{output_dir}/final.pt')


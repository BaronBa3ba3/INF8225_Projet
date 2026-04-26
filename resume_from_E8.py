#!/usr/bin/env python3
"""
Resume 2055734_2062203_Projet.ipynb from E8 onward without rerunning E1-E7.

Run from the same folder that contains your notebook/results folder:
    python resume_from_E8.py

What this script does:
  - rebuilds the dataset, vocabularies, model classes, and training helpers needed for E8+
  - loads saved results from E1-E7 in ./results when they exist
  - runs E8, E9, E10, E11 only if their result files are not already present
  - creates results/summary_table.csv and figures/*.png/*.pdf, then zips the figures

Important: a fresh .py process cannot recover variables that only lived in the stopped
notebook kernel. It can avoid retraining E1-E7 only if their results/checkpoints were
saved to disk. Your notebook already saves E1/E3/E4/E5/E6/E7 results in ./results.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pickle
import shutil
import sys
import time
import zipfile
from collections import Counter, defaultdict
from itertools import takewhile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

try:
    import einops
    from einops import rearrange, repeat
except ImportError as exc:
    raise SystemExit("Missing dependency: einops. Install it with: pip install einops") from exc

try:
    import spacy
except ImportError as exc:
    raise SystemExit("Missing dependency: spacy. Install it with: pip install spacy") from exc

try:
    import sacrebleu
except ImportError as exc:
    raise SystemExit("Missing dependency: sacrebleu. Install it with: pip install sacrebleu") from exc

# Avoid accidental online wandb logging. The training function below does not require wandb.
os.environ.setdefault("WANDB_MODE", "disabled")

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data and tokenization
# ---------------------------------------------------------------------------
SPECIALS = ["<unk>", "<pad>", "<bos>", "<eos>"]


def ensure_spacy_model(model_name: str):
    """Load a spaCy model, with a clear message if it is missing."""
    try:
        return spacy.load(model_name)
    except OSError as exc:
        raise SystemExit(
            f"Missing spaCy model '{model_name}'. Install it with:\n"
            f"    python -m spacy download {model_name}"
        ) from exc


def ensure_manythings_dataset():
    """Download/extract fra.txt if it is not already in the working folder."""
    fra_txt = ROOT / "fra.txt"
    if fra_txt.exists():
        return

    zip_path = ROOT / "fra-eng.zip"
    if not zip_path.exists():
        print("fra.txt not found; downloading fra-eng.zip...")
        urlretrieve("http://www.manythings.org/anki/fra-eng.zip", zip_path)

    print("Extracting fra-eng.zip...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(ROOT)

    if not fra_txt.exists():
        raise FileNotFoundError("Expected fra.txt after extracting fra-eng.zip, but it was not found.")


class Vocab:
    """Simple vocabulary class for token-to-index mapping."""

    def __init__(self, token_to_idx, default_index=None):
        self.token_to_idx = token_to_idx
        self.idx_to_token = {idx: token for token, idx in token_to_idx.items()}
        self.default_index = default_index

    def __len__(self):
        return len(self.token_to_idx)

    def __getitem__(self, token):
        if self.default_index is not None:
            return self.token_to_idx.get(token, self.default_index)
        return self.token_to_idx[token]

    def __call__(self, tokens):
        return [self[token] for token in tokens]

    def set_default_index(self, default_index):
        self.default_index = default_index

    def lookup_token(self, idx):
        return self.idx_to_token.get(int(idx), "<unk>")

    def lookup_tokens(self, indices):
        return [self.lookup_token(idx) for idx in indices]


def build_vocab_from_iterator(iterator, min_freq=1, specials=None):
    counter = Counter()
    for tokens in iterator:
        counter.update(tokens)

    token_to_idx = {}
    if specials:
        for token in specials:
            token_to_idx[token] = len(token_to_idx)

    for token, freq in counter.items():
        if freq >= min_freq and token not in token_to_idx:
            token_to_idx[token] = len(token_to_idx)
    return Vocab(token_to_idx)


class TranslationDataset(Dataset):
    def __init__(self, dataset, en_vocab, fr_vocab, en_tokenizer, fr_tokenizer):
        super().__init__()
        self.dataset = dataset
        self.en_vocab = en_vocab
        self.fr_vocab = fr_vocab
        self.en_tokenizer = en_tokenizer
        self.fr_tokenizer = fr_tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        en_sentence, fr_sentence = self.dataset[index]
        en_tokens = ["<bos>"] + self.en_tokenizer(en_sentence) + ["<eos>"]
        fr_tokens = ["<bos>"] + self.fr_tokenizer(fr_sentence) + ["<eos>"]
        return torch.LongTensor(self.en_vocab(en_tokens)), torch.LongTensor(self.fr_vocab(fr_tokens))


def yield_tokens(dataset, tokenizer, lang):
    sentence_idx = 0 if lang == "en" else 1
    for sentences in dataset:
        yield tokenizer(sentences[sentence_idx])


def build_vocab(dataset, en_tokenizer, fr_tokenizer, min_freq):
    en_vocab = build_vocab_from_iterator(yield_tokens(dataset, en_tokenizer, "en"), min_freq, SPECIALS)
    en_vocab.set_default_index(en_vocab["<unk>"])
    fr_vocab = build_vocab_from_iterator(yield_tokens(dataset, fr_tokenizer, "fr"), min_freq, SPECIALS)
    fr_vocab.set_default_index(fr_vocab["<unk>"])
    return en_vocab, fr_vocab


def preprocess(dataset, en_tokenizer, fr_tokenizer, max_words):
    filtered = []
    for en_s, fr_s in dataset:
        if len(en_tokenizer(en_s)) >= max_words or len(fr_tokenizer(fr_s)) >= max_words:
            continue
        filtered.append((en_s.replace("\n", ""), fr_s.replace("\n", "")))
    return filtered


def build_datasets(max_sequence_length, min_token_freq, en_tokenizer, fr_tokenizer, train, val):
    datasets = [preprocess(samples, en_tokenizer, fr_tokenizer, max_sequence_length) for samples in [train, val]]
    en_vocab, fr_vocab = build_vocab(datasets[0], en_tokenizer, fr_tokenizer, min_token_freq)
    return [TranslationDataset(samples, en_vocab, fr_vocab, en_tokenizer, fr_tokenizer) for samples in datasets]


def generate_batch(data_batch, src_pad_idx, tgt_pad_idx):
    en_batch, fr_batch = [], []
    for en_tokens, fr_tokens in data_batch:
        en_batch.append(en_tokens)
        fr_batch.append(fr_tokens)
    en_batch = pad_sequence(en_batch, padding_value=src_pad_idx, batch_first=True)
    fr_batch = pad_sequence(fr_batch, padding_value=tgt_pad_idx, batch_first=True)
    return en_batch, fr_batch


def load_translation_data(dataset_fraction=0.33, max_seq_len=60, min_tok_freq=2):
    ensure_manythings_dataset()
    df = pd.read_csv(ROOT / "fra.txt", sep="\t", names=["english", "french", "attribution"])
    train_pairs = [(en, fr) for en, fr in zip(df["english"], df["french"])]
    train_pairs, valid_pairs = train_test_split(train_pairs, test_size=0.1, random_state=0)
    train_pairs = train_pairs[: int(len(train_pairs) * dataset_fraction)]
    print(f"Using reduced training set: {len(train_pairs)} examples ({dataset_fraction * 100:.1f}%)")

    en_nlp = ensure_spacy_model("en_core_web_sm")
    fr_nlp = ensure_spacy_model("fr_core_news_sm")

    def en_tokenizer(text):
        return [token.text for token in en_nlp.tokenizer(text)]

    def fr_tokenizer(text):
        return [token.text for token in fr_nlp.tokenizer(text)]

    train_dataset, val_dataset = build_datasets(
        max_seq_len, min_tok_freq, en_tokenizer, fr_tokenizer, train_pairs, valid_pairs
    )
    print(f"English vocabulary size: {len(train_dataset.en_vocab):,}")
    print(f"French vocabulary size:  {len(train_dataset.fr_vocab):,}")
    print(f"Training examples:       {len(train_dataset):,}")
    print(f"Validation examples:     {len(val_dataset):,}")
    return train_dataset, val_dataset, en_tokenizer, fr_tokenizer


# ---------------------------------------------------------------------------
# Mamba model definitions
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SelectiveSSM(nn.Module):
    def __init__(self, d_inner: int, d_state: int = 16, dt_rank: int | None = None, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        if dt_rank is None:
            dt_rank = max(1, d_inner // 32)
        self.dt_rank = dt_rank

        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        dt_init_std = dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float), "n -> d n", d=d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x):
        batch, sl, _ = x.shape
        A = -torch.exp(self.A_log)
        x_dbl = self.x_proj(x)
        delta_raw, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta_raw))
        delta_A = torch.exp(torch.einsum("bld,dn->bldn", delta, A))
        delta_B_x = torch.einsum("bld,bln,bld->bldn", delta, B, x)

        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(sl):
            h = delta_A[:, t] * h + delta_B_x[:, t]
            y_t = torch.einsum("bdn,bn->bd", h, C[:, t])
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        return y + x * self.D


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_conv = d_conv
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True,
        )
        self.ssm = SelectiveSSM(d_inner=self.d_inner, d_state=d_state)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        sl = x.size(1)
        xz = self.in_proj(x)
        x_ssm, z = xz.chunk(2, dim=-1)
        x_conv = rearrange(x_ssm, "b l d -> b d l")
        x_conv = self.conv1d(x_conv)[:, :, :sl]
        x_conv = rearrange(x_conv, "b d l -> b l d")
        x_conv = F.silu(x_conv)
        y = self.ssm(x_conv)
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaStack(nn.Module):
    def __init__(self, d_model: int, n_layers: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand) for _ in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.final_norm = RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for block, norm in zip(self.layers, self.norms):
            x = x + self.dropout(block(norm(x)))
        return self.final_norm(x)


class TranslationMamba(nn.Module):
    def __init__(
        self,
        n_tokens_src: int,
        n_tokens_tgt: int,
        dim_embedding: int,
        n_layers: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        src_pad_idx: int = 0,
        tgt_pad_idx: int = 0,
    ):
        super().__init__()
        self.dim_embedding = dim_embedding
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.embedding_src = nn.Embedding(n_tokens_src, dim_embedding, padding_idx=src_pad_idx)
        self.embedding_tgt = nn.Embedding(n_tokens_tgt, dim_embedding, padding_idx=tgt_pad_idx)
        self.emb_dropout = nn.Dropout(dropout)
        self.mamba = MambaStack(dim_embedding, n_layers, d_state, d_conv, expand, dropout)
        self.out_layer = nn.Linear(dim_embedding, n_tokens_tgt)

    def forward(self, source, target):
        src_emb = self.embedding_src(source) * math.sqrt(self.dim_embedding)
        tgt_emb = self.embedding_tgt(target) * math.sqrt(self.dim_embedding)
        src_emb = self.emb_dropout(src_emb)
        tgt_emb = self.emb_dropout(tgt_emb)
        full = torch.cat([src_emb, tgt_emb], dim=1)
        out = self.mamba(full)
        tgt_out = out[:, source.size(1):, :]
        return self.out_layer(tgt_out)


class MambaLMForBenchmark(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.body = MambaStack(d_model=d_model, n_layers=n_layers, dropout=0.0)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return self.head(self.body(self.embed(x)))


class TinyTransformerLM(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, n_layers=4, n_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.body = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        sl = x.size(1)
        mask = torch.triu(torch.ones(sl, sl, device=x.device, dtype=torch.bool), diagonal=1)
        h = self.body(self.embed(x), mask=mask, is_causal=True)
        return self.head(h)


class TinyGRULM(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, n_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, num_layers=n_layers, batch_first=True, dropout=0.0)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h, _ = self.gru(self.embed(x))
        return self.head(h)


# ---------------------------------------------------------------------------
# Search, training, BLEU, and persistence
# ---------------------------------------------------------------------------
def indices_terminated(target, eos_token):
    terminated = [i for i, t in enumerate(target) if eos_token in t]
    non_terminated = [i for i, t in enumerate(target) if eos_token not in t]
    return torch.LongTensor(terminated), torch.LongTensor(non_terminated)


def beautify(sentence: str) -> str:
    for p in {".", ",", ";"}:
        sentence = sentence.replace(f" {p}", p)
    for l in {"-", "'"}:
        sentence = sentence.replace(f"{l} ", l).replace(f" {l}", l)
    return sentence


def append_beams(target, beams):
    batch_size, n_beams = beams.shape
    n_tokens = target.shape[1]
    target = einops.repeat(target, "b t -> b c t", c=n_beams)
    beams = beams.unsqueeze(dim=2)
    target = torch.cat((target, beams), dim=2)
    return target.view(batch_size * n_beams, n_tokens + 1)


def beam_search(model, source, src_vocab, tgt_vocab, src_tokenizer, device, beam_width, max_target, max_sentence_length):
    src_tokens = ["<bos>"] + src_tokenizer(source) + ["<eos>"]
    src_tokens = src_vocab(src_tokens)
    tgt_tokens = tgt_vocab(["<bos>"])

    src_tokens = torch.LongTensor(src_tokens).to(device)
    tgt_tokens = torch.LongTensor(tgt_tokens).unsqueeze(dim=0).to(device)
    target_probs = torch.FloatTensor([1]).to(device)
    model.to(device)
    model.eval()
    eos_idx = tgt_vocab["<eos>"]

    with torch.no_grad():
        while tgt_tokens.shape[1] < max_sentence_length:
            src = einops.repeat(src_tokens, "t -> b t", b=tgt_tokens.shape[0])
            predicted = torch.softmax(model(src, tgt_tokens), dim=-1)
            probs, predicted = predicted[:, -1].topk(k=beam_width, dim=-1)

            idx_terminated, idx_not_terminated = indices_terminated(tgt_tokens, eos_idx)
            idx_terminated = idx_terminated.to(device)
            idx_not_terminated = idx_not_terminated.to(device)

            tgt_terminated = torch.index_select(tgt_tokens, dim=0, index=idx_terminated)
            tgt_probs_terminated = torch.index_select(target_probs, dim=0, index=idx_terminated)
            filter_t = lambda t: torch.index_select(t, dim=0, index=idx_not_terminated)

            tgt_others = filter_t(tgt_tokens)
            if tgt_others.shape[0] == 0:
                break
            tgt_probs_others = filter_t(target_probs)
            predicted = filter_t(predicted)
            probs = filter_t(probs)

            tgt_others = append_beams(tgt_others, predicted)
            padd = torch.zeros((len(tgt_terminated), 1), dtype=torch.long, device=device)
            tgt_terminated = torch.cat((tgt_terminated, padd), dim=1)

            tgt_probs_others = torch.repeat_interleave(tgt_probs_others, beam_width)
            tgt_probs_others *= probs.flatten()
            tgt_probs_terminated *= 0.999

            target_probs = torch.cat((tgt_probs_others, tgt_probs_terminated), dim=0)
            tgt_tokens = torch.cat((tgt_others, tgt_terminated), dim=0)

            if target_probs.shape[0] > max_target:
                target_probs, indices = target_probs.topk(k=max_target, dim=0)
                tgt_tokens = torch.index_select(tgt_tokens, dim=0, index=indices)

    sentences = []
    for tgt_sentence in tgt_tokens:
        tgt_sentence = tgt_sentence[1:].tolist()
        tgt_sentence = list(takewhile(lambda t: t != eos_idx, tgt_sentence))
        sentences.append(" ".join(tgt_vocab.lookup_tokens(tgt_sentence)))
    sentences = [beautify(s) for s in sentences]
    sentences = [(s, p.item()) for s, p in zip(sentences, target_probs)]
    return sorted(sentences, key=lambda k: k[1], reverse=True)


def topk_accuracy(real_tokens, probs_tokens, k, tgt_pad_idx):
    total = (real_tokens != tgt_pad_idx).sum()
    if total.item() == 0:
        return torch.tensor(0.0, device=real_tokens.device)
    k = min(k, probs_tokens.shape[-1])
    _, pred_tokens = probs_tokens.topk(k=k, dim=-1)
    real_tokens_rep = einops.repeat(real_tokens, "b -> b k", k=k)
    good = (pred_tokens == real_tokens_rep) & (real_tokens_rep != tgt_pad_idx)
    return good.sum() / total


def loss_batch(model, source, target, config):
    loss_fn = config["loss"].to(config["device"])
    source, target = source.to(config["device"]), target.to(config["device"])
    target_in, target_out = target[:, :-1], target[:, 1:]
    pred = model(source, target_in)
    pred = pred.reshape(-1, pred.shape[2])
    target_out = target_out.flatten()
    metrics = {"loss": loss_fn(pred, target_out)}
    for k in [1, 5, 10]:
        metrics[f"top-{k}"] = topk_accuracy(target_out, pred, k, config["tgt_pad_idx"])
    return metrics


def eval_model(model, dataloader, config):
    logs = defaultdict(list)
    model.to(config["device"])
    model.eval()
    with torch.no_grad():
        for source, target in dataloader:
            metrics = loss_batch(model, source, target, config)
            for name, value in metrics.items():
                logs[name].append(value.detach().cpu().item())
    return {name: float(np.mean(values)) for name, values in logs.items()}


def train_model_for_sweep(model, config):
    """Training loop for E8-E10. It skips notebook-only wandb tables and sample beam logs."""
    train_loader, val_loader = config["train_loader"], config["val_loader"]
    optimizer = config["optimizer"]
    clip = config["clip"]
    print(f"Starting training for {config['epochs']} epochs on {config['device']}.")
    last_train_logs = {}
    last_val_logs = {}

    for e in range(config["epochs"]):
        model.to(config["device"])
        model.train()
        logs = defaultdict(list)
        for source, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            metrics = loss_batch(model, source, target, config)
            loss = metrics["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            for name, value in metrics.items():
                logs[name].append(value.detach().cpu().item())

        last_train_logs = {name: float(np.mean(values)) for name, values in logs.items()}
        last_val_logs = eval_model(model, val_loader, config)
        print(
            f"Epoch {e + 1:02d}/{config['epochs']} | "
            f"train_loss={last_train_logs.get('loss', float('nan')):.4f} | "
            f"val_loss={last_val_logs.get('loss', float('nan')):.4f} | "
            f"val_top1={last_val_logs.get('top-1', float('nan')):.4f}"
        )
    return {"train": last_train_logs, "validation": last_val_logs}


@torch.no_grad()
def compute_bleu(model, dataset, config, n_samples=200):
    model.eval()
    hypotheses, references = [], []
    n_samples = min(n_samples, len(dataset))
    indices = torch.randperm(len(dataset))[:n_samples]
    for idx in indices:
        en_sentence, fr_sentence = dataset.dataset[int(idx)]
        try:
            pred, _ = beam_search(
                model, en_sentence, config["src_vocab"], config["tgt_vocab"], config["src_tokenizer"],
                config["device"], beam_width=5, max_target=50, max_sentence_length=config["max_sequence_length"]
            )[0]
            hypotheses.append(pred)
            references.append(fr_sentence)
        except Exception as exc:
            print(f"BLEU sample skipped: {exc}")
    if not hypotheses:
        return float("nan")
    return sacrebleu.corpus_bleu(hypotheses, [references]).score


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if not isinstance(v, torch.Tensor)}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return None
    return obj


def save_results(results, name, outdir=RESULTS_DIR):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"{name}.pkl", "wb") as f:
        pickle.dump(results, f)
    with open(outdir / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(results), f, indent=2, default=str)
    print(f"Saved: {outdir / (name + '.pkl')}")


def load_results(name, outdir=RESULTS_DIR):
    pkl = Path(outdir) / f"{name}.pkl"
    if not pkl.exists():
        raise FileNotFoundError(pkl)
    with open(pkl, "rb") as f:
        return pickle.load(f)


def maybe_load_results(name):
    try:
        return load_results(name)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Configs and sweeps
# ---------------------------------------------------------------------------
MAMBA_TRANSLATION_BASE = {
    "dim_embedding": 256,
    "n_layers": 4,
    "d_state": 16,
    "d_conv": 4,
    "expand": 2,
    "dropout": 0.1,
    "epochs": 20,
    "batch_size": 256,
    "lr": 5e-4,
    "betas": (0.9, 0.98),
    "weight_decay": 1e-5,
    "clip": 1.0,
    "log_every": 100,
    "max_sequence_length": 40,
}

MAMBA_TRANSLATION_FAST = {
    **MAMBA_TRANSLATION_BASE,
    "dim_embedding": 128,
    "n_layers": 3,
    "epochs": 20,
    "batch_size": 256,
}


def make_config(train_dataset, val_dataset, en_tokenizer, fr_tokenizer, epochs_override=None, batch_size_override=None):
    max_seq_len = 60
    min_tok_freq = 2
    cfg = {
        "epochs": 20,
        "batch_size": 128,
        "lr": 5e-4,
        "betas": (0.9, 0.98),
        "weight_decay": 1e-5,
        "clip": 1.0,
        "device": device,
        "dim_embedding": 256,
        "n_layers": 4,
        "d_state": 16,
        "d_conv": 4,
        "expand": 2,
        "dropout": 0.1,
        "n_tokens_src": len(train_dataset.en_vocab),
        "n_tokens_tgt": len(train_dataset.fr_vocab),
        "src_vocab": train_dataset.en_vocab,
        "tgt_vocab": train_dataset.fr_vocab,
        "src_tokenizer": en_tokenizer,
        "tgt_tokenizer": fr_tokenizer,
        "src_pad_idx": train_dataset.en_vocab["<pad>"],
        "tgt_pad_idx": train_dataset.fr_vocab["<pad>"],
        "max_sequence_length": max_seq_len,
        "min_token_freq": min_tok_freq,
        "seed": 0,
        "log_every": 50,
    }
    if epochs_override is not None:
        cfg["epochs"] = epochs_override
    if batch_size_override is not None:
        cfg["batch_size"] = batch_size_override

    torch.manual_seed(cfg["seed"])
    cfg["train_loader"] = DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=True,
        collate_fn=lambda batch: generate_batch(batch, cfg["src_pad_idx"], cfg["tgt_pad_idx"])
    )
    cfg["val_loader"] = DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False,
        collate_fn=lambda batch: generate_batch(batch, cfg["src_pad_idx"], cfg["tgt_pad_idx"])
    )
    weight_classes = torch.ones(cfg["n_tokens_tgt"], dtype=torch.float)
    weight_classes[cfg["tgt_vocab"]["<unk>"]] = 0.1
    cfg["loss"] = nn.CrossEntropyLoss(weight=weight_classes, ignore_index=cfg["tgt_pad_idx"])

    config_fast = dict(cfg)
    config_fast.update(MAMBA_TRANSLATION_FAST)
    # Keep the already-created DataLoaders/loss/vocabs/tokenizers.
    for key in ["train_loader", "val_loader", "loss", "src_vocab", "tgt_vocab", "src_tokenizer", "tgt_tokenizer", "src_pad_idx", "tgt_pad_idx", "n_tokens_src", "n_tokens_tgt", "device"]:
        config_fast[key] = cfg[key]
    if epochs_override is not None:
        config_fast["epochs"] = epochs_override
    return cfg, config_fast


def run_translation_sweep(param_name, values, base_config, train_dataset, val_dataset, result_name, force=False, n_bleu_samples=200):
    if not force:
        existing = maybe_load_results(result_name)
        if existing is not None:
            print(f"Skipping {result_name}; found existing results/{result_name}.pkl")
            return existing

    print(f"Translation sweep: {param_name} in {values}")
    results = {"param_name": param_name, "values": [], "val_loss": [], "top1": [], "bleu": [], "training_time_s": []}

    for val in values:
        print(f"\n--- {param_name} = {val} ---")
        model_kwargs = {
            "n_tokens_src": len(train_dataset.en_vocab),
            "n_tokens_tgt": len(train_dataset.fr_vocab),
            "dim_embedding": base_config["dim_embedding"],
            "n_layers": base_config["n_layers"],
            "d_state": base_config["d_state"],
            "d_conv": base_config["d_conv"],
            "expand": base_config["expand"],
            "dropout": base_config["dropout"],
            "src_pad_idx": train_dataset.en_vocab["<pad>"],
            "tgt_pad_idx": train_dataset.fr_vocab["<pad>"],
        }
        model_kwargs[param_name] = val
        torch.manual_seed(0)
        model = TranslationMamba(**model_kwargs).to(device)

        config_run = dict(base_config)
        config_run[param_name] = val
        config_run["optimizer"] = optim.AdamW(
            model.parameters(), lr=config_run["lr"], betas=config_run["betas"], weight_decay=config_run["weight_decay"]
        )

        t0 = time.time()
        logs = train_model_for_sweep(model, config_run)
        elapsed = time.time() - t0

        print("Computing BLEU...")
        bleu_score = compute_bleu(model, val_dataset, config_run, n_samples=n_bleu_samples)
        print(f"  -> BLEU = {bleu_score:.2f}")

        results["values"].append(val)
        results["val_loss"].append(logs["validation"].get("loss", float("nan")))
        results["top1"].append(logs["validation"].get("top-1", float("nan")))
        results["bleu"].append(bleu_score)
        results["training_time_s"].append(elapsed)
        save_results(results, result_name)  # incremental save after each value

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


# ---------------------------------------------------------------------------
# E11 benchmark
# ---------------------------------------------------------------------------
def benchmark_model(make_model, seq_lengths, batch_size=4, vocab_size=1000, n_warmup=2, n_trials=3, include_backward=True):
    results = {"seq_len": [], "fwd_ms": [], "bwd_ms": [], "peak_mb": [], "oom": []}
    for L in seq_lengths:
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        try:
            model = make_model().to(device)
            model.train()
            for _ in range(n_warmup):
                x = torch.randint(0, vocab_size, (batch_size, L), device=device)
                out = model(x)
                if include_backward:
                    out.sum().backward()
                    model.zero_grad(set_to_none=True)
                del out, x
            if device.type == "cuda":
                torch.cuda.synchronize()

            fwd_times = []
            for _ in range(n_trials):
                x = torch.randint(0, vocab_size, (batch_size, L), device=device)
                if device.type == "cuda": torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad():
                    out = model(x)
                if device.type == "cuda": torch.cuda.synchronize()
                fwd_times.append(time.time() - t0)
                del out, x

            bwd_times = []
            if include_backward:
                if device.type == "cuda": torch.cuda.reset_peak_memory_stats()
                for _ in range(n_trials):
                    x = torch.randint(0, vocab_size, (batch_size, L), device=device)
                    if device.type == "cuda": torch.cuda.synchronize()
                    t0 = time.time()
                    out = model(x)
                    out.sum().backward()
                    if device.type == "cuda": torch.cuda.synchronize()
                    bwd_times.append(time.time() - t0)
                    model.zero_grad(set_to_none=True)
                    del out, x

            peak_mb = torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else 0.0
            results["seq_len"].append(L)
            results["fwd_ms"].append(float(np.mean(fwd_times) * 1000))
            results["bwd_ms"].append(float(np.mean(bwd_times) * 1000 if include_backward else 0.0))
            results["peak_mb"].append(float(peak_mb))
            results["oom"].append(False)
            print(f"  L={L:>5d} | fwd={np.mean(fwd_times)*1000:7.2f}ms | bwd={np.mean(bwd_times)*1000 if bwd_times else 0:7.2f}ms | peak={peak_mb:7.1f}MB")
            del model
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                print(f"  L={L:>5d} | OUT OF MEMORY")
                results["seq_len"].append(L)
                results["fwd_ms"].append(float("nan"))
                results["bwd_ms"].append(float("nan"))
                results["peak_mb"].append(float("nan"))
                results["oom"].append(True)
            else:
                raise
        finally:
            gc.collect()
            if device.type == "cuda": torch.cuda.empty_cache()
    return results


def run_full_benchmark(seq_lengths=None, d_model=128, n_layers=4, batch_size=4):
    if seq_lengths is None:
        seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192]
    print(f"Efficiency benchmark (d_model={d_model}, n_layers={n_layers}, batch_size={batch_size}, device={device})")
    results = {}
    print("\nBenchmarking Mamba...")
    results["mamba"] = benchmark_model(lambda: MambaLMForBenchmark(d_model=d_model, n_layers=n_layers), seq_lengths, batch_size=batch_size)
    print("\nBenchmarking Transformer...")
    results["transformer"] = benchmark_model(lambda: TinyTransformerLM(d_model=d_model, n_layers=n_layers), seq_lengths, batch_size=batch_size)
    print("\nBenchmarking GRU...")
    results["gru"] = benchmark_model(lambda: TinyGRULM(d_model=d_model, n_layers=n_layers), seq_lengths, batch_size=batch_size)
    return results


def pretty_print_results(results):
    seq_lens = results["mamba"]["seq_len"]
    print(f'\n{"Seq Len":>8} | {"Mamba fwd":>11} | {"Trans fwd":>11} | {"GRU fwd":>11} | {"Mamba mem":>11} | {"Trans mem":>11}')
    print("-" * 80)
    for i, L in enumerate(seq_lens):
        def fmt(v, unit="ms"):
            if isinstance(v, float) and np.isnan(v):
                return "OOM".rjust(11)
            return f"{v:>8.1f} {unit}"
        print(f'{L:>8d} | {fmt(results["mamba"]["fwd_ms"][i]):>11} | {fmt(results["transformer"]["fwd_ms"][i]):>11} | {fmt(results["gru"]["fwd_ms"][i]):>11} | {fmt(results["mamba"]["peak_mb"][i], "MB"):>11} | {fmt(results["transformer"]["peak_mb"][i], "MB"):>11}')


# ---------------------------------------------------------------------------
# Summary and figures
# ---------------------------------------------------------------------------
def make_summary_table(results_by_name):
    summary_data = []

    def add(exp, metric, value, params="-"):
        summary_data.append({"Expérience": exp, "Métrique clé": metric, "Valeur": value, "Paramètres": params})

    r = results_by_name
    if r.get("E1_translation"):
        x = r["E1_translation"]
        add("E1 - Traduction EN→FR", "BLEU", f"{x.get('bleu', float('nan')):.2f}", f"{x.get('total_params', 0):,}")
    if r.get("E3_selective_copying"):
        x = r["E3_selective_copying"]
        add("E3 - Selective Copying", "Val accuracy", f"{x.get('final_val_acc', float('nan')):.4f}")
    if r.get("E4_induction_heads"):
        x = r["E4_induction_heads"]
        add("E4 - Induction Heads (train)", "Val accuracy", f"{x.get('final_val_acc', float('nan')):.4f}", f"train_len={x.get('train_len', '-')}")
        extrap = x.get("extrapolation", {})
        if 2048 in extrap:
            add("E4 - Induction Heads (L=2048)", "Val accuracy", f"{extrap[2048]:.4f}")
    if r.get("E5_selectivity_ablation"):
        x = r["E5_selectivity_ablation"]
        try:
            full_acc = [c["final_val_acc"] for c in x["configs"] if "Full" in c["name"]][0]
            none_acc = [c["final_val_acc"] for c in x["configs"] if "non-selective" in c["name"]][0]
            add("E5 - Full Mamba", "Val accuracy", f"{full_acc:.4f}", f"vs {none_acc:.4f} sans sélection")
        except Exception:
            pass
    if r.get("E6_state_size"):
        x = r["E6_state_size"]
        best = max(range(len(x["val_acc"])), key=lambda i: x["val_acc"][i])
        add("E6 - Best d_state", "Val accuracy", f"{x['val_acc'][best]:.4f}", f"N={x['d_state_values'][best]}")
    if r.get("E7_A_init"):
        x = r["E7_A_init"]
        best = max(range(len(x["inits"])), key=lambda i: x["inits"][i]["final_val_acc"])
        add("E7 - Best A init", "Val accuracy", f"{x['inits'][best]['final_val_acc']:.4f}", x["inits"][best]["A_init"])
    if r.get("E11_efficiency"):
        x = r["E11_efficiency"]
        L_min = x["mamba"]["seq_len"][0]
        L_max = x["mamba"]["seq_len"][-1]
        ratio = x["mamba"]["fwd_ms"][-1] / x["mamba"]["fwd_ms"][0]
        add("E11 - Mamba scaling", f"Time ratio L={L_max}/L={L_min}", f"{ratio:.1f}x", f"for L x {L_max // L_min}")
    for key, label, param in [
        ("E8_layers", "E8 - Best n_layers", "n_layers"),
        ("E9_expand", "E9 - Best expand", "expand"),
        ("E10_dropout", "E10 - Best dropout", "dropout"),
    ]:
        if r.get(key):
            x = r[key]
            best = max(range(len(x["bleu"])), key=lambda i: x["bleu"][i])
            add(label, "BLEU", f"{x['bleu'][best]:.2f}", f"{param}={x['values'][best]}")

    df = pd.DataFrame(summary_data)
    out = RESULTS_DIR / "summary_table.csv"
    df.to_csv(out, index=False)
    print("\n" + df.to_string(index=False))
    print(f"\nSaved: {out}")
    return df


def make_figures(results_by_name):
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    COLORS = {
        "mamba": "#2E86AB", "transformer": "#A23B72", "gru": "#F18F01", "rnn": "#C73E1D",
        "full_mamba": "#2E86AB", "non_select": "#B0B0B0", "delta_only": "#6FA8DC", "B_only": "#93C47D", "C_only": "#F6B26B",
    }
    mpl.rcParams.update({"font.size": 11, "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": "--"})

    def _save(fig, save_as):
        for ext in ("png", "pdf"):
            path = FIGURES_DIR / f"{save_as}.{ext}"
            fig.savefig(path)
            print(f"Saved: {path}")
        plt.close(fig)

    def plot_hyperparameter_sweep(results, metric="bleu", ylabel="BLEU", title=None, save_as=None):
        xs, ys, name = results["values"], results[metric], results["param_name"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(xs, ys, color=COLORS["mamba"], marker="o")
        ax.set_xlabel(name); ax.set_ylabel(ylabel); ax.set_title(title or f"Effect of {name} on {ylabel}")
        for x, y in zip(xs, ys):
            if y is not None and not np.isnan(y):
                ax.annotate(f"{y:.2f}", xy=(x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
        fig.tight_layout(); _save(fig, save_as)

    def plot_efficiency_benchmark(results, metric="fwd_ms", title="Efficiency benchmark", save_as=None):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for name, r in results.items():
            xs = np.asarray(r["seq_len"]); ys = np.asarray(r[metric], dtype=float)
            ax.plot(xs, ys, marker="o", label=name.capitalize())
        ax.set_xscale("log", base=2); ax.set_yscale("log")
        ax.set_xlabel("Sequence length"); ax.set_ylabel("Forward time (ms)" if metric == "fwd_ms" else "Peak memory (MB)")
        ax.set_title(title); ax.legend(); fig.tight_layout(); _save(fig, save_as)

    for key, fig_name, title in [
        ("E8_layers", "fig_06_E8_layers", "E8: Effect of n_layers on BLEU"),
        ("E9_expand", "fig_07_E9_expand", "E9: Effect of expand factor on BLEU"),
        ("E10_dropout", "fig_08_E10_dropout", "E10: Effect of dropout on BLEU"),
    ]:
        if results_by_name.get(key):
            plot_hyperparameter_sweep(results_by_name[key], title=title, save_as=fig_name)

    if results_by_name.get("E11_efficiency"):
        plot_efficiency_benchmark(results_by_name["E11_efficiency"], "fwd_ms", "E11: Forward pass time vs sequence length", "fig_09a_E11_efficiency_time")
        plot_efficiency_benchmark(results_by_name["E11_efficiency"], "peak_mb", "E11: Peak memory vs sequence length", "fig_09b_E11_efficiency_memory")

    zip_path = ROOT / "mamba_figures.zip"
    if FIGURES_DIR.exists():
        shutil.make_archive(str(zip_path.with_suffix("")), "zip", FIGURES_DIR)
        print(f"Archive created: {zip_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Resume the project notebook from E8 onward.")
    parser.add_argument("--force", action="store_true", help="re-run E8/E9/E10/E11 even if result files already exist")
    parser.add_argument("--epochs", type=int, default=None, help="override epochs for E8-E10, useful for a smoke test")
    parser.add_argument("--bleu-samples", type=int, default=200, help="number of validation samples for BLEU")
    parser.add_argument("--skip-e11", action="store_true", help="skip the E11 efficiency benchmark")
    parser.add_argument("--skip-figures", action="store_true", help="skip figure generation")
    args = parser.parse_args()

    print(f"Working directory: {ROOT}")
    print(f"Device: {device}")

    train_dataset, val_dataset, en_tokenizer, fr_tokenizer = load_translation_data()
    _, config_fast = make_config(train_dataset, val_dataset, en_tokenizer, fr_tokenizer, epochs_override=args.epochs)
    print("\nConfig FAST ready:")
    print(f"  dim_embedding = {config_fast['dim_embedding']}")
    print(f"  n_layers      = {config_fast['n_layers']}")
    print(f"  epochs        = {config_fast['epochs']}")
    print(f"  batch_size    = {config_fast['batch_size']} (DataLoader was built from the notebook config)")

    results_E8 = run_translation_sweep("n_layers", [2, 4, 6], config_fast, train_dataset, val_dataset, "E8_layers", force=args.force, n_bleu_samples=args.bleu_samples)
    results_E9 = run_translation_sweep("expand", [1, 2, 4], config_fast, train_dataset, val_dataset, "E9_expand", force=args.force, n_bleu_samples=args.bleu_samples)
    results_E10 = run_translation_sweep("dropout", [0.0, 0.1, 0.3], config_fast, train_dataset, val_dataset, "E10_dropout", force=args.force, n_bleu_samples=args.bleu_samples)

    if args.skip_e11:
        results_E11 = maybe_load_results("E11_efficiency")
    else:
        results_E11 = None if args.force else maybe_load_results("E11_efficiency")
        if results_E11 is None:
            results_E11 = run_full_benchmark()
            pretty_print_results(results_E11)
            save_results(results_E11, "E11_efficiency")
        else:
            print("Skipping E11_efficiency; found existing results/E11_efficiency.pkl")

    result_names = [
        "E1_translation", "E2_decoding", "E3_selective_copying", "E4_induction_heads",
        "E5_selectivity_ablation", "E6_state_size", "E7_A_init",
        "E8_layers", "E9_expand", "E10_dropout", "E11_efficiency",
    ]
    results_by_name = {name: maybe_load_results(name) for name in result_names}
    results_by_name["E8_layers"] = results_E8
    results_by_name["E9_expand"] = results_E9
    results_by_name["E10_dropout"] = results_E10
    results_by_name["E11_efficiency"] = results_E11

    missing = [name for name, value in results_by_name.items() if value is None and name.startswith(("E1", "E3", "E4", "E5", "E6", "E7"))]
    if missing:
        print("\nWarning: these earlier result files were not found, so their rows/figures will be skipped:")
        for name in missing:
            print(f"  - results/{name}.pkl")

    make_summary_table(results_by_name)
    if not args.skip_figures:
        make_figures(results_by_name)

    print("\nDone. New outputs are in ./results, ./figures, and ./mamba_figures.zip")


if __name__ == "__main__":
    main()

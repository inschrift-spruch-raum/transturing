"""
Phase 10: Digit-Level Decomposition

Phase 9 proved the DIFF+ADD wall is NOT a gradient signal problem — it's
representational. FF layers can't learn f(emb(a), emb(b)) → emb(a+b) for
the full 200×200 input space while also handling execution logic.

Fix: decompose numbers into individual digits (little-endian) so the FF
layers only need to learn 10×10→10 per-digit addition with carry.

  Old: 42 → single token (index 51 in vocab of 210)
  New: 42 → [2, 4, 0] (units, tens, hundreds) in vocab of 19

Little-endian is critical: autoregressive prediction sees units first,
so the model can compute (a_units + b_units) mod 10 trivially. The tens
digit prediction has carry info available from the units digit already
in context. This matches how humans learn column addition.

Micro-op step expands from 6 → 16 tokens:
  Old: [OP, ARG, F1, F2, SP, TOP]
  New: [OP, ARG×3, F1×3, F2×3, SP×3, TOP×3]

Vocab shrinks from 210 → 19: smaller embedding, more capacity for logic.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import time
import json
import os
from typing import List, Dict, Set, Tuple
from dataclasses import dataclass, asdict

import sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase4_stack_machine import (
    Instruction, OP_PUSH, OP_POP, OP_ADD, OP_DUP, OP_HALT, OP_NAMES,
    TOK_PROG_START, TOK_PROG_END, TOK_TRACE_START,
)
from phase6_curriculum import (
    constrained_random_program, CheckpointMeta, save_checkpoint, load_checkpoint, STAGES
)
from phase8_microop_traces import (
    MicroOpExecutor, MicroOpTrace, MicroOpStep,
)


# ─── Digit-Level Vocabulary ────────────────────────────────────

N_DIGITS = 3           # 3 digits → values 0-999 (covers our 0-200 range)
PAD_TOKEN = 0
OPCODE_OFFSET = 1      # opcodes at 1..5
SPECIAL_OFFSET = 6     # PROG_START=6, PROG_END=7, TRACE_START=8
DIGIT_OFFSET = 9       # digits 0-9 at indices 9..18
DIGIT_VOCAB_SIZE = 19   # 0(PAD) + 5(opcodes) + 3(specials) + 10(digits)

DIGIT_TOKENS_PER_STEP = 1 + N_DIGITS * 5  # OP + 5 numeric fields × 3 digits = 16


def num_to_digits(n: int, n_digits: int = N_DIGITS) -> List[int]:
    """Convert number to little-endian digit list. 42 → [2, 4, 0]."""
    n = max(0, min(n, 10**n_digits - 1))  # clamp
    digits = []
    for _ in range(n_digits):
        digits.append(n % 10)
        n //= 10
    return digits


def digits_to_num(digits: List[int]) -> int:
    """Convert little-endian digit list to number. [2, 4, 0] → 42."""
    n = 0
    for i, d in enumerate(digits):
        n += d * (10 ** i)
    return n


def encode_digit(d: int) -> int:
    """Encode a single digit (0-9) to vocab index."""
    return DIGIT_OFFSET + d


def decode_digit(idx: int) -> int:
    """Decode vocab index to digit (0-9). Returns -1 if not a digit."""
    if DIGIT_OFFSET <= idx < DIGIT_OFFSET + 10:
        return idx - DIGIT_OFFSET
    return -1


def encode_opcode(op: int) -> int:
    """Encode opcode (1-5) to vocab index."""
    return OPCODE_OFFSET + op - 1


def decode_opcode(idx: int) -> int:
    """Decode vocab index to opcode. Returns -1 if not an opcode."""
    if OPCODE_OFFSET <= idx < OPCODE_OFFSET + 5:
        return idx - OPCODE_OFFSET + 1
    return -1


def encode_special(raw: int) -> int:
    """Encode special token to vocab index."""
    if raw == TOK_PROG_START:
        return SPECIAL_OFFSET + 0
    elif raw == TOK_PROG_END:
        return SPECIAL_OFFSET + 1
    elif raw == TOK_TRACE_START:
        return SPECIAL_OFFSET + 2
    raise ValueError(f"Unknown special token: {raw}")


def encode_num_field(val: int) -> List[int]:
    """Encode a numeric value as N_DIGITS digit tokens (little-endian)."""
    return [encode_digit(d) for d in num_to_digits(val)]


def decode_num_field(tokens: List[int]) -> int:
    """Decode N_DIGITS digit tokens back to a number."""
    digits = [decode_digit(t) for t in tokens]
    if any(d < 0 for d in digits):
        return -1
    return digits_to_num(digits)


# ─── Digit-Level Trace Encoding ─────────────────────────────────

def microop_trace_to_digit_tokens(trace: MicroOpTrace) -> List[int]:
    """Convert a MicroOpTrace to digit-level token sequence.

    Program section: [PROG_START, OP1, arg1_d0, arg1_d1, arg1_d2, OP2, ..., PROG_END]
    Trace section:   [TRACE_START, OP, ARG_d0..d2, F1_d0..d2, F2_d0..d2, SP_d0..d2, TOP_d0..d2, ...]
    """
    tokens = [encode_special(TOK_PROG_START)]
    for instr in trace.program_instrs:
        tokens.append(encode_opcode(instr.op))
        tokens.extend(encode_num_field(instr.arg))
    tokens.append(encode_special(TOK_PROG_END))
    tokens.append(encode_special(TOK_TRACE_START))

    for step in trace.steps:
        tokens.append(encode_opcode(step.op))
        tokens.extend(encode_num_field(step.arg))
        tokens.extend(encode_num_field(step.fetch1))
        tokens.extend(encode_num_field(step.fetch2))
        tokens.extend(encode_num_field(step.sp))
        tokens.extend(encode_num_field(step.top))

    return tokens


# ─── Data Generation ────────────────────────────────────────────

def generate_digit_data(
    allowed_ops: Set[int],
    n_samples: int,
    min_len: int = 3,
    max_len: int = 8,
    max_push_val: int = 30
) -> List[List[int]]:
    """Generate digit-encoded micro-op trace sequences."""
    executor = MicroOpExecutor()
    seqs = []
    attempts = 0
    max_attempts = n_samples * 5

    while len(seqs) < n_samples and attempts < max_attempts:
        attempts += 1
        prog = constrained_random_program(allowed_ops, min_len, max_len, max_push_val)
        try:
            trace = executor.execute(prog)
            tokens = microop_trace_to_digit_tokens(trace)
            if all(0 <= t < DIGIT_VOCAB_SIZE for t in tokens):
                seqs.append(tokens)
        except Exception:
            continue

    return seqs


# ─── Dataset ────────────────────────────────────────────────────

class DigitTraceDataset(Dataset):
    """Padded digit-level sequences for next-token prediction."""

    def __init__(self, sequences: List[List[int]], max_len: int = None):
        if max_len is None:
            max_len = max(len(s) for s in sequences)
        self.max_len = max_len

        self.data = []
        for seq in sequences:
            if len(seq) > max_len:
                seq = seq[:max_len]
            padded = seq + [PAD_TOKEN] * (max_len - len(seq))
            self.data.append(padded)

        self.data = torch.tensor(self.data, dtype=torch.long)
        self.lengths = torch.tensor([min(len(s), max_len) for s in sequences])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.lengths[idx]


# ─── Model ──────────────────────────────────────────────────────

class DigitTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=mask)
        x = x + self.dropout(attn_out)
        x = x + self.ff(self.ln2(x))
        return x


class DigitTransformer(nn.Module):
    """Decoder-only transformer with configurable vocab size for digit tokens."""

    def __init__(self, vocab_size: int = DIGIT_VOCAB_SIZE,
                 d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, d_ff: int = None,
                 max_len: int = 300, dropout: float = 0.1):
        super().__init__()

        if d_ff is None:
            d_ff = d_model * 4

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout_layer = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DigitTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

        self.n_params = sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        h = self.dropout_layer(tok + pos)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        for layer in self.layers:
            h = layer(h, mask)

        h = self.ln_final(h)
        return self.output(h)


# ─── Evaluation ─────────────────────────────────────────────────

def evaluate_digit_execution(
    model: DigitTransformer,
    test_progs: List[List[Instruction]],
    verbose: bool = False
) -> Dict:
    """Evaluate model on digit-level micro-op trace format."""
    executor = MicroOpExecutor()
    model.eval()

    results = {
        'total': len(test_progs),
        'perfect': 0,
        'final_correct': 0,
        'token_errors': [],
        'add_top_correct': 0,
        'add_top_total': 0,
        'examples': [],
    }

    trace_start_enc = encode_special(TOK_TRACE_START)

    with torch.no_grad():
        for prog in test_progs:
            ref_trace = executor.execute(prog)
            ref_tokens = microop_trace_to_digit_tokens(ref_trace)

            try:
                trace_start_idx = ref_tokens.index(trace_start_enc)
            except ValueError:
                continue

            prefix = ref_tokens[:trace_start_idx + 1]
            generated = list(prefix)
            n_trace_tokens = len(ref_tokens) - len(prefix)

            for _ in range(n_trace_tokens + DIGIT_TOKENS_PER_STEP):
                inp = torch.tensor([generated], dtype=torch.long)
                logits = model(inp)
                next_tok = logits[0, -1].argmax().item()
                generated.append(next_tok)
                if len(generated) >= len(ref_tokens):
                    break

            gen_trace = generated[trace_start_idx + 1:]
            ref_trace_tokens = ref_tokens[trace_start_idx + 1:]

            min_len_t = min(len(gen_trace), len(ref_trace_tokens))
            gen_trace = gen_trace[:min_len_t]
            ref_trace_tokens = ref_trace_tokens[:min_len_t]

            errors = sum(1 for g, r in zip(gen_trace, ref_trace_tokens) if g != r)
            results['token_errors'].append(errors)

            if errors == 0:
                results['perfect'] += 1

            # Check final TOP value (last N_DIGITS tokens of last step)
            if min_len_t >= DIGIT_TOKENS_PER_STEP:
                ref_final = ref_trace_tokens[-N_DIGITS:]
                gen_final = gen_trace[-N_DIGITS:] if len(gen_trace) >= len(ref_trace_tokens) else [-1]*N_DIGITS
                if ref_final == gen_final:
                    results['final_correct'] += 1

            # Check ADD TOP digits specifically
            step_offset = 0
            for step in ref_trace.steps:
                if step_offset + DIGIT_TOKENS_PER_STEP > min_len_t:
                    break
                if step.op == OP_ADD:
                    # TOP digits are at positions [step_offset + 1 + 4*N_DIGITS .. step_offset + 1 + 5*N_DIGITS)
                    top_start = step_offset + 1 + 4 * N_DIGITS
                    top_end = top_start + N_DIGITS
                    if top_end <= min_len_t:
                        results['add_top_total'] += 1
                        ref_top = ref_trace_tokens[top_start:top_end]
                        gen_top = gen_trace[top_start:top_end]
                        if ref_top == gen_top:
                            results['add_top_correct'] += 1
                step_offset += DIGIT_TOKENS_PER_STEP

            if len(results['examples']) < 5:
                prog_str = ' ; '.join(str(i) for i in prog)
                results['examples'].append({
                    'program': prog_str,
                    'ref_trace': ref_trace_tokens[:32],
                    'gen_trace': gen_trace[:32],
                    'errors': errors,
                })

    return results


def run_add_diagnostic_digit(
    model: DigitTransformer,
    n_tests: int = 30,
    verbose: bool = True
) -> Dict:
    """Test the three ADD patterns with digit-level traces."""
    results = {}

    patterns = {
        'DUP+ADD (PUSH a, DUP, ADD)': [],
        'SAME+ADD (PUSH a, PUSH a, ADD)': [],
        'DIFF+ADD (PUSH a, PUSH b, ADD)': [],
    }

    random.seed(99)

    for _ in range(n_tests):
        a = random.randint(1, 25)
        b = random.randint(1, 25)
        while b == a:
            b = random.randint(1, 25)

        patterns['DUP+ADD (PUSH a, DUP, ADD)'].append(
            [Instruction(OP_PUSH, a), Instruction(OP_DUP, 0),
             Instruction(OP_ADD, 0), Instruction(OP_HALT, 0)])

        patterns['SAME+ADD (PUSH a, PUSH a, ADD)'].append(
            [Instruction(OP_PUSH, a), Instruction(OP_PUSH, a),
             Instruction(OP_ADD, 0), Instruction(OP_HALT, 0)])

        patterns['DIFF+ADD (PUSH a, PUSH b, ADD)'].append(
            [Instruction(OP_PUSH, a), Instruction(OP_PUSH, b),
             Instruction(OP_ADD, 0), Instruction(OP_HALT, 0)])

    if verbose:
        print("\n  ADD Diagnostic (digit-level traces):")
        print(f"  {'Pattern':<45} {'Perfect':>10} {'Final OK':>10}")
        print("  " + "-" * 65)

    for name, progs in patterns.items():
        res = evaluate_digit_execution(model, progs)
        results[name] = {
            'perfect': res['perfect'],
            'total': res['total'],
            'final_correct': res['final_correct'],
            'pct_perfect': res['perfect'] / res['total'] * 100,
            'add_top_correct': res['add_top_correct'],
            'add_top_total': res['add_top_total'],
        }
        if verbose:
            pf = f"{res['perfect']}/{res['total']}"
            fc = f"{res['final_correct']}/{res['total']}"
            add_str = ""
            if res['add_top_total'] > 0:
                add_str = f"  ADD TOP: {res['add_top_correct']}/{res['add_top_total']}"
            print(f"  {name:<45} {pf:>10} {fc:>10}{add_str}")

    return results


# ─── Training ───────────────────────────────────────────────────

def train_digit_stage(
    model: DigitTransformer,
    train_data: DigitTraceDataset,
    val_data: DigitTraceDataset,
    stage: int,
    max_epochs: int = 80,
    lr: float = 3e-4,
    batch_size: int = 64,
    patience: int = 25,
    max_wall_time: float = 500.0,
    checkpoint_prefix: str = "phase10",
    checkpoint_dir: str = ".",
    resume: bool = True,
    verbose: bool = True
) -> CheckpointMeta:
    """Train one curriculum stage on digit-level traces."""

    ckpt_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_ckpt_stage{stage}.pt")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    start_epoch = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state = None
    total_wall = 0.0

    if resume and os.path.exists(ckpt_path):
        meta = load_checkpoint(ckpt_path, model, optimizer)
        start_epoch = meta['epoch'] + 1
        history = meta['history']
        best_val_loss = meta['best_val_loss']
        best_val_acc = meta['best_val_acc']
        total_wall = meta['wall_time_s']
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if verbose:
            print(f"  Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, max_epochs, last_epoch=start_epoch - 1 if start_epoch > 0 else -1
    )

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    no_improve = 0
    wall_start = time.time()

    for epoch in range(start_epoch, max_epochs):
        elapsed = time.time() - wall_start
        if elapsed > max_wall_time:
            if verbose:
                print(f"  Wall-clock limit ({max_wall_time:.0f}s) reached at epoch {epoch}")
            break

        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch, lengths in train_loader:
            inp = batch[:, :-1]
            tgt = batch[:, 1:]
            logits = model(inp)

            mask = torch.zeros_like(tgt, dtype=torch.bool)
            for i, l in enumerate(lengths):
                mask[i, :l-1] = True

            loss = F.cross_entropy(logits[mask], tgt[mask])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss /= max(n_batches, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        n_val = 0
        with torch.no_grad():
            for batch, lengths in val_loader:
                inp = batch[:, :-1]
                tgt = batch[:, 1:]
                logits = model(inp)

                mask = torch.zeros_like(tgt, dtype=torch.bool)
                for i, l in enumerate(lengths):
                    mask[i, :l-1] = True

                loss = F.cross_entropy(logits[mask], tgt[mask])
                val_loss += loss.item()
                n_val += 1

                preds = logits.argmax(dim=-1)
                val_correct += (preds[mask] == tgt[mask]).sum().item()
                val_total += mask.sum().item()

        val_loss /= max(n_val, 1)
        val_acc = val_correct / max(val_total, 1)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if verbose and (epoch % 10 == 0 or epoch == max_epochs - 1):
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
                  f"[{time.time()-wall_start:.0f}s]")

        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch}")
                break

    total_wall += time.time() - wall_start
    final_epoch = len(history['val_acc']) - 1

    if best_state:
        model.load_state_dict(best_state)

    meta = CheckpointMeta(
        stage=stage,
        epoch=final_epoch,
        best_val_acc=best_val_acc,
        best_val_loss=best_val_loss,
        total_epochs_trained=len(history['val_acc']),
        wall_time_s=total_wall,
        history=history,
    )
    save_checkpoint(model, optimizer, meta, ckpt_path)
    if verbose:
        print(f"  Checkpoint saved: stage={stage}, epochs={meta.total_epochs_trained}, "
              f"best_acc={best_val_acc:.4f}, wall={total_wall:.1f}s")

    return meta


# ─── Stage Runner ───────────────────────────────────────────────

def run_stage(
    stage: int,
    model: DigitTransformer,
    n_train: int = 5000,
    n_val: int = 500,
    n_test: int = 50,
    checkpoint_prefix: str = "phase10",
    checkpoint_dir: str = ".",
    verbose: bool = True,
) -> Dict:
    """Run a single curriculum stage with digit-level traces."""

    cfg = STAGES[stage]
    if verbose:
        print(f"\n{'='*60}")
        print(f"Stage {stage}: {cfg['name']} (digit-level decomposition)")
        print(f"  {cfg['description']}")
        print(f"  Target: >{cfg['target_acc']*100:.0f}% token accuracy")
        print(f"{'='*60}\n")

    random.seed(42 + stage)
    np.random.seed(42 + stage)
    torch.manual_seed(42 + stage)

    gen_ops = cfg['ops'] - {OP_HALT}

    if verbose:
        print("  Generating digit-level data...")
    train_seqs = generate_digit_data(gen_ops, n_train, max_push_val=cfg['max_push_val'])
    val_seqs = generate_digit_data(gen_ops, n_val, max_push_val=cfg['max_push_val'])

    max_seq_len = max(
        max(len(s) for s in train_seqs),
        max(len(s) for s in val_seqs)
    )

    if verbose:
        print(f"  {len(train_seqs)} train, {len(val_seqs)} val, max_len={max_seq_len}")
        # Show a sample sequence
        sample = train_seqs[0]
        print(f"  Sample seq length: {len(sample)} tokens (vs ~43 in single-token format)")

    train_data = DigitTraceDataset(train_seqs, max_len=max_seq_len)
    val_data = DigitTraceDataset(val_seqs, max_len=max_seq_len)

    meta = train_digit_stage(
        model, train_data, val_data,
        stage=stage,
        max_epochs=cfg['max_epochs'],
        patience=cfg['patience'],
        checkpoint_prefix=checkpoint_prefix,
        checkpoint_dir=checkpoint_dir,
        verbose=verbose,
    )

    if verbose:
        print(f"\n  Execution evaluation...")

    test_progs = [
        constrained_random_program(gen_ops, max_push_val=cfg['max_push_val'])
        for _ in range(n_test)
    ]
    exec_results = evaluate_digit_execution(model, test_progs, verbose=verbose)

    if verbose:
        print(f"  Token accuracy (validation): {meta.best_val_acc:.4f}")
        print(f"  Perfect traces: {exec_results['perfect']}/{exec_results['total']}")
        print(f"  Final value correct: {exec_results['final_correct']}/{exec_results['total']}")
        if exec_results['add_top_total'] > 0:
            pct = exec_results['add_top_correct'] / exec_results['add_top_total'] * 100
            print(f"  ADD TOP digit accuracy: {exec_results['add_top_correct']}/{exec_results['add_top_total']} ({pct:.0f}%)")
        target_met = "YES" if meta.best_val_acc >= cfg['target_acc'] else "NO"
        print(f"  Target met (>{cfg['target_acc']*100:.0f}%): {target_met}")

        if exec_results.get('examples'):
            print(f"\n  Sample traces:")
            for ex in exec_results['examples'][:3]:
                print(f"    {ex['program']}")
                print(f"      ref: {ex['ref_trace'][:32]}...")
                print(f"      gen: {ex['gen_trace'][:32]}...")
                print(f"      errors: {ex['errors']}")

    return {
        'stage': stage,
        'name': cfg['name'],
        'meta': meta.to_dict(),
        'execution': {
            'total': exec_results['total'],
            'perfect': exec_results['perfect'],
            'final_correct': exec_results['final_correct'],
            'add_top_correct': exec_results.get('add_top_correct', 0),
            'add_top_total': exec_results.get('add_top_total', 0),
            'avg_token_errors': float(np.mean(exec_results['token_errors'])) if exec_results['token_errors'] else 0,
        },
    }


# ─── Main Experiment ────────────────────────────────────────────

def run_digit_experiment(checkpoint_dir: str = ".") -> Dict:
    """Run the full digit decomposition experiment.

    Same curriculum stages as Phase 8/9, same model size (d=64, h=4, L=2),
    only change is digit-level tokenization.
    """

    print("=" * 60)
    print("Phase 10: Digit-Level Decomposition")
    print("=" * 60)
    print()
    print("Hypothesis: decomposing numbers into individual digits enables")
    print("the FF layers to learn per-digit addition (10×10→10 mapping)")
    print("instead of the impossible full-range mapping (200×200→200).")
    print()
    print(f"Vocab: {DIGIT_VOCAB_SIZE} tokens (vs 210 in single-token format)")
    print(f"Digits per number: {N_DIGITS} (little-endian)")
    print(f"Tokens per micro-op step: {DIGIT_TOKENS_PER_STEP} (vs 6 in single-token)")
    print()

    model = DigitTransformer(
        vocab_size=DIGIT_VOCAB_SIZE,
        d_model=64, n_heads=4, n_layers=2, d_ff=256,
        max_len=300, dropout=0.1,
    )
    print(f"Model: d=64, h=4, L=2, d_ff=256, params={model.n_params:,}")
    print(f"  (Phase 8/9 baseline: ~137K params with vocab=210)")
    print()

    all_results = {'model_params': model.n_params, 'vocab_size': DIGIT_VOCAB_SIZE}
    total_start = time.time()

    for stage in [1, 2, 3]:
        results = run_stage(
            stage=stage,
            model=model,
            n_train=5000,
            n_val=500,
            n_test=50,
            checkpoint_prefix="phase10",
            checkpoint_dir=checkpoint_dir,
        )
        all_results[f'stage_{stage}'] = results

        # Save incremental
        results_path = os.path.join(checkpoint_dir, "phase10_results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

    # ADD diagnostic
    print(f"\n{'='*60}")
    print("Phase 10: ADD Diagnostic")
    print(f"{'='*60}")
    add_results = run_add_diagnostic_digit(model, n_tests=30, verbose=True)
    all_results['add_diagnostic'] = add_results

    total_time = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print("Phase 10 Summary: Digit-Level Decomposition")
    print(f"{'='*60}")
    print(f"\nTotal time: {total_time:.0f}s ({total_time/60:.1f} min)")

    for stage in [1, 2, 3]:
        key = f'stage_{stage}'
        if key in all_results:
            s = all_results[key]
            print(f"\nStage {stage} ({s['name']}):")
            print(f"  Val accuracy: {s['meta']['best_val_acc']:.4f}")
            print(f"  Perfect traces: {s['execution']['perfect']}/{s['execution']['total']}")
            if s['execution']['add_top_total'] > 0:
                pct = s['execution']['add_top_correct'] / s['execution']['add_top_total'] * 100
                print(f"  ADD TOP accuracy: {s['execution']['add_top_correct']}/{s['execution']['add_top_total']} ({pct:.0f}%)")

    print(f"\nADD Diagnostic:")
    print(f"  {'Pattern':<45} {'Perfect':>10}")
    print("  " + "-" * 55)
    for name, data in add_results.items():
        pf = f"{data['perfect']}/{data['total']}"
        print(f"  {name:<45} {pf:>10}")

    dup_pct = add_results.get('DUP+ADD (PUSH a, DUP, ADD)', {}).get('pct_perfect', 0)
    same_pct = add_results.get('SAME+ADD (PUSH a, PUSH a, ADD)', {}).get('pct_perfect', 0)
    diff_pct = add_results.get('DIFF+ADD (PUSH a, PUSH b, ADD)', {}).get('pct_perfect', 0)

    print(f"\nPhase 9 baseline: DUP+ADD 100%, SAME+ADD 100%, DIFF+ADD 0%")
    print(f"Phase 10 result:  DUP+ADD {dup_pct:.0f}%, SAME+ADD {same_pct:.0f}%, DIFF+ADD {diff_pct:.0f}%")

    if diff_pct > 50:
        print(f"\n>>> DIGIT DECOMPOSITION BREAKS THE WALL! DIFF+ADD {diff_pct:.0f}%")
        print(">>> The bottleneck was representational: single-token numbers prevented")
        print(">>> the FF layers from learning compositional addition.")
    elif diff_pct > 10:
        print(f"\n>>> Significant improvement: DIFF+ADD {diff_pct:.0f}% (was 0%)")
        print(">>> Digit decomposition helps but doesn't fully solve arithmetic.")
    elif diff_pct > 0:
        print(f"\n>>> Marginal improvement: DIFF+ADD {diff_pct:.0f}% (was 0%)")
    else:
        print(f"\n>>> Digit decomposition alone insufficient for DIFF+ADD.")

    all_results['total_time_s'] = total_time
    results_path = os.path.join(checkpoint_dir, "phase10_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    return all_results


if __name__ == "__main__":
    results = run_digit_experiment(checkpoint_dir=".")

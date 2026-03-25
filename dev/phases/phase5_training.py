"""
Phase 5: Trained Micro-Executor

Train a small transformer from scratch on (program, execution_trace) pairs.
The model learns to predict the next token in execution traces — if it gets
100% accuracy, it has learned to execute programs.

Key questions:
  1. Can a tiny transformer learn perfect execution?
  2. Does it discover the parabolic encoding (or something else)?
  3. How does model size affect learnability?

Architecture: decoder-only transformer, causal attention.
Training: next-token prediction (cross-entropy) on execution traces.
Evaluation: token-level accuracy + program-level exact match.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import time
import math
from typing import List, Tuple, Dict
from dataclasses import dataclass

# Import Phase 4 for data generation
import sys
sys.path.insert(0, '.')
from phase4_stack_machine import (
    program, Instruction, ReferenceExecutor, Trace, TraceStep,
    OP_PUSH, OP_POP, OP_ADD, OP_DUP, OP_HALT, OP_NAMES,
    TOK_PROG_START, TOK_PROG_END, TOK_TRACE_START,
    TOKENS_PER_STEP
)


# ─── Vocabulary ────────────────────────────────────────────────────

# Token space: [0, VOCAB_SIZE)
# We need to encode: opcodes (1-5), special tokens (100-102), 
# and numeric values (SP, TOP, ARG) which can be 0..MAX_VAL.
#
# Strategy: offset everything into a flat vocab.
#   0         = PAD
#   1..5      = opcodes (PUSH..HALT) 
#   6         = PROG_START
#   7         = PROG_END
#   8         = TRACE_START
#   9..9+MAX_VAL = numeric values (0, 1, 2, ... MAX_VAL)

PAD_TOKEN = 0
OPCODE_OFFSET = 1      # opcodes at 1..5
SPECIAL_OFFSET = 6     # PROG_START=6, PROG_END=7, TRACE_START=8
NUM_OFFSET = 9          # numbers at 9..9+MAX_VAL
MAX_VAL = 200           # max numeric value in traces
VOCAB_SIZE = NUM_OFFSET + MAX_VAL + 1  # = 210


def encode_token(raw: int) -> int:
    """Convert a raw token (from Phase 4's trace format) to vocab index."""
    if raw == TOK_PROG_START:
        return SPECIAL_OFFSET + 0
    elif raw == TOK_PROG_END:
        return SPECIAL_OFFSET + 1
    elif raw == TOK_TRACE_START:
        return SPECIAL_OFFSET + 2
    elif 1 <= raw <= 5:  # opcodes
        return OPCODE_OFFSET + raw - 1  # PUSH=1, POP=2, ...
    else:
        # Numeric value — clamp to [0, MAX_VAL]
        val = max(0, min(raw, MAX_VAL))
        return NUM_OFFSET + val


def decode_token(idx: int) -> int:
    """Convert vocab index back to raw token."""
    if idx == PAD_TOKEN:
        return -1
    elif OPCODE_OFFSET <= idx < OPCODE_OFFSET + 5:
        return idx - OPCODE_OFFSET + 1
    elif idx == SPECIAL_OFFSET + 0:
        return TOK_PROG_START
    elif idx == SPECIAL_OFFSET + 1:
        return TOK_PROG_END
    elif idx == SPECIAL_OFFSET + 2:
        return TOK_TRACE_START
    elif NUM_OFFSET <= idx < VOCAB_SIZE:
        return idx - NUM_OFFSET
    return -1


# ─── Data Generation ──────────────────────────────────────────────

def random_program(min_len: int = 3, max_len: int = 12, 
                   max_push_val: int = 50) -> List[Instruction]:
    """Generate a random valid program that doesn't crash.
    
    Strategy: track stack depth, only emit ops that are valid.
    Always end with HALT.
    """
    instrs = []
    stack_depth = 0
    target_len = random.randint(min_len, max_len)
    
    for _ in range(target_len - 1):  # -1 for HALT
        # Which ops are valid?
        valid_ops = [OP_PUSH]  # PUSH is always valid
        if stack_depth >= 1:
            valid_ops.append(OP_POP)
            valid_ops.append(OP_DUP)
        if stack_depth >= 2:
            valid_ops.append(OP_ADD)
        
        op = random.choice(valid_ops)
        
        if op == OP_PUSH:
            arg = random.randint(0, max_push_val)
            instrs.append(Instruction(op, arg))
            stack_depth += 1
        elif op == OP_POP:
            instrs.append(Instruction(op, 0))
            stack_depth -= 1
        elif op == OP_ADD:
            instrs.append(Instruction(op, 0))
            stack_depth -= 1  # net -1 (pop 2, push 1)
        elif op == OP_DUP:
            instrs.append(Instruction(op, 0))
            stack_depth += 1
        
        # Ensure we have at least 1 value on stack for HALT
        if stack_depth == 0:
            arg = random.randint(0, max_push_val)
            instrs.append(Instruction(OP_PUSH, arg))
            stack_depth += 1
    
    # Ensure at least one value on stack
    if stack_depth == 0:
        instrs.append(Instruction(OP_PUSH, random.randint(0, max_push_val)))
    
    instrs.append(Instruction(OP_HALT, 0))
    return instrs


def generate_dataset(n_samples: int, max_prog_len: int = 12,
                     max_push_val: int = 50) -> List[List[int]]:
    """Generate encoded token sequences from random programs."""
    executor = ReferenceExecutor()
    sequences = []
    
    for _ in range(n_samples):
        prog = random_program(max_len=max_prog_len, max_push_val=max_push_val)
        trace = executor.execute(prog)
        raw_tokens = trace.to_token_sequence()
        encoded = [encode_token(t) for t in raw_tokens]
        
        # Sanity: all values should be within vocab
        for t in encoded:
            assert 0 <= t < VOCAB_SIZE, f"Token {t} out of range"
        
        sequences.append(encoded)
    
    return sequences


# ─── Dataset ──────────────────────────────────────────────────────

class TraceDataset(Dataset):
    """Padded sequences for next-token prediction."""
    
    def __init__(self, sequences: List[List[int]], max_len: int = None):
        if max_len is None:
            max_len = max(len(s) for s in sequences)
        self.max_len = max_len
        
        # Pad sequences
        self.data = []
        for seq in sequences:
            if len(seq) > max_len:
                seq = seq[:max_len]
            padded = seq + [PAD_TOKEN] * (max_len - len(seq))
            self.data.append(padded)
        
        self.data = torch.tensor(self.data, dtype=torch.long)
        # Lengths for masking
        self.lengths = torch.tensor([min(len(s), max_len) for s in sequences])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.lengths[idx]


# ─── Model ────────────────────────────────────────────────────────

class MicroTransformer(nn.Module):
    """Tiny decoder-only transformer for trace prediction.
    
    Deliberately small to test whether the minimal architecture
    from Phase 4 (4 heads, 2 layers) is sufficient.
    """
    
    def __init__(self, d_model: int = 32, n_heads: int = 4, 
                 n_layers: int = 2, d_ff: int = None, 
                 max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        
        if d_ff is None:
            d_ff = d_model * 4
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Embeddings
        self.token_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.ln_final = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, VOCAB_SIZE)
        
        # Count params
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len) -> logits: (batch, seq_len, vocab_size)"""
        B, T = x.shape
        
        # Embeddings
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(T, device=x.device))
        h = self.dropout(tok + pos)
        
        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        
        for layer in self.layers:
            h = layer(h, mask)
        
        h = self.ln_final(h)
        return self.output(h)


class TransformerBlock(nn.Module):
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
        # Self-attention with causal mask
        h = self.ln1(x)
        attn_out, self.attn_weights = self.attn(h, h, h, attn_mask=mask)
        x = x + self.dropout(attn_out)
        
        # FF
        x = x + self.ff(self.ln2(x))
        return x


# ─── Training ─────────────────────────────────────────────────────

def train_model(model: MicroTransformer, train_data: TraceDataset,
                val_data: TraceDataset, epochs: int = 100,
                lr: float = 3e-4, batch_size: int = 64,
                patience: int = 15, verbose: bool = True) -> Dict:
    """Train with early stopping on validation loss."""
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    
    best_val_loss = float('inf')
    best_state = None
    no_improve = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0
        for batch, lengths in train_loader:
            # Input: tokens[:-1], target: tokens[1:]
            inp = batch[:, :-1]
            tgt = batch[:, 1:]
            
            logits = model(inp)
            
            # Mask padding from loss
            mask = torch.zeros_like(tgt, dtype=torch.bool)
            for i, l in enumerate(lengths):
                mask[i, :l-1] = True  # -1 because target is shifted
            
            loss = F.cross_entropy(logits[mask], tgt[mask])
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        train_loss /= n_batches
        
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
        
        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"  epoch {epoch:3d}  train_loss={train_loss:.4f}  "
                  f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    return history


# ─── Evaluation ────────────────────────────────────────────────────

def evaluate_execution(model: MicroTransformer, test_progs: List[List[Instruction]],
                      verbose: bool = False) -> Dict:
    """Test whether the model can actually execute programs end-to-end.
    
    For each program:
    1. Encode the program prefix
    2. Auto-regressively generate the trace
    3. Compare with reference execution
    """
    executor = ReferenceExecutor()
    model.eval()
    
    results = {
        'total': len(test_progs),
        'perfect': 0,       # entire trace matches
        'final_correct': 0,  # final TOP value matches
        'token_errors': [],  # per-program token error count
        'examples': [],      # first few for display
    }
    
    with torch.no_grad():
        for prog in test_progs:
            # Reference trace
            ref_trace = executor.execute(prog)
            ref_tokens = ref_trace.to_token_sequence()
            ref_encoded = [encode_token(t) for t in ref_tokens]
            
            # Find where trace starts (after TRACE_START token)
            trace_start_tok = encode_token(TOK_TRACE_START)
            try:
                trace_start_idx = ref_encoded.index(trace_start_tok)
            except ValueError:
                continue
            
            # Feed program prefix, generate trace tokens
            prefix = ref_encoded[:trace_start_idx + 1]  # include TRACE_START
            generated = list(prefix)
            
            # Generate enough tokens for the full trace
            n_trace_tokens = len(ref_encoded) - len(prefix)
            
            for _ in range(n_trace_tokens + 4):  # small buffer
                inp = torch.tensor([generated], dtype=torch.long)
                logits = model(inp)
                next_tok = logits[0, -1].argmax().item()
                generated.append(next_tok)
                
                if len(generated) >= len(ref_encoded):
                    break
            
            # Compare
            gen_trace = generated[trace_start_idx + 1:]
            ref_trace_tokens = ref_encoded[trace_start_idx + 1:]
            
            # Truncate to same length
            min_len = min(len(gen_trace), len(ref_trace_tokens))
            gen_trace = gen_trace[:min_len]
            ref_trace_tokens = ref_trace_tokens[:min_len]
            
            errors = sum(1 for g, r in zip(gen_trace, ref_trace_tokens) if g != r)
            results['token_errors'].append(errors)
            
            if errors == 0:
                results['perfect'] += 1
            
            # Check final TOP value
            if len(gen_trace) >= TOKENS_PER_STEP:
                gen_top = decode_token(gen_trace[-1])
                ref_top = decode_token(ref_trace_tokens[-1])
                if gen_top == ref_top:
                    results['final_correct'] += 1
            
            if verbose and len(results['examples']) < 5:
                prog_str = ' ; '.join(str(i) for i in prog)
                results['examples'].append({
                    'program': prog_str,
                    'ref_trace': [decode_token(t) for t in ref_trace_tokens],
                    'gen_trace': [decode_token(t) for t in gen_trace],
                    'errors': errors,
                })
    
    return results


# ─── Attention Analysis ───────────────────────────────────────────

def analyze_attention(model: MicroTransformer, sample_prog: List[Instruction]):
    """Extract and analyze attention patterns on a sample program."""
    executor = ReferenceExecutor()
    trace = executor.execute(sample_prog)
    tokens = trace.to_token_sequence()
    encoded = [encode_token(t) for t in tokens]
    
    inp = torch.tensor([encoded], dtype=torch.long)
    model.eval()
    
    with torch.no_grad():
        _ = model(inp)
    
    # Collect attention weights from each layer
    patterns = {}
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'attn_weights') and layer.attn_weights is not None:
            w = layer.attn_weights[0].cpu().numpy()  # (n_heads, seq, seq)
            patterns[f'layer_{i}'] = w
    
    return patterns, tokens, encoded


# ─── Main Experiment ───────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase 5: Trained Micro-Executor")
    print("=" * 60)
    print()
    
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # ── Data Generation ──
    print("Generating training data...")
    t0 = time.time()
    
    train_seqs = generate_dataset(2000, max_prog_len=10, max_push_val=50)
    val_seqs = generate_dataset(200, max_prog_len=10, max_push_val=50)
    
    # Separate test programs (not just sequences — we need the programs for execution test)
    test_progs = [random_program(max_len=10, max_push_val=50) for _ in range(100)]
    # Also test on longer programs to check generalization
    test_progs_long = [random_program(min_len=10, max_len=20, max_push_val=80) for _ in range(50)]
    
    max_len = max(max(len(s) for s in train_seqs), max(len(s) for s in val_seqs))
    print(f"  {len(train_seqs)} train, {len(val_seqs)} val sequences")
    print(f"  Max sequence length: {max_len}")
    print(f"  Vocab size: {VOCAB_SIZE}")
    print(f"  Generated in {time.time()-t0:.1f}s")
    
    train_data = TraceDataset(train_seqs, max_len=max_len)
    val_data = TraceDataset(val_seqs, max_len=max_len)
    
    # ── Experiment 1: Minimal model (matches Phase 4 head decomposition) ──
    print()
    print("─" * 50)
    print("Experiment 1: Minimal (d=32, heads=4, layers=2)")
    print("─" * 50)
    
    model_small = MicroTransformer(d_model=32, n_heads=4, n_layers=2, max_len=max_len*2)
    print(f"  Parameters: {model_small.n_params:,}")
    
    t0 = time.time()
    hist1 = train_model(model_small, train_data, val_data, epochs=80, lr=5e-4, patience=15)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Final val accuracy: {hist1['val_acc'][-1]:.4f}")
    
    # Execution test
    print("\n  Execution test (in-distribution):")
    res1 = evaluate_execution(model_small, test_progs, verbose=True)
    print(f"    Perfect traces: {res1['perfect']}/{res1['total']} "
          f"({100*res1['perfect']/res1['total']:.1f}%)")
    print(f"    Final value correct: {res1['final_correct']}/{res1['total']} "
          f"({100*res1['final_correct']/res1['total']:.1f}%)")
    avg_errs = np.mean(res1['token_errors']) if res1['token_errors'] else 0
    print(f"    Avg token errors per program: {avg_errs:.2f}")
    
    if res1['examples']:
        print("\n  Sample traces:")
        for ex in res1['examples'][:3]:
            print(f"    {ex['program']}")
            print(f"      ref: {ex['ref_trace'][:20]}...")
            print(f"      gen: {ex['gen_trace'][:20]}...")
            print(f"      errors: {ex['errors']}")
    
    print("\n  Execution test (out-of-distribution, longer programs):")
    res1_long = evaluate_execution(model_small, test_progs_long)
    print(f"    Perfect traces: {res1_long['perfect']}/{res1_long['total']} "
          f"({100*res1_long['perfect']/res1_long['total']:.1f}%)")
    print(f"    Final value correct: {res1_long['final_correct']}/{res1_long['total']} "
          f"({100*res1_long['final_correct']/res1_long['total']:.1f}%)")
    
    # ── Experiment 2: Deeper model ──
    print()
    print("─" * 50)
    print("Experiment 2: Deeper (d=32, heads=4, layers=4)")
    print("─" * 50)
    
    model_deep = MicroTransformer(d_model=32, n_heads=4, n_layers=4, max_len=max_len*2)
    print(f"  Parameters: {model_deep.n_params:,}")
    
    t0 = time.time()
    hist2 = train_model(model_deep, train_data, val_data, epochs=80, lr=5e-4, patience=15)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Final val accuracy: {hist2['val_acc'][-1]:.4f}")
    
    res2 = evaluate_execution(model_deep, test_progs)
    print(f"\n  Execution test (in-distribution):")
    print(f"    Perfect traces: {res2['perfect']}/{res2['total']} "
          f"({100*res2['perfect']/res2['total']:.1f}%)")
    print(f"    Final value correct: {res2['final_correct']}/{res2['total']} "
          f"({100*res2['final_correct']/res2['total']:.1f}%)")
    
    res2_long = evaluate_execution(model_deep, test_progs_long)
    print(f"\n  Execution test (OOD longer programs):")
    print(f"    Perfect traces: {res2_long['perfect']}/{res2_long['total']} "
          f"({100*res2_long['perfect']/res2_long['total']:.1f}%)")
    
    # ── Experiment 3: Wider model ──
    print()
    print("─" * 50)
    print("Experiment 3: Wider (d=64, heads=4, layers=2)")
    print("─" * 50)
    
    model_wide = MicroTransformer(d_model=64, n_heads=4, n_layers=2, max_len=max_len*2)
    print(f"  Parameters: {model_wide.n_params:,}")
    
    t0 = time.time()
    hist3 = train_model(model_wide, train_data, val_data, epochs=80, lr=5e-4, patience=15)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")
    print(f"  Final val accuracy: {hist3['val_acc'][-1]:.4f}")
    
    res3 = evaluate_execution(model_wide, test_progs)
    print(f"\n  Execution test (in-distribution):")
    print(f"    Perfect traces: {res3['perfect']}/{res3['total']} "
          f"({100*res3['perfect']/res3['total']:.1f}%)")
    print(f"    Final value correct: {res3['final_correct']}/{res3['total']} "
          f"({100*res3['final_correct']/res3['total']:.1f}%)")
    
    res3_long = evaluate_execution(model_wide, test_progs_long)
    print(f"\n  Execution test (OOD longer programs):")
    print(f"    Perfect traces: {res3_long['perfect']}/{res3_long['total']} "
          f"({100*res3_long['perfect']/res3_long['total']:.1f}%)")
    
    # ── Attention Pattern Analysis ──
    print()
    print("─" * 50)
    print("Attention Pattern Analysis (best model)")
    print("─" * 50)
    
    # Use whichever model performed best
    best_model = model_small
    best_name = "minimal"
    best_perfect = res1['perfect']
    if res2['perfect'] > best_perfect:
        best_model, best_name, best_perfect = model_deep, "deep", res2['perfect']
    if res3['perfect'] > best_perfect:
        best_model, best_name, best_perfect = model_wide, "wide", res3['perfect']
    
    print(f"  Analyzing {best_name} model ({best_perfect}/{res1['total']} perfect)")
    
    sample_prog = [Instruction(OP_PUSH, 10), Instruction(OP_PUSH, 20),
                   Instruction(OP_ADD, 0), Instruction(OP_HALT, 0)]
    
    patterns, raw_tokens, encoded_tokens = analyze_attention(best_model, sample_prog)
    
    print(f"\n  Program: {' ; '.join(str(i) for i in sample_prog)}")
    print(f"  Token sequence ({len(raw_tokens)} tokens)")
    
    for layer_name, weights in patterns.items():
        print(f"\n  {layer_name}:")
        # For each head, describe what it attends to
        n_heads = weights.shape[0] if len(weights.shape) == 3 else 1
        if len(weights.shape) == 2:
            weights = weights[np.newaxis, :, :]
        
        for head in range(min(n_heads, 4)):
            # Look at attention from trace positions (the interesting part)
            # Find where trace starts
            trace_start = SPECIAL_OFFSET + 2  # TRACE_START
            trace_idx = None
            for i, t in enumerate(encoded_tokens):
                if t == trace_start:
                    trace_idx = i + 1
                    break
            
            if trace_idx and trace_idx < weights.shape[1]:
                # Average attention from trace tokens
                trace_attn = weights[head, trace_idx:, :].mean(axis=0)
                # Where does this head look?
                top5 = np.argsort(trace_attn)[-5:][::-1]
                print(f"    head {head}: top positions = {list(top5)}, "
                      f"weights = {[f'{trace_attn[p]:.3f}' for p in top5]}")
    
    # ── Summary ──
    print()
    print("=" * 60)
    print("Phase 5 Summary")
    print("=" * 60)
    print()
    print(f"{'Model':<20} {'Params':>8} {'Val Acc':>8} {'Perfect':>10} {'Final OK':>10} {'OOD Perfect':>12}")
    print("-" * 70)
    
    configs = [
        ("minimal (32/4/2)", model_small.n_params, hist1, res1, res1_long),
        ("deep (32/4/4)", model_deep.n_params, hist2, res2, res2_long),
        ("wide (64/4/2)", model_wide.n_params, hist3, res3, res3_long),
    ]
    
    for name, params, hist, res, res_long in configs:
        va = hist['val_acc'][-1]
        pf = f"{res['perfect']}/{res['total']}"
        fc = f"{res['final_correct']}/{res['total']}"
        ood = f"{res_long['perfect']}/{res_long['total']}"
        print(f"{name:<20} {params:>8,} {va:>8.4f} {pf:>10} {fc:>10} {ood:>12}")
    
    print()
    print("Key questions:")
    print("  1. Can a tiny transformer learn perfect execution?")
    if any(r['perfect'] == r['total'] for r in [res1, res2, res3]):
        print("     → YES. At least one model achieved 100% perfect traces.")
    elif any(r['perfect'] / r['total'] > 0.8 for r in [res1, res2, res3]):
        print("     → MOSTLY. >80% perfect but not 100%. More data/capacity might close the gap.")
    else:
        best_pct = max(r['perfect']/r['total'] for r in [res1, res2, res3])
        print(f"     → PARTIALLY. Best: {best_pct:.1%}. The model learns significant structure but not perfect execution.")
    
    print("  2. Does depth or width matter more?")
    perfs = [res1['perfect'], res2['perfect'], res3['perfect']]
    if perfs[1] > perfs[0] and perfs[1] > perfs[2]:
        print("     → DEPTH wins. The deeper model outperforms, suggesting multi-step")
        print("       reasoning (opcode → lookup → compute → emit) benefits from layers.")
    elif perfs[2] > perfs[0] and perfs[2] > perfs[1]:
        print("     → WIDTH wins. The wider model outperforms, suggesting the FF routing")
        print("       for opcode-dependent logic needs more capacity per layer.")
    else:
        print("     → MINIMAL is sufficient (or all models hit the same ceiling).")
    
    return {
        'models': {'small': model_small, 'deep': model_deep, 'wide': model_wide},
        'histories': [hist1, hist2, hist3],
        'results': [res1, res2, res3],
        'results_ood': [res1_long, res2_long, res3_long],
    }


if __name__ == "__main__":
    results = main()

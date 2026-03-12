"""
Phase 6: Curriculum Learning for Stack Machine Execution

Hypothesis: Phase 5's 56% accuracy gap exists because the model must
simultaneously learn state tracking AND arithmetic. Decompose via curriculum:

  Stage 1: PUSH + HALT only           → learn token structure, SP tracking
  Stage 2: PUSH + POP + DUP + HALT    → learn stack recall (non-arithmetic)
  Stage 3: Full instruction set        → learn arithmetic (ADD)

Each stage initializes from the previous stage's best checkpoint.
Compare final Stage 3 accuracy against Phase 5's 56% baseline.

Container constraints: ~200s timeout per bash call.
Protocol: train in chunks, checkpoint between calls, push results to GitHub.
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
from typing import List, Dict, Optional, Set
from dataclasses import dataclass, asdict

# Import Phase 4 stack machine + Phase 5 model/encoding
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase4_stack_machine import (
    program, Instruction, ReferenceExecutor, Trace, TraceStep,
    OP_PUSH, OP_POP, OP_ADD, OP_DUP, OP_HALT, OP_NAMES,
    TOK_PROG_START, TOK_PROG_END, TOK_TRACE_START,
    TOKENS_PER_STEP
)
from phase5_training import (
    MicroTransformer, TraceDataset,
    encode_token, decode_token, encode_trace,
    evaluate_execution,
    VOCAB_SIZE, PAD_TOKEN, MAX_VAL
)


# ─── Constrained Program Generation ──────────────────────────────

def constrained_random_program(
    allowed_ops: Set[int],
    min_len: int = 3,
    max_len: int = 8,
    max_push_val: int = 30
) -> List[Instruction]:
    """Generate a random valid program using only allowed opcodes.
    
    Always includes HALT at the end.
    Respects stack depth constraints (e.g., POP needs depth >= 1).
    """
    instrs = []
    stack_depth = 0
    target_len = random.randint(min_len, max_len)
    
    for _ in range(target_len - 1):  # -1 for HALT
        valid = []
        if OP_PUSH in allowed_ops:
            valid.append(OP_PUSH)
        if OP_POP in allowed_ops and stack_depth >= 1:
            valid.append(OP_POP)
        if OP_DUP in allowed_ops and stack_depth >= 1:
            valid.append(OP_DUP)
        if OP_ADD in allowed_ops and stack_depth >= 2:
            valid.append(OP_ADD)
        
        if not valid:
            valid = [OP_PUSH]  # fallback: always can push
        
        op = random.choice(valid)
        
        if op == OP_PUSH:
            arg = random.randint(0, max_push_val)
            instrs.append(Instruction(OP_PUSH, arg))
            stack_depth += 1
        elif op == OP_POP:
            instrs.append(Instruction(OP_POP, 0))
            stack_depth -= 1
        elif op == OP_DUP:
            instrs.append(Instruction(OP_DUP, 0))
            stack_depth += 1
        elif op == OP_ADD:
            instrs.append(Instruction(OP_ADD, 0))
            stack_depth -= 1  # pops 2, pushes 1
    
    instrs.append(Instruction(OP_HALT, 0))
    return instrs


def generate_stage_data(
    allowed_ops: Set[int],
    n_samples: int,
    min_len: int = 3,
    max_len: int = 8,
    max_push_val: int = 30
) -> List[List[int]]:
    """Generate encoded trace sequences for a curriculum stage."""
    executor = ReferenceExecutor()
    seqs = []
    attempts = 0
    max_attempts = n_samples * 5
    
    while len(seqs) < n_samples and attempts < max_attempts:
        attempts += 1
        prog = constrained_random_program(allowed_ops, min_len, max_len, max_push_val)
        try:
            trace = executor.execute(prog)
            tokens = trace.to_token_sequence()
            encoded = [encode_token(t) for t in tokens]
            # Sanity: all values within vocab
            if all(0 <= t < VOCAB_SIZE for t in encoded):
                seqs.append(encoded)
        except Exception:
            continue  # invalid program, skip
    
    return seqs


# ─── Checkpoint Management ────────────────────────────────────────

@dataclass
class CheckpointMeta:
    stage: int
    epoch: int
    best_val_acc: float
    best_val_loss: float
    total_epochs_trained: int
    wall_time_s: float
    history: Dict  # train_loss, val_loss, val_acc lists

    def to_dict(self):
        return asdict(self)


def save_checkpoint(
    model: MicroTransformer,
    optimizer,
    meta: CheckpointMeta,
    path: str
):
    """Save model + optimizer + metadata for resume."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'meta': meta.to_dict(),
    }, path)


def load_checkpoint(path: str, model: MicroTransformer, optimizer=None):
    """Load checkpoint. Returns meta dict."""
    ckpt = torch.load(path, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['meta']


# ─── Training (single stage, resumable) ──────────────────────────

def train_stage(
    model: MicroTransformer,
    train_data: TraceDataset,
    val_data: TraceDataset,
    stage: int,
    max_epochs: int = 80,
    lr: float = 3e-4,
    batch_size: int = 64,
    patience: int = 20,
    max_wall_time: float = 170.0,  # stop before 200s container limit
    checkpoint_dir: str = ".",
    resume: bool = True,
    verbose: bool = True
) -> CheckpointMeta:
    """Train one curriculum stage with wall-clock safety and checkpointing."""
    
    ckpt_path = os.path.join(checkpoint_dir, f"phase6_ckpt_stage{stage}.pt")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Resume from checkpoint if exists
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
        # Wall-clock safety
        elapsed = time.time() - wall_start
        if elapsed > max_wall_time:
            if verbose:
                print(f"  Wall-clock limit ({max_wall_time:.0f}s) reached at epoch {epoch}")
            break
        
        # Train
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
        
        # Track best
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
    
    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)
    
    # Save checkpoint
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


# ─── Stage Definitions ────────────────────────────────────────────

STAGES = {
    1: {
        'name': 'PUSH + HALT',
        'ops': {OP_PUSH, OP_HALT},
        'description': 'Trivial: learn token structure, SP always increments, TOP = last push',
        'target_acc': 0.95,
        'max_push_val': 50,
        'max_epochs': 60,
        'patience': 15,
    },
    2: {
        'name': 'PUSH + POP + DUP + HALT',
        'ops': {OP_PUSH, OP_POP, OP_DUP, OP_HALT},
        'description': 'Non-arithmetic: SP can decrement, TOP requires stack recall',
        'target_acc': 0.85,
        'max_push_val': 50,
        'max_epochs': 80,
        'patience': 20,
    },
    3: {
        'name': 'Full instruction set',
        'ops': {OP_PUSH, OP_POP, OP_ADD, OP_DUP, OP_HALT},
        'description': 'Full: ADD requires reading two values + computing sum',
        'target_acc': 0.70,
        'max_push_val': 30,  # keep sums in range
        'max_epochs': 100,
        'patience': 25,
    },
}


# ─── Main Runner ──────────────────────────────────────────────────

def run_stage(
    stage: int,
    model: MicroTransformer,
    n_train: int = 1000,
    n_val: int = 150,
    n_test: int = 50,
    checkpoint_dir: str = ".",
    verbose: bool = True,
) -> Dict:
    """Run a single curriculum stage. Returns results dict."""
    
    cfg = STAGES[stage]
    if verbose:
        print(f"\n{'='*60}")
        print(f"Stage {stage}: {cfg['name']}")
        print(f"  {cfg['description']}")
        print(f"  Target: >{cfg['target_acc']*100:.0f}% token accuracy")
        print(f"{'='*60}\n")
    
    random.seed(42 + stage)
    np.random.seed(42 + stage)
    torch.manual_seed(42 + stage)
    
    # Generate data — exclude HALT from allowed_ops for generation
    # (constrained_random_program always appends HALT)
    gen_ops = cfg['ops'] - {OP_HALT}
    
    if verbose:
        print("  Generating data...")
    train_seqs = generate_stage_data(gen_ops, n_train, max_push_val=cfg['max_push_val'])
    val_seqs = generate_stage_data(gen_ops, n_val, max_push_val=cfg['max_push_val'])
    
    max_seq_len = max(
        max(len(s) for s in train_seqs),
        max(len(s) for s in val_seqs)
    )
    
    if verbose:
        print(f"  {len(train_seqs)} train, {len(val_seqs)} val, max_len={max_seq_len}")
    
    train_data = TraceDataset(train_seqs, max_len=max_seq_len)
    val_data = TraceDataset(val_seqs, max_len=max_seq_len)
    
    # Train
    meta = train_stage(
        model, train_data, val_data,
        stage=stage,
        max_epochs=cfg['max_epochs'],
        patience=cfg['patience'],
        checkpoint_dir=checkpoint_dir,
        verbose=verbose,
    )
    
    # Evaluate: generate test programs and run execution test
    if verbose:
        print(f"\n  Execution evaluation...")
    
    test_progs = [
        constrained_random_program(gen_ops, max_push_val=cfg['max_push_val'])
        for _ in range(n_test)
    ]
    exec_results = evaluate_execution(model, test_progs, verbose=verbose)
    
    if verbose:
        print(f"  Token accuracy (validation): {meta.best_val_acc:.4f}")
        print(f"  Perfect traces: {exec_results['perfect']}/{exec_results['total']}")
        print(f"  Final value correct: {exec_results['final_correct']}/{exec_results['total']}")
        target_met = "YES" if meta.best_val_acc >= cfg['target_acc'] else "NO"
        print(f"  Target met (>{cfg['target_acc']*100:.0f}%): {target_met}")
        
        if exec_results.get('examples'):
            print(f"\n  Sample traces:")
            for ex in exec_results['examples'][:3]:
                print(f"    {ex['program']}")
                print(f"      ref: {ex['ref_trace'][:16]}...")
                print(f"      gen: {ex['gen_trace'][:16]}...")
                print(f"      errors: {ex['errors']}")
    
    return {
        'stage': stage,
        'name': cfg['name'],
        'meta': meta.to_dict(),
        'execution': {
            'total': exec_results['total'],
            'perfect': exec_results['perfect'],
            'final_correct': exec_results['final_correct'],
            'avg_token_errors': float(np.mean(exec_results['token_errors'])) if exec_results['token_errors'] else 0,
        },
    }


def run_all_stages(checkpoint_dir: str = ".") -> Dict:
    """Run all curriculum stages sequentially."""
    
    print("Phase 6: Curriculum Learning for Stack Machine Execution")
    print(f"Model: d=64, heads=4, layers=2 (matching Phase 5 'wide' config)")
    print(f"Checkpoint dir: {checkpoint_dir}")
    print()
    
    # Create model — same as Phase 5's best "wide" config
    model = MicroTransformer(d_model=64, n_heads=4, n_layers=2, max_len=200)
    print(f"Parameters: {model.n_params:,}")
    
    all_results = {}
    
    for stage in [1, 2, 3]:
        # Load previous stage's checkpoint as starting weights
        if stage > 1:
            prev_ckpt = os.path.join(checkpoint_dir, f"phase6_ckpt_stage{stage-1}.pt")
            if os.path.exists(prev_ckpt):
                print(f"\n  Loading Stage {stage-1} weights as initialization...")
                load_checkpoint(prev_ckpt, model)
            else:
                print(f"\n  WARNING: No Stage {stage-1} checkpoint found. Training from scratch.")
        
        results = run_stage(stage, model, checkpoint_dir=checkpoint_dir)
        all_results[f'stage_{stage}'] = results
        
        # Save cumulative results JSON
        results_path = os.path.join(checkpoint_dir, "phase6_results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved to {results_path}")
    
    # Final comparison
    print(f"\n{'='*60}")
    print("Curriculum Learning Summary")
    print(f"{'='*60}")
    print(f"\n{'Stage':<30} {'Val Acc':>8} {'Perfect':>10} {'Final OK':>10}")
    print("-" * 60)
    for key, res in all_results.items():
        name = res['name']
        acc = res['meta']['best_val_acc']
        pf = f"{res['execution']['perfect']}/{res['execution']['total']}"
        fc = f"{res['execution']['final_correct']}/{res['execution']['total']}"
        print(f"{name:<30} {acc:>8.4f} {pf:>10} {fc:>10}")
    
    print(f"\nPhase 5 baseline: 56% val acc, 0/50 perfect, 5/50 final correct")
    s3 = all_results.get('stage_3', {})
    if s3:
        s3_acc = s3['meta']['best_val_acc']
        improvement = s3_acc - 0.56
        print(f"Stage 3 vs baseline: {s3_acc:.4f} vs 0.56 ({improvement:+.4f})")
        if s3_acc > 0.56:
            print("→ CURRICULUM LEARNING HELPS")
        else:
            print("→ Curriculum did not improve over baseline. Bottleneck is likely model capacity.")
    
    return all_results


# ─── Entry Point ──────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_all_stages(checkpoint_dir=".")

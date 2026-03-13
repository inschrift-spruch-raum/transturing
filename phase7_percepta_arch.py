"""
Phase 7: Percepta Architecture Replication

The gap in Phases 5-6: we never tested Percepta's actual architecture.
Their config: d_model=36, n_heads=18, n_layers=7, head_dim=2.

This matters because:
  - head_dim=2 IS the 2D convex hull attention (Phases 1-4 validated the primitive)
  - 7 layers (vs our 2) allow multi-step retrieval: layer 1 fetches operand A,
    layer 2 fetches operand B, later layers compute
  - 18 heads provide many parallel retrieval slots without the per-head capacity
    tradeoff that killed the 8-head experiment in Phase 6

Uses curriculum learning (Phase 6's approach) with 5K samples per stage.
Also runs the ADD diagnostic to test the 3% wall.
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
from typing import List, Dict, Set
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
    encode_token, decode_token,
    evaluate_execution,
    VOCAB_SIZE, PAD_TOKEN, MAX_VAL
)
from phase6_curriculum import (
    constrained_random_program, generate_stage_data,
    CheckpointMeta, save_checkpoint, load_checkpoint,
    STAGES
)


# ─── Percepta Architecture Config ────────────────────────────────

PERCEPTA_CONFIG = {
    'd_model': 36,
    'n_heads': 18,
    'n_layers': 7,
    'd_ff': 144,       # 36 * 4
    'max_len': 200,
    'dropout': 0.1,
}

# For comparison, also run our Phase 6 config
PHASE6_CONFIG = {
    'd_model': 64,
    'n_heads': 4,
    'n_layers': 2,
    'd_ff': 256,       # 64 * 4
    'max_len': 200,
    'dropout': 0.1,
}


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
    max_wall_time: float = 500.0,
    checkpoint_prefix: str = "phase7",
    checkpoint_dir: str = ".",
    resume: bool = True,
    verbose: bool = True
) -> CheckpointMeta:
    """Train one curriculum stage with wall-clock safety and checkpointing."""

    ckpt_path = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_ckpt_stage{stage}.pt")

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


# ─── ADD Diagnostic ──────────────────────────────────────────────

def run_add_diagnostic(model: MicroTransformer, n_tests: int = 30,
                       verbose: bool = True) -> Dict:
    """Test the three ADD patterns that revealed the 3% wall in Phase 6.

    Pattern 1: PUSH a, DUP, ADD       → result = 2a  (one lookup + double)
    Pattern 2: PUSH a, PUSH a, ADD    → result = 2a  (same values, two lookups)
    Pattern 3: PUSH a, PUSH b, ADD    → result = a+b (different values, two lookups)
    """
    executor = ReferenceExecutor()
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

        # Pattern 1: PUSH a, DUP, ADD
        prog1 = [Instruction(OP_PUSH, a), Instruction(OP_DUP, 0),
                 Instruction(OP_ADD, 0), Instruction(OP_HALT, 0)]
        patterns['DUP+ADD (PUSH a, DUP, ADD)'].append(prog1)

        # Pattern 2: PUSH a, PUSH a, ADD
        prog2 = [Instruction(OP_PUSH, a), Instruction(OP_PUSH, a),
                 Instruction(OP_ADD, 0), Instruction(OP_HALT, 0)]
        patterns['SAME+ADD (PUSH a, PUSH a, ADD)'].append(prog2)

        # Pattern 3: PUSH a, PUSH b, ADD (a != b)
        prog3 = [Instruction(OP_PUSH, a), Instruction(OP_PUSH, b),
                 Instruction(OP_ADD, 0), Instruction(OP_HALT, 0)]
        patterns['DIFF+ADD (PUSH a, PUSH b, ADD)'].append(prog3)

    if verbose:
        print("\n  ADD Diagnostic (the 3% wall test):")
        print(f"  {'Pattern':<45} {'Perfect':>10} {'Final OK':>10}")
        print("  " + "-" * 65)

    for name, progs in patterns.items():
        res = evaluate_execution(model, progs)
        results[name] = {
            'perfect': res['perfect'],
            'total': res['total'],
            'final_correct': res['final_correct'],
            'pct_perfect': res['perfect'] / res['total'] * 100,
        }
        if verbose:
            pf = f"{res['perfect']}/{res['total']}"
            fc = f"{res['final_correct']}/{res['total']}"
            print(f"  {name:<45} {pf:>10} {fc:>10}")

    return results


# ─── Stage Runner ────────────────────────────────────────────────

def run_stage(
    stage: int,
    model: MicroTransformer,
    n_train: int = 5000,
    n_val: int = 500,
    n_test: int = 50,
    checkpoint_prefix: str = "phase7",
    checkpoint_dir: str = ".",
    verbose: bool = True,
) -> Dict:
    """Run a single curriculum stage."""

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

    meta = train_stage(
        model, train_data, val_data,
        stage=stage,
        max_epochs=cfg['max_epochs'],
        patience=cfg['patience'],
        checkpoint_prefix=checkpoint_prefix,
        checkpoint_dir=checkpoint_dir,
        verbose=verbose,
    )

    # Evaluate
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


# ─── Main: Percepta Architecture Curriculum ──────────────────────

def run_percepta_curriculum(checkpoint_dir: str = ".") -> Dict:
    """Run full curriculum with Percepta's architecture."""

    print("=" * 60)
    print("Phase 7: Percepta Architecture Replication")
    print("=" * 60)
    print()
    print("Percepta config: d_model=36, n_heads=18, n_layers=7, head_dim=2")
    print("Phase 6 config:  d_model=64, n_heads=4,  n_layers=2, head_dim=16")
    print()

    cfg = PERCEPTA_CONFIG
    model = MicroTransformer(
        d_model=cfg['d_model'],
        n_heads=cfg['n_heads'],
        n_layers=cfg['n_layers'],
        d_ff=cfg['d_ff'],
        max_len=cfg['max_len'],
        dropout=cfg['dropout'],
    )
    print(f"Percepta model parameters: {model.n_params:,}")
    print(f"  d_model={cfg['d_model']}, n_heads={cfg['n_heads']}, "
          f"n_layers={cfg['n_layers']}, head_dim={cfg['d_model']//cfg['n_heads']}, "
          f"d_ff={cfg['d_ff']}")
    print()

    all_results = {'config': cfg, 'config_name': 'percepta'}
    total_start = time.time()

    for stage in [1, 2, 3]:
        if stage > 1:
            prev_ckpt = os.path.join(checkpoint_dir, f"phase7_ckpt_stage{stage-1}.pt")
            if os.path.exists(prev_ckpt):
                print(f"\n  Loading Stage {stage-1} weights as initialization...")
                load_checkpoint(prev_ckpt, model)
            else:
                print(f"\n  WARNING: No Stage {stage-1} checkpoint found. Training from scratch.")

        results = run_stage(
            stage, model,
            n_train=5000, n_val=500, n_test=50,
            checkpoint_prefix="phase7",
            checkpoint_dir=checkpoint_dir,
        )
        all_results[f'stage_{stage}'] = results

        # Save after each stage
        results_path = os.path.join(checkpoint_dir, "phase7_results.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  Results saved to {results_path}")

    # ADD diagnostic on final model
    print(f"\n{'='*60}")
    print("ADD Diagnostic (post Stage 3)")
    print(f"{'='*60}")
    add_results = run_add_diagnostic(model, n_tests=30, verbose=True)
    all_results['add_diagnostic'] = add_results

    total_time = time.time() - total_start

    # Summary
    print(f"\n{'='*60}")
    print("Phase 7 Summary: Percepta Architecture")
    print(f"{'='*60}")
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"\n{'Stage':<30} {'Val Acc':>8} {'Perfect':>10} {'Final OK':>10}")
    print("-" * 60)
    for key in ['stage_1', 'stage_2', 'stage_3']:
        if key in all_results:
            res = all_results[key]
            name = res['name']
            acc = res['meta']['best_val_acc']
            pf = f"{res['execution']['perfect']}/{res['execution']['total']}"
            fc = f"{res['execution']['final_correct']}/{res['execution']['total']}"
            print(f"{name:<30} {acc:>8.4f} {pf:>10} {fc:>10}")

    print(f"\nPhase 6 baseline (d=64, h=4, L=2, 5K samples):")
    print(f"  Stage 3: 85% val acc, 39/50 perfect, 44/50 final correct")
    print(f"  ADD diagnostic: DUP+ADD 97%, SAME+ADD 57%, DIFF+ADD 3%")

    s3 = all_results.get('stage_3', {})
    if s3:
        s3_acc = s3['meta']['best_val_acc']
        print(f"\nPercepta architecture Stage 3: {s3_acc:.4f} val acc")

        if 'add_diagnostic' in all_results:
            diff_add = all_results['add_diagnostic'].get('DIFF+ADD (PUSH a, PUSH b, ADD)', {})
            diff_pct = diff_add.get('pct_perfect', 0)
            print(f"  DIFF+ADD: {diff_pct:.0f}% perfect (Phase 6: 3%)")
            if diff_pct > 10:
                print("  → Percepta architecture BREAKS the 3% wall!")
            elif diff_pct > 3:
                print("  → Some improvement on DIFF+ADD, but wall not fully broken")
            else:
                print("  → 3% wall persists even with Percepta architecture")

    # Save final results
    all_results['total_time_s'] = total_time
    results_path = os.path.join(checkpoint_dir, "phase7_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFinal results saved to {results_path}")

    return all_results


# ─── Entry Point ──────────────────────────────────────────────────

if __name__ == "__main__":
    results = run_percepta_curriculum(checkpoint_dir=".")

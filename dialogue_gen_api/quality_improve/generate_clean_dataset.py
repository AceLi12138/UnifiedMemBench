#!/usr/bin/env python3
"""
Generate a clean dataset by removing QA tasks with incomplete evidence.

Functionality:
1. Load the dialogue dataset and coverage validation report.
2. Remove QA tasks with missing evidence while preserving the dialogue itself.
3. Export a clean dataset where each retained QA task has complete evidence.

Usage:
    python generate_clean_dataset.py
"""

import os
import json
import argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]


def generate_clean_dataset(
    dialogues_path: str,
    report_path: str,
    output_dir: str,
):
    """Generate a clean dialogue benchmark dataset."""
    
    print(f"Loading dialogue data: {dialogues_path}")
    with open(dialogues_path, 'r', encoding='utf-8') as f:
        all_dialogues = json.load(f)
    print(f"  Dialogues: {len(all_dialogues)}")

    print(f"Loading validation report: {report_path}")
    with open(report_path, 'r', encoding='utf-8') as f:
        report = json.load(f)

    # Collect all affected (dialogue_idx, task_idx) pairs.
    affected_tasks = set()
    all_tasks = set()

    for res in report['results']:
        didx = res['dialogue_idx']
        for d in res['details']:
            if d['final_status'] == 'SKIPPED':
                continue
            key = (didx, d['task_idx'])
            all_tasks.add(key)
            if d['final_status'] == 'MISSING':
                affected_tasks.add(key)

    print("\nStatistics:")
    print(f"  Total QA tasks: {len(all_tasks)}")
    print(f"  Affected QA tasks: {len(affected_tasks)} ({len(affected_tasks)/len(all_tasks)*100:.1f}%)")
    print(f"  Valid QA tasks: {len(all_tasks) - len(affected_tasks)} ({(len(all_tasks)-len(affected_tasks))/len(all_tasks)*100:.1f}%)")

    # Build the clean dataset.
    os.makedirs(output_dir, exist_ok=True)
    clean_dialogues = []

    total_original_tasks = 0
    total_clean_tasks = 0
    removed_tasks = 0
    tt_stats = {}  # task_type -> {original, clean}

    for didx, dialogue in enumerate(all_dialogues):
        tasks = dialogue.get('tasks_covered', [])
        total_original_tasks += len(tasks)

        # Filter affected tasks.
        clean_tasks = []
        for tidx, task in enumerate(tasks):
            tt = task.get('task_type', 'Unknown')
            tt_stats.setdefault(tt, {'original': 0, 'clean': 0})
            tt_stats[tt]['original'] += 1

            if (didx, tidx) not in affected_tasks:
                clean_tasks.append(task)
                tt_stats[tt]['clean'] += 1
                total_clean_tasks += 1
            else:
                removed_tasks += 1

        # Copy dialogue fields and replace tasks_covered.
        clean_dialogue = {
            'id': dialogue.get('id', ''),
            'character': dialogue.get('character', ''),
            'generation_mode': dialogue.get('generation_mode', ''),
            'tasks_covered': clean_tasks,
            'segment_outlines': dialogue.get('segment_outlines', []),
            'dialogue': dialogue.get('dialogue', []),
            'scene_config': dialogue.get('scene_config', {}),
            'statistics': dialogue.get('statistics', {}),
        }

        # Update statistics.
        if 'statistics' in clean_dialogue:
            clean_dialogue['statistics']['tasks_count'] = len(clean_tasks)
            clean_dialogue['statistics']['tasks_removed'] = len(tasks) - len(clean_tasks)

        # Keep only dialogues with at least one QA task.
        if clean_tasks:
            clean_dialogues.append(clean_dialogue)

    # Save the clean dataset.
    output_path = os.path.join(output_dir, 'UMB_dialogue_benchmark.json')
    print(f"\nSaving clean dataset: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(clean_dialogues, f, indent=2, ensure_ascii=False)

    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  File size: {file_size:.0f} MB")

    # Save summary statistics.
    summary = {
        'source_dialogues': dialogues_path,
        'validation_report': report_path,
        'dialogue_counts': {
            'raw': len(all_dialogues),
            'retained': len(clean_dialogues),
            'removed_no_valid_qa': len(all_dialogues) - len(clean_dialogues),
        },
        'qa_task_counts': {
            'raw': total_original_tasks,
            'valid_retained': total_clean_tasks,
            'removed_incomplete_evidence': removed_tasks,
            'valid_rate': f"{total_clean_tasks/total_original_tasks*100:.1f}%",
        },
        'by_task_type': {},
    }
    for tt in sorted(tt_stats.keys()):
        s = tt_stats[tt]
        summary['by_task_type'][tt] = {
            'raw': s['original'],
            'valid': s['clean'],
            'removed': s['original'] - s['clean'],
            'valid_rate': f"{s['clean']/s['original']*100:.1f}%",
        }

    summary_path = os.path.join(output_dir, 'dataset_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Print a run summary.
    print("\n" + "=" * 60)
    print("Clean dataset generation complete")
    print("=" * 60)
    print(f"  Dialogues: {len(clean_dialogues)}")
    print(f"  QA tasks: {total_clean_tasks}")
    print(f"  Removed QA tasks: {removed_tasks}")
    print()
    print("  By task type:")
    for tt in sorted(tt_stats.keys()):
        s = tt_stats[tt]
        print(f"    {tt:30s}: {s['clean']:5d} ({s['clean']/s['original']*100:.1f}%)")
    print()
    print(f"  Output file: {output_path}")
    print(f"  Summary file: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate a clean dataset by removing QA tasks with incomplete evidence")
    parser.add_argument("--dialogues", type=str,
                        default=str(REPO_ROOT / "dialogue_gen_api/output/final/repaired_dialogues_final.json"),
                        help="Dialogue dataset to clean")
    parser.add_argument("--report", type=str,
                        default=str(REPO_ROOT / "dialogue_gen_api/output/final/reeval_report_final.json"),
                        help="Validation report")
    parser.add_argument("--output_dir", type=str,
                        default=str(REPO_ROOT / "dialogue_gen_api/output/final/clean_v8_budget_direct"),
                        help="Output directory")
    args = parser.parse_args()

    generate_clean_dataset(
        dialogues_path=args.dialogues,
        report_path=args.report,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

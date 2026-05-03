from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from dialogue_training.memory_eval_utils import score_result_rows
from dialogue_training.run_local_memory_eval import _load_judge_llm_module
from dialogue_training.run_parallel_memory_eval import _load_compute_aggregate_scores


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def _dedupe_rows_by_id(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ordered: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        row_id = str(row.get("id", ""))
        if not row_id:
            continue
        ordered[row_id] = row
    return list(ordered.values())


def _load_existing_scored(path: Path) -> Tuple[List[Dict[str, Any]], set[str]]:
    if not path.exists():
        return [], set()
    rows = _dedupe_rows_by_id(_load_jsonl(path))
    return rows, {str(row["id"]) for row in rows if row.get("id")}


def _judge_api_failure(row: Dict[str, Any]) -> bool:
    judge_meta = row.get("judge_meta")
    if not isinstance(judge_meta, dict):
        return False
    failure_counts = judge_meta.get("failure_counts")
    if isinstance(failure_counts, dict) and int(failure_counts.get("api_exception", 0) or 0) > 0:
        return True
    failures = judge_meta.get("failures")
    if isinstance(failures, list):
        return any(isinstance(item, dict) and item.get("failure_type") == "api_exception" for item in failures)
    return False


def _write_state(
    path: Path,
    *,
    input_path: Path,
    output_path: Path,
    scores_path: Path,
    total_rows: int,
    completed_rows: int,
    pending_rows: int,
    last_id: Optional[str],
    api_failures_seen: int,
    done: bool,
) -> None:
    payload = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "scores_path": str(scores_path),
        "total_rows": total_rows,
        "completed_rows": completed_rows,
        "pending_rows": pending_rows,
        "last_id": last_id,
        "api_failures_seen": api_failures_seen,
        "done": done,
        "updated_at_unix": time.time(),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_scores(path: Path, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    aggregate = _load_compute_aggregate_scores()(rows)
    path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8")
    return aggregate


def _score_one_with_retries(
    row: Dict[str, Any],
    *,
    judge_llm: Any,
    row_retries: int,
    row_retry_sleep: float,
    fail_on_api_error: bool,
) -> Dict[str, Any]:
    last_scored: Optional[Dict[str, Any]] = None
    for attempt in range(1, int(row_retries) + 1):
        scored_rows, _ = score_result_rows([row], judge_llm=judge_llm)
        if not scored_rows:
            raise RuntimeError(f"Scoring returned no rows for id={row.get('id')}")
        scored = scored_rows[0]
        last_scored = scored
        if not fail_on_api_error or not _judge_api_failure(scored):
            return scored
        if attempt < row_retries:
            print(
                f"[WARN] judge API failure for id={row.get('id')} "
                f"attempt={attempt}/{row_retries}; sleep={row_retry_sleep}s",
                flush=True,
            )
            time.sleep(float(row_retry_sleep))

    raise RuntimeError(
        f"judge API failure persisted for id={row.get('id')} "
        f"after {row_retries} row-level attempts; last_judge_meta={last_scored.get('judge_meta') if last_scored else None}"
    )


def rescore_context_results(
    *,
    input_path: Path,
    output_dir: Path,
    judge_model: str,
    output_filename: str = "detailed_results.rescored.jsonl",
    scores_filename: str = "scores.rescored.json",
    state_filename: str = "rescore_state.json",
    flush_every: int = 1,
    scores_every: int = 100,
    row_retries: int = 2,
    row_retry_sleep: float = 30.0,
    workers: int = 1,
    fail_on_api_error: bool = True,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / output_filename
    scores_path = output_dir / scores_filename
    state_path = output_dir / state_filename

    input_rows = _load_jsonl(input_path)
    if limit is not None:
        input_rows = input_rows[: int(limit)]

    existing_rows, completed_ids = _load_existing_scored(output_path)
    pending_rows = [row for row in input_rows if str(row.get("id", "")) not in completed_ids]

    print(
        f"Loaded rows={len(input_rows)} from {input_path}. "
        f"Completed={len(completed_ids)}, pending={len(pending_rows)}.",
        flush=True,
    )
    if not pending_rows:
        aggregate = _write_scores(scores_path, existing_rows)
        _write_state(
            state_path,
            input_path=input_path,
            output_path=output_path,
            scores_path=scores_path,
            total_rows=len(input_rows),
            completed_rows=len(existing_rows),
            pending_rows=0,
            last_id=existing_rows[-1].get("id") if existing_rows else None,
            api_failures_seen=0,
            done=True,
        )
        return aggregate

    worker_count = max(1, int(workers))
    print(
        f"Scoring with workers={worker_count}, flush_every={flush_every}, "
        f"scores_every={scores_every}, row_retries={row_retries}.",
        flush=True,
    )
    judge_llm = _load_judge_llm_module().get_llm(judge_model, max_workers=worker_count)
    all_scored = list(existing_rows)
    api_failures_seen = 0

    def handle_scored(row_index: int, scored: Dict[str, Any], fh: Any) -> None:
        nonlocal api_failures_seen
        if _judge_api_failure(scored):
            api_failures_seen += 1
        fh.write(json.dumps(scored, ensure_ascii=False) + "\n")
        all_scored.append(scored)

        should_flush = row_index % max(1, int(flush_every)) == 0
        should_score = int(scores_every) > 0 and row_index % int(scores_every) == 0
        if should_flush:
            fh.flush()
            _write_state(
                state_path,
                input_path=input_path,
                output_path=output_path,
                scores_path=scores_path,
                total_rows=len(input_rows),
                completed_rows=len(all_scored),
                pending_rows=len(input_rows) - len(all_scored),
                last_id=str(scored.get("id")),
                api_failures_seen=api_failures_seen,
                done=False,
            )
            print(
                f"Progress {len(all_scored)}/{len(input_rows)} "
                f"(just scored id={scored.get('id')})",
                flush=True,
            )
        if should_score:
            _write_scores(scores_path, all_scored)

    with output_path.open("a", encoding="utf-8") as fh:
        if worker_count == 1:
            for index, row in enumerate(pending_rows, start=1):
                scored = _score_one_with_retries(
                    row,
                    judge_llm=judge_llm,
                    row_retries=row_retries,
                    row_retry_sleep=row_retry_sleep,
                    fail_on_api_error=fail_on_api_error,
                )
                handle_scored(index, scored, fh)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        _score_one_with_retries,
                        row,
                        judge_llm=judge_llm,
                        row_retries=row_retries,
                        row_retry_sleep=row_retry_sleep,
                        fail_on_api_error=fail_on_api_error,
                    ): row
                    for row in pending_rows
                }
                for index, future in enumerate(as_completed(futures), start=1):
                    handle_scored(index, future.result(), fh)

    aggregate = _write_scores(scores_path, all_scored)
    _write_state(
        state_path,
        input_path=input_path,
        output_path=output_path,
        scores_path=scores_path,
        total_rows=len(input_rows),
        completed_rows=len(all_scored),
        pending_rows=0,
        last_id=all_scored[-1].get("id") if all_scored else None,
        api_failures_seen=api_failures_seen,
        done=True,
    )
    print(f"Finished rescoring {len(all_scored)}/{len(input_rows)} rows.", flush=True)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Incrementally rescore existing context-eval detailed_results.jsonl without loading a local model."
    )
    parser.add_argument("--input_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--judge_model", type=str, default="mimo-v2-flash")
    parser.add_argument("--output_filename", type=str, default="detailed_results.rescored.jsonl")
    parser.add_argument("--scores_filename", type=str, default="scores.rescored.json")
    parser.add_argument("--state_filename", type=str, default="rescore_state.json")
    parser.add_argument("--flush_every", type=int, default=1)
    parser.add_argument("--scores_every", type=int, default=100)
    parser.add_argument("--row_retries", type=int, default=2)
    parser.add_argument("--row_retry_sleep", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--allow_api_failures",
        action="store_true",
        help="Write rows even when the judge reports API failures. By default, stop before writing such rows.",
    )
    args = parser.parse_args()

    aggregate = rescore_context_results(
        input_path=args.input_path,
        output_dir=args.output_dir,
        judge_model=args.judge_model,
        output_filename=args.output_filename,
        scores_filename=args.scores_filename,
        state_filename=args.state_filename,
        flush_every=args.flush_every,
        scores_every=args.scores_every,
        row_retries=args.row_retries,
        row_retry_sleep=args.row_retry_sleep,
        workers=args.workers,
        fail_on_api_error=not args.allow_api_failures,
        limit=args.limit,
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

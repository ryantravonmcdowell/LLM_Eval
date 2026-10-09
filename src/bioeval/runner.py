"""Run one model over a question set: generate, grade, store.

Model calls run in a thread pool (they are I/O-bound); grading and database
writes stay on the calling thread, so SQLite is only ever touched from one
thread. Each result is committed as it arrives, which is what makes
``resume_run_id`` safe after a crash or a rate-limit storm.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from bioeval import storage
from bioeval.adapters.base import (
    DEFAULT_SYSTEM_PROMPT,
    Adapter,
    AdapterConfigError,
    AdapterError,
    Completion,
    GenerationRequest,
)
from bioeval.dataset import DatasetReport
from bioeval.judge import JudgeParseError, LLMJudge
from bioeval.schema import Question
from bioeval.scoring.rubric import RubricWeights, score_response
from bioeval.scoring.taxonomy import ErrorType


@dataclass
class RunSummary:
    run_id: str
    adapter: str
    attempted: int
    skipped: int
    failed: int
    mean_score: float | None


def _call(adapter: Adapter, q: Question, system: str) -> Completion:
    return adapter.generate(GenerationRequest(question_id=q.id, prompt=q.full_prompt(),
                                              system=system))


def _graded_row(q: Question, completion: Completion, weights: RubricWeights,
                judge: LLMJudge | None) -> tuple[dict, list[str]]:
    card = score_response(q, completion.text, weights)
    details = {
        "violations": list(card.violations),
        "covered": list(card.facts.covered) if card.facts else [],
        "missing_facts": list(card.facts.missing_facts) if card.facts else [],
        "missing_reasoning": list(card.facts.missing_reasoning) if card.facts else [],
        "misconceptions": list(card.facts.misconceptions) if card.facts else [],
    }
    row = {
        "question_id": q.id,
        "response": completion.text,
        "model": completion.model,
        "latency_ms": completion.latency_ms,
        "input_tokens": completion.input_tokens,
        "output_tokens": completion.output_tokens,
        "stop_reason": completion.stop_reason,
        "score": card.score,
        "coverage": card.coverage,
        "instruction_ok": int(card.instruction_ok),
        "n_misconceptions": len(details["misconceptions"]),
    }
    if judge is not None and ErrorType.REFUSAL_OR_EMPTY not in card.errors:
        try:
            verdict = judge.grade(q, completion.text)
            row["judge_score"] = verdict.score
            row["judge_rationale"] = verdict.rationale
            details["judge_errors"] = list(verdict.errors)
        except (AdapterError, JudgeParseError) as exc:
            details["judge_failure"] = str(exc)
    row["details_json"] = json.dumps(details, sort_keys=True)
    return row, card.error_labels


def run_evaluation(
    db_path: str | Path,
    dataset: DatasetReport,
    adapter: Adapter,
    *,
    judge: LLMJudge | None = None,
    weights: RubricWeights | None = None,
    workers: int = 4,
    limit: int | None = None,
    resume_run_id: str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    progress: Callable[[str], None] | None = None,
) -> RunSummary:
    if not dataset.ok:
        raise ValueError("refusing to run on a dataset with errors; run `bioeval validate` first")
    weights = weights or RubricWeights()
    say = progress or (lambda msg: print(msg, file=sys.stderr))
    questions = dataset.questions[:limit] if limit else list(dataset.questions)

    conn = storage.connect(db_path)
    try:
        storage.register_questions(conn, dataset.sha256, dataset.questions)
        if resume_run_id:
            run = storage.get_run(conn, resume_run_id)
            if run is None:
                raise ValueError(f"no run {resume_run_id!r} in {db_path}")
            if run["dataset_sha256"] != dataset.sha256:
                raise ValueError("cannot resume: the dataset has changed since that run started")
            if run["adapter"] != adapter.name:
                raise ValueError(
                    f"cannot resume: run used {run['adapter']!r}, not {adapter.name!r}"
                )
            run_id = resume_run_id
        else:
            config = {
                "weights": asdict(weights),
                "judge": judge.name if judge else None,
                "limit": limit,
                "system_prompt": system_prompt,
            }
            run_id = storage.create_run(conn, adapter.name, dataset.path, dataset.sha256, config)

        done = storage.completed_question_ids(conn, run_id)
        todo = [q for q in questions if q.id not in done]
        say(f"[{adapter.name}] run {run_id}: {len(todo)} to answer, {len(done)} already done")

        failed = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_call, adapter, q, system_prompt): q for q in todo}
            try:
                for i, fut in enumerate(as_completed(futures), start=1):
                    q = futures[fut]
                    try:
                        completion = fut.result()
                    except AdapterConfigError:
                        raise
                    except AdapterError as exc:
                        failed += 1
                        storage.save_result(conn, run_id,
                                            {"question_id": q.id, "adapter_error": str(exc)},
                                            [ErrorType.ADAPTER_ERROR.value])
                        continue
                    row, errors = _graded_row(q, completion, weights, judge)
                    storage.save_result(conn, run_id, row, errors)
                    if i % 10 == 0 or i == len(todo):
                        say(f"[{adapter.name}] {i}/{len(todo)} graded")
            except AdapterConfigError:
                for f in futures:
                    f.cancel()
                storage.finish_run(conn, run_id, status="aborted")
                raise

        storage.finish_run(conn, run_id, status="complete" if not failed else "partial")
        mean = conn.execute(
            "SELECT AVG(score) FROM results WHERE run_id = ? AND score IS NOT NULL", (run_id,)
        ).fetchone()[0]
    finally:
        conn.close()

    return RunSummary(run_id=run_id, adapter=adapter.name, attempted=len(todo),
                      skipped=len(questions) - len(todo), failed=failed,
                      mean_score=round(mean, 4) if mean is not None else None)

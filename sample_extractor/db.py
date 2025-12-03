"""
SQLite persistence layer for EmbodiedBench step + score caching.

Tables:
  steps(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    task TEXT NOT NULL,
    subtask TEXT NOT NULL,
    model_name TEXT,
    episode INTEGER NOT NULL,
    step INTEGER NOT NULL,
    prompt TEXT,
    raw_output TEXT NOT NULL,
    parseable INTEGER NOT NULL,
    retries_before_success INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    log_path TEXT NOT NULL,
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(run_id, model_name, task, subtask, episode, step)
  )

  scores(
    key TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    model_name TEXT,
    episode INTEGER NOT NULL,
    step INTEGER NOT NULL,
    final_quality INTEGER,
    tightness INTEGER,
    task_relevance INTEGER,
    logical_progression INTEGER,
    fidelity INTEGER,
    efficiency INTEGER,
    insight INTEGER,
    structure INTEGER,
    robustness INTEGER,
    explanation TEXT,
    hallucination INTEGER,
    overly_verbose INTEGER,
    raw_output_len INTEGER,
    scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  )

  run_stats(
    run_id TEXT PRIMARY KEY,
    task_count INTEGER,
    model_count INTEGER,
    total_steps INTEGER,
    total_episodes INTEGER,
    last_scan TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  )

Indices:
  CREATE INDEX idx_steps_task_episode_step ON steps(task, episode, step);
  CREATE INDEX idx_steps_model_episode_step ON steps(model_name, episode, step);
  CREATE INDEX idx_scores_run_episode_step ON scores(run_id, episode, step);
  CREATE INDEX idx_scores_model_episode_step ON scores(model_name, episode, step);

Functions provided:
  get_conn(db_path) -> sqlite3.Connection
  create_schema(conn)
  upsert_step(...)
  upsert_score(...)
  fetch_missing_scores(conn) -> list of rows needing scoring
  update_run_stats(conn, run_id)
"""

from __future__ import annotations

import hashlib
import sqlite3
from typing import Iterable, Optional, Sequence, Tuple

def get_conn(db_path: str = "outputs/steps_cache.sqlite") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=OFF;")
    return conn

def create_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS steps(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          run_id TEXT NOT NULL,
          task TEXT NOT NULL,
          subtask TEXT NOT NULL,
          model_name TEXT,
          episode INTEGER NOT NULL,
          step INTEGER NOT NULL,
          prompt TEXT,
            raw_output TEXT NOT NULL,
          parseable INTEGER NOT NULL,
          retries_before_success INTEGER NOT NULL,
          content_hash TEXT NOT NULL,
          log_path TEXT NOT NULL,
          ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          UNIQUE(run_id, model_name, task, subtask, episode, step)
        );

        CREATE TABLE IF NOT EXISTS scores(
          key TEXT PRIMARY KEY,
          run_id TEXT NOT NULL,
          model_name TEXT,
          episode INTEGER NOT NULL,
          step INTEGER NOT NULL,
          final_quality INTEGER,
          tightness INTEGER,
          task_relevance INTEGER,
          logical_progression INTEGER,
          fidelity INTEGER,
          efficiency INTEGER,
          insight INTEGER,
          structure INTEGER,
          robustness INTEGER,
          explanation TEXT,
          hallucination INTEGER,
          overly_verbose INTEGER,
          raw_output_len INTEGER,
          scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS run_stats(
          run_id TEXT PRIMARY KEY,
          task_count INTEGER,
          model_count INTEGER,
          total_steps INTEGER,
          total_episodes INTEGER,
          last_scan TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS divergence_metrics(
          run_group TEXT NOT NULL,
          episode INTEGER NOT NULL,
          step INTEGER NOT NULL,
          model_count INTEGER NOT NULL,
          max_quality INTEGER,
          min_quality INTEGER,
          quality_range INTEGER,
          stddev_quality REAL,
          updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY(run_group, episode, step)
        );

        CREATE TABLE IF NOT EXISTS episode_outcomes(
          run_id TEXT NOT NULL,
          task TEXT NOT NULL,
          episode INTEGER NOT NULL,
          task_success REAL,
          task_progress REAL,
          num_invalid_actions INTEGER,
          num_steps INTEGER,
          instruction TEXT,
          ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          PRIMARY KEY(run_id, task, episode)
        );

        CREATE TABLE IF NOT EXISTS comparison_candidates(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          task TEXT NOT NULL,
          episode INTEGER NOT NULL,
          target_run_id TEXT NOT NULL,
          baseline_run_id TEXT NOT NULL,
          target_best_step INTEGER,
          target_best_quality INTEGER,
          baseline_worst_step INTEGER,
          baseline_worst_quality INTEGER,
          baseline_has_hallucination INTEGER,
          contrast_score REAL,
          created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
          UNIQUE(task, episode, target_run_id, baseline_run_id)
        );

        CREATE INDEX IF NOT EXISTS idx_steps_task_episode_step ON steps(task, episode, step);
        CREATE INDEX IF NOT EXISTS idx_steps_model_episode_step ON steps(model_name, episode, step);
        CREATE INDEX IF NOT EXISTS idx_steps_content_hash ON steps(content_hash);
        CREATE INDEX IF NOT EXISTS idx_scores_run_episode_step ON scores(run_id, episode, step);
        CREATE INDEX IF NOT EXISTS idx_scores_model_episode_step ON scores(model_name, episode, step);
        CREATE INDEX IF NOT EXISTS idx_divergence_run_group ON divergence_metrics(run_group, episode, step);
        CREATE INDEX IF NOT EXISTS idx_episode_outcomes_task ON episode_outcomes(task, episode);
        CREATE INDEX IF NOT EXISTS idx_comparison_candidates_contrast ON comparison_candidates(contrast_score DESC);
        """
    )
    conn.commit()

def _hash_raw(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def upsert_step(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    task: str,
    subtask: str,
    model_name: Optional[str],
    episode: int,
    step: int,
    prompt: Optional[str],
    raw_output: str,
    parseable: bool,
    retries_before_success: int,
    log_path: str,
) -> None:
    content_hash = _hash_raw(raw_output)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO steps(run_id, task, subtask, model_name, episode, step,
                          prompt, raw_output, parseable, retries_before_success,
                          content_hash, log_path)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(run_id, model_name, task, subtask, episode, step) DO UPDATE SET
          prompt=excluded.prompt,
          raw_output=excluded.raw_output,
          parseable=excluded.parseable,
          retries_before_success=excluded.retries_before_success,
          content_hash=excluded.content_hash,
          log_path=excluded.log_path
        """,
        (
            run_id,
            task,
            subtask,
            model_name,
            episode,
            step,
            prompt,
            raw_output,
            int(parseable),
            retries_before_success,
            content_hash,
            log_path,
        ),
    )

def upsert_score(
    conn: sqlite3.Connection,
    *,
    key: str,
    run_id: str,
    model_name: Optional[str],
    episode: int,
    step: int,
    final_quality: int,
    intermediate_scores: dict,
    explanation: str,
    flags: dict,
    raw_output_len: int,
) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO scores(
          key, run_id, model_name, episode, step,
          final_quality,
          tightness, task_relevance, logical_progression, fidelity,
          efficiency, insight, structure, robustness,
          explanation, hallucination, overly_verbose, raw_output_len
        )
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(key) DO UPDATE SET
          final_quality=excluded.final_quality,
          tightness=excluded.tightness,
          task_relevance=excluded.task_relevance,
          logical_progression=excluded.logical_progression,
          fidelity=excluded.fidelity,
          efficiency=excluded.efficiency,
          insight=excluded.insight,
          structure=excluded.structure,
          robustness=excluded.robustness,
          explanation=excluded.explanation,
          hallucination=excluded.hallucination,
          overly_verbose=excluded.overly_verbose,
          raw_output_len=excluded.raw_output_len
        """,
        (
            key,
            run_id,
            model_name,
            episode,
            step,
            final_quality,
            intermediate_scores.get("tightness"),
            intermediate_scores.get("task_relevance"),
            intermediate_scores.get("logical_progression"),
            intermediate_scores.get("fidelity"),
            intermediate_scores.get("efficiency"),
            intermediate_scores.get("insight"),
            intermediate_scores.get("structure"),
            intermediate_scores.get("robustness"),
            explanation,
            int(flags.get("possible_hallucination", False)),
            int(flags.get("overly_verbose", False)),
            raw_output_len,
        ),
    )

def fetch_missing_scores(conn: sqlite3.Connection, run_filter: Optional[Sequence[str]] = None) -> Iterable[Tuple]:
    cur = conn.cursor()
    if run_filter:
        q_marks = ",".join("?" for _ in run_filter)
        cur.execute(
            f"""
            SELECT s.run_id, s.model_name, s.task, s.subtask,
                   s.episode, s.step, s.prompt, s.raw_output,
                   s.parseable, s.retries_before_success
            FROM steps s
            LEFT JOIN scores sc ON sc.key = SUBSTR(s.content_hash,1,16)
            WHERE sc.key IS NULL AND s.run_id IN ({q_marks})
            ORDER BY s.episode, s.step, s.task, s.subtask, s.run_id
            """,
            tuple(run_filter),
        )
    else:
        cur.execute(
            """
            SELECT s.run_id, s.model_name, s.task, s.subtask,
                   s.episode, s.step, s.prompt, s.raw_output,
                   s.parseable, s.retries_before_success
            FROM steps s
            LEFT JOIN scores sc ON sc.key = SUBSTR(s.content_hash,1,16)
            WHERE sc.key IS NULL
            ORDER BY s.episode, s.step, s.task, s.subtask, s.run_id
            """
        )
    for row in cur.fetchall():
        yield row

def update_run_stats(conn: sqlite3.Connection, run_id: str) -> None:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(DISTINCT task) , COUNT(DISTINCT model_name),
               COUNT(*) , COUNT(DISTINCT episode)
        FROM steps WHERE run_id=?
        """,
        (run_id,),
    )
    task_count, model_count, total_steps, total_episodes = cur.fetchone()
    cur.execute(
        """
        INSERT INTO run_stats(run_id, task_count, model_count, total_steps, total_episodes)
        VALUES(?,?,?,?,?)
        ON CONFLICT(run_id) DO UPDATE SET
          task_count=excluded.task_count,
          model_count=excluded.model_count,
          total_steps=excluded.total_steps,
          total_episodes=excluded.total_episodes,
          last_scan=CURRENT_TIMESTAMP
        """,
        (run_id, task_count, model_count, total_steps, total_episodes),
    )

def compute_divergence(conn: sqlite3.Connection, task_filter: Optional[Sequence[str]] = None, episode_filter: Optional[Sequence[int]] = None) -> int:
    """
    Recompute divergence metrics (quality spread) across all scored models for each (task, episode, step).

    run_group = task|episode
    Returns number of rows written/updated.
    """
    cur = conn.cursor()
    params = []
    where = []
    if task_filter:
        where.append("s.task IN (%s)" % ",".join("?" for _ in task_filter))
        params.extend(task_filter)
    if episode_filter:
        where.append("sc.episode IN (%s)" % ",".join("?" for _ in episode_filter))
        params.extend(episode_filter)
    where_clause = ("WHERE " + " AND ".join(where)) if where else ""
    cur.execute(
        f"""
        SELECT s.task, sc.episode, sc.step, sc.final_quality
        FROM scores sc
        JOIN steps s
          ON sc.run_id = s.run_id
         AND sc.model_name = s.model_name
         AND sc.episode = s.episode
         AND sc.step = s.step
        {where_clause}
        ORDER BY s.task, sc.episode, sc.step
        """,
        tuple(params),
    )
    rows = cur.fetchall()
    # Aggregate
    from collections import defaultdict
    agg = defaultdict(list)
    for task, ep, step, fq in rows:
        agg[(task, ep, step)].append(fq if fq is not None else 0)
    written = 0
    for (task, ep, step), quals in agg.items():
        if not quals:
            continue
        model_count = len(quals)
        max_q = max(quals)
        min_q = min(quals)
        range_q = max_q - min_q
        mean = sum(quals) / model_count
        # population stddev
        import math
        stddev = math.sqrt(sum((q - mean) ** 2 for q in quals) / model_count) if model_count > 0 else 0.0
        run_group = f"{task}|{ep}"
        cur.execute(
            """
            INSERT INTO divergence_metrics(run_group, episode, step, model_count,
                                           max_quality, min_quality, quality_range, stddev_quality)
            VALUES(?,?,?,?,?,?,?,?)
            ON CONFLICT(run_group, episode, step) DO UPDATE SET
              model_count=excluded.model_count,
              max_quality=excluded.max_quality,
              min_quality=excluded.min_quality,
              quality_range=excluded.quality_range,
              stddev_quality=excluded.stddev_quality,
              updated_at=CURRENT_TIMESTAMP
            """,
            (run_group, ep, step, model_count, max_q, min_q, range_q, stddev),
        )
        written += 1
    conn.commit()
    return written

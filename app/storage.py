"""SQLite 持久化：机组群、调度结果、异步作业。

机组群修改后新建版本，旧调度结果保留其当时参数（快照）。
数据库文件落在挂载卷 /data（默认 ./data）。
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager

DB_PATH = os.environ.get("ED_DB_PATH", os.path.join(os.getcwd(), "data", "dispatch.db"))
_lock = threading.Lock()


def _init_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


_conn = _init_conn()


@contextmanager
def tx():
    with _lock:
        try:
            yield _conn
            _conn.commit()
        except Exception:
            _conn.rollback()
            raise


def init_db():
    with tx() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS fleets (
                fleet_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                version INTEGER NOT NULL,
                params TEXT NOT NULL,
                b_matrix TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS schedules (
                schedule_id TEXT PRIMARY KEY,
                fleet_id TEXT NOT NULL,
                fleet_version INTEGER NOT NULL,
                params_snapshot TEXT NOT NULL,
                loads TEXT NOT NULL,
                result TEXT NOT NULL,
                total_cost REAL,
                status TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                status TEXT NOT NULL,
                fleet_id TEXT,
                payload TEXT,
                result TEXT,
                error TEXT,
                progress INTEGER NOT NULL DEFAULT 0,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                cancel_flag INTEGER NOT NULL DEFAULT 0
            );
            """)


# ---------------- 机组群 ----------------

def create_fleet(name: str, params: list, b_matrix) -> str:
    fid = uuid.uuid4().hex
    now = time.time()
    with tx() as c:
        c.execute(
            "INSERT INTO fleets(fleet_id,name,version,params,b_matrix,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?,?)",
            (fid, name, 1, json.dumps(params, ensure_ascii=False),
             json.dumps(b_matrix) if b_matrix is not None else None, now, now))
    return fid


def get_fleet(fid: str):
    with tx() as c:
        row = c.execute("SELECT * FROM fleets WHERE fleet_id=?", (fid,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["params"] = json.loads(d["params"])
    d["b_matrix"] = json.loads(d["b_matrix"]) if d["b_matrix"] else None
    return d


def update_fleet(fid: str, params: list, b_matrix):
    with tx() as c:
        row = c.execute("SELECT version FROM fleets WHERE fleet_id=?", (fid,)).fetchone()
        if row is None:
            return None
        ver = row["version"] + 1
        c.execute(
            "UPDATE fleets SET version=?, params=?, b_matrix=?, updated_at=? WHERE fleet_id=?",
            (ver, json.dumps(params, ensure_ascii=False),
             json.dumps(b_matrix) if b_matrix is not None else None, time.time(), fid))
    return ver


# ---------------- 调度结果 ----------------

def save_schedule(sid, fid, version, params_snapshot, loads, result_json, total_cost):
    with tx() as c:
        c.execute(
            "INSERT INTO schedules(schedule_id,fleet_id,fleet_version,params_snapshot,"
            "loads,result,total_cost,status,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (sid, fid, version, json.dumps(params_snapshot, ensure_ascii=False),
             json.dumps(list(loads)), result_json, total_cost, "optimal", time.time()))
    return sid


def get_schedule(sid):
    with tx() as c:
        row = c.execute("SELECT * FROM schedules WHERE schedule_id=?", (sid,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["result"] = json.loads(d["result"])
    d["loads"] = json.loads(d["loads"])
    d["params_snapshot"] = json.loads(d["params_snapshot"])
    return d


# ---------------- 作业 ----------------

def create_job(kind: str, fleet_id, payload) -> str:
    jid = uuid.uuid4().hex
    now = time.time()
    with tx() as c:
        c.execute(
            "INSERT INTO jobs(job_id,kind,status,fleet_id,payload,result,error,progress,"
            "created_at,updated_at,cancel_flag) VALUES(?,?,?,?,?,?,?,?,?,?,0)",
            (jid, kind, "pending", fleet_id, json.dumps(payload, ensure_ascii=False),
             None, None, 0, now, now))
    return jid


def update_job(jid, **fields):
    if not fields:
        return
    for jk in ("payload", "result", "error"):
        if jk in fields and fields[jk] is not None and not isinstance(fields[jk], str):
            fields[jk] = json.dumps(fields[jk], ensure_ascii=False)
    cols = ", ".join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [time.time(), jid]
    with tx() as c:
        c.execute(f"UPDATE jobs SET {cols}, updated_at=? WHERE job_id=?", vals)


def get_job(jid):
    with tx() as c:
        row = c.execute("SELECT * FROM jobs WHERE job_id=?", (jid,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    for k in ("payload", "result", "error"):
        d[k] = json.loads(d[k]) if d[k] else None
    return d


def request_cancel(jid) -> bool:
    with tx() as c:
        cur = c.execute("UPDATE jobs SET cancel_flag=1, updated_at=? WHERE job_id=?",
                        (time.time(), jid))
        return cur.rowcount > 0


def is_cancel_requested(jid) -> bool:
    with tx() as c:
        row = c.execute("SELECT cancel_flag FROM jobs WHERE job_id=?", (jid,)).fetchone()
    return bool(row and row["cancel_flag"])


init_db()

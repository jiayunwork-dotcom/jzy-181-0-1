"""FastAPI 端到端测试：机组群、调度、局部重算、异步作业、校验错误。"""
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app import storage

client = TestClient(app)


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    # 每个测试用独立 SQLite 文件
    path = tmp_path / "test.db"
    monkeypatch.setattr(storage, "DB_PATH", str(path))
    storage._conn = storage._init_conn()
    storage.init_db()
    yield


FLEET_BODY = {
    "name": "三机群",
    "units": [
        {"name": "G1", "Pmin": 0, "pmax": 100, "a": 0, "b": 4, "c": 0.5,
         "ramp_up": 40, "ramp_down": 40, "initial_output": 0},
        {"name": "G2", "Pmin": 0, "pmax": 100, "a": 0, "b": 6, "c": 0.25,
         "ramp_up": 1000, "ramp_down": 1000, "initial_output": 0},
        {"name": "G3", "Pmin": 0, "pmax": 100, "a": 0, "b": 8, "c": 0.1,
         "ramp_up": 1000, "ramp_down": 1000, "initial_output": 0},
    ],
}


def _create_fleet(body=FLEET_BODY):
    r = client.post("/fleets", json=body)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def test_fleet_create_get_update_versioning():
    fid = _create_fleet()
    r = client.get(f"/fleets/{fid}")
    assert r.status_code == 200
    assert r.json()["version"] == 1

    body = dict(FLEET_BODY)
    body["units"] = [dict(u, b=5) for u in FLEET_BODY["units"]]
    r = client.put(f"/fleets/{fid}", json=body)
    assert r.status_code == 200 and r.json()["version"] == 2


def test_old_schedule_retains_params():
    fid = _create_fleet()
    r = client.post("/schedules", json={"fleet_id": fid,
                                        "loads": [220, 100, 80, 50]})
    assert r.status_code == 200, r.text
    sid = r.json()["schedule_id"]
    ver = r.json()["fleet_version"]

    # 改机组群
    body = dict(FLEET_BODY)
    body["units"] = [dict(u, b=99) for u in FLEET_BODY["units"]]
    client.put(f"/fleets/{fid}", json=body)

    # 旧调度仍能取回，且快照版本/参数保持旧版
    s = client.get(f"/schedules/{sid}").json()
    assert s["fleet_version"] == ver
    g1 = [u for u in s["params_snapshot"] if u["name"] == "G1"][0]
    assert g1["b"] == 4


def test_dispatch_response_fields_and_balance():
    fid = _create_fleet()
    loads = [220.0, 100.0, 80.0, 50.0]
    r = client.post("/schedules", json={"fleet_id": fid, "loads": loads})
    assert r.status_code == 200, r.text
    d = r.json()
    gens = np.array([d["generations"][f"G{i}"] for i in (1, 2, 3)])
    assert np.allclose(gens.sum(axis=0), loads, atol=1e-5)
    assert len(d["lambda"]) == 4
    assert d["kkt_verified"] is True
    assert d["power_imbalance_mw"] < 1e-5
    # t0 G1 顶爬坡
    assert "G1" in d["status_per_period"][0]["ramp_up_blocked"]


def test_infeasible_returns_period_and_shortfall():
    fid = _create_fleet()
    r = client.post("/schedules", json={"fleet_id": fid,
                                        "loads": [60, 1000, 80, 50]})
    assert r.status_code == 422
    err = r.json()["detail"]
    assert err["type"] == "infeasible"
    assert 1 in err["periods"]
    assert abs(err["shortfall_mw"]["1"] - 720.0) < 1.0
    assert abs(err["attainable_generation_mw"]["1"] - 280.0) < 1.0


def test_validation_error_has_field_name():
    body = {"name": "bad", "units": [
        {"name": "G", "Pmin": 0, "pmax": 100, "a": 0, "b": 4, "c": -1,
         "ramp_up": 10, "ramp_down": 10, "initial_output": 50}]}
    r = client.post("/fleets", json=body)
    assert r.status_code == 422
    assert r.json()["detail"][0]["field"] == "c"


def test_load_not_finite():
    fid = _create_fleet()
    # 用 Infinity（标准 JSON 允许序列化为 Infinity token）触发有限性校验
    r = client.post("/schedules",
                    content='{"fleet_id":"%s","loads":[100,1e999]}' % fid,
                    headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_bad_b_matrix_dimension():
    body = dict(FLEET_BODY)
    body["loss"] = {"B0": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    "B1": [0.0, 0.0, 0.0], "B2": 0.0}
    r = client.post("/fleets", json=body)
    assert r.status_code == 422
    assert r.json()["detail"][0]["field"] == "B"


def test_local_recompute_consistency():
    fid = _create_fleet()
    base = [120, 160, 100, 90]
    r = client.post("/schedules", json={"fleet_id": fid, "loads": base})
    sid = r.json()["schedule_id"]
    new = [120, 180, 100, 90]
    r2 = client.post(f"/schedules/{sid}/recompute",
                     json={"loads": new, "changed_periods": [1]})
    assert r2.status_code == 200, r2.text
    d = r2.json()
    assert d["matches_full_resolve"] is True
    assert d["cost_relative_diff"] < 1e-8
    gens = np.array([d["generations"][f"G{i}"] for i in (1, 2, 3)])
    assert np.allclose(gens.sum(axis=0), new, atol=1e-4)
    # 窗口包含改动时段
    lo, hi = d["local_window"]
    assert lo <= 1 <= hi


def test_async_job_lifecycle():
    fid = _create_fleet()
    r = client.post("/jobs/dispatch", json={"fleet_id": fid,
                                            "loads": [220, 100, 80, 50]})
    assert r.status_code == 202
    jid = r.json()["id"]
    # 轮询直到完成
    final = None
    for _ in range(60):
        j = client.get(f"/jobs/{jid}").json()
        if j["status"] in ("succeeded", "failed", "cancelled"):
            final = j
            break
        time.sleep(0.1)
    assert final is not None and final["status"] == "succeeded"
    assert final["result"]["kkt_verified"] is True


def test_cancel_nonexistent_job_404():
    r = client.post("/jobs/nope/cancel")
    assert r.status_code == 404


def test_health():
    assert client.get("/health").json()["status"] == "ok"

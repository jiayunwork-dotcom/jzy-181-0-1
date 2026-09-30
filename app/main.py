"""FastAPI 主程序：机组群、调度、局部重算、异步作业路由。"""
from __future__ import annotations

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from . import storage
from .models import (FleetCreate, FleetUpdate, ScheduleRequest,
                     LocalRecomputeRequest, Created, JobOut)
from .validation import (validate_unit, validate_b_coeff, validate_load,
                         ValidationError)
from .dispatch import economic_dispatch, verify_kkt, InfeasibleError
from .relocal import recompute as local_recompute
from . import jobs as jobmgr

app = FastAPI(title="多时段经济调度服务", version="1.0")


def _units_payload(units_in):
    return [validate_unit(u.model_dump()) for u in units_in]


def _loss_payload(loss, n):
    if loss is None:
        return None
    B0 = validate_b_coeff(loss.B0, n)
    B1 = np.asarray(loss.B1, dtype=float)
    if B1.shape != (n,):
        raise ValidationError("B1", f"B1 长度应为 {n}")
    if not np.all(np.isfinite(B1)):
        raise ValidationError("B1", "B1 含非有限元素")
    if not np.isfinite(loss.B2):
        raise ValidationError("B2", "B2 必须有限")
    return {"B0": B0, "B1": B1, "B2": float(loss.B2)}


@app.exception_handler(ValidationError)
async def validation_handler(request, exc: ValidationError):
    return JSONResponse(status_code=422,
                        content={"detail": [{"field": exc.field, "msg": exc.message}]})


@app.exception_handler(InfeasibleError)
async def infeasible_handler(request, exc: InfeasibleError):
    return JSONResponse(
        status_code=422,
        content={"detail": {
            "type": "infeasible",
            "message": "部分时段负荷超出爬坡约束下的可达出力范围",
            "periods": exc.periods,
            "shortfall_mw": exc.shortfalls,
            "attainable_generation_mw": exc.attainable_generation,
            "load_mw": exc.load,
        }})


# ---------------- 机组群 ----------------

@app.post("/fleets", response_model=Created, status_code=201)
def create_fleet(body: FleetCreate):
    units = _units_payload(body.units)
    B = _loss_payload(body.loss, len(units))
    fid = storage.create_fleet(body.name, units, B)
    return Created(id=fid)


@app.get("/fleets/{fid}")
def get_fleet(fid: str):
    f = storage.get_fleet(fid)
    if f is None:
        raise HTTPException(404, "机组群不存在")
    return {"fleet_id": f["fleet_id"], "name": f["name"], "version": f["version"],
            "units": f["params"], "loss": f["b_matrix"],
            "updated_at": f["updated_at"]}


@app.put("/fleets/{fid}")
def update_fleet(fid: str, body: FleetUpdate):
    if storage.get_fleet(fid) is None:
        raise HTTPException(404, "机组群不存在")
    units = _units_payload(body.units)
    B = _loss_payload(body.loss, len(units))
    ver = storage.update_fleet(fid, units, B)
    return {"fleet_id": fid, "version": ver}


# ---------------- 调度 ----------------

def _run_dispatch(fid, loads, persist=True, sid=None):
    f = storage.get_fleet(fid)
    if f is None:
        raise HTTPException(404, "机组群不存在")
    units = f["params"]
    T = len(loads)
    loads = validate_load(loads, T)
    B = f["b_matrix"]
    dr = economic_dispatch(units, T, loads, B=B)
    kkt = verify_kkt(dr, units, loads, B=B, tol=1e-5)
    result = _serialize_result(dr, units, loads, B, kkt)
    if persist:
        import json
        import uuid
        sid = sid or uuid.uuid4().hex
        result["schedule_id"] = sid
        result["fleet_id"] = fid
        result["fleet_version"] = f["version"]
        storage.save_schedule(sid, fid, f["version"], units, loads,
                              json.dumps(result, ensure_ascii=False), dr.cost)
    return result


def _serialize_result(dr, units, loads, B, kkt):
    names = [u["name"] for u in units]
    out = {
        "generations": {names[i]: dr.P[i].tolist() for i in range(len(units))},
        "lambda": dr.lam.tolist(),
        "total_cost": dr.cost,
        "iterations": dr.iterations,
        "penalty_iterations": dr.penalty_iters,
        "status_per_period": [
            {
                "period": t,
                "at_max": [names[i] for i in range(len(units)) if dr.flags["at_high"][i][t]],
                "at_min": [names[i] for i in range(len(units)) if dr.flags["at_low"][i][t]],
                "ramp_up_blocked": [names[i] for i in range(len(units)) if dr.flags["ramp_up_active"][i][t]],
                "ramp_down_blocked": [names[i] for i in range(len(units)) if dr.flags["ramp_down_active"][i][t]],
            } for t in range(len(loads))
        ],
        "ramp_multipliers": {
            "ramp_up": {names[i]: dr.multipliers["ramp_up"][i] for i in range(len(units))},
            "ramp_down": {names[i]: dr.multipliers["ramp_down"][i] for i in range(len(units))},
        },
        "kkt_verified": kkt["ok"],
        "kkt_violations": kkt["violations"],
    }
    if dr.loss is not None:
        out["loss"] = dr.loss.tolist()
        out["power_imbalance_mw"] = np.max(np.abs(dr.imbalance)).item()
    else:
        out["power_imbalance_mw"] = float(np.max(np.abs(dr.P.sum(axis=0) - loads)))
    return out


@app.post("/schedules")
def create_schedule(body: ScheduleRequest):
    return _run_dispatch(body.fleet_id, body.loads)


@app.get("/schedules/{sid}")
def get_schedule(sid: str):
    s = storage.get_schedule(sid)
    if s is None:
        raise HTTPException(404, "调度结果不存在")
    out = dict(s["result"])
    out["params_snapshot"] = s["params_snapshot"]
    out["fleet_id"] = s["fleet_id"]
    out["fleet_version"] = s["fleet_version"]
    out["loads"] = s["loads"]
    return out


@app.post("/schedules/{sid}/recompute")
def recompute_route(sid: str, body: LocalRecomputeRequest):
    s = storage.get_schedule(sid)
    if s is None:
        raise HTTPException(404, "调度结果不存在")
    units = s["params_snapshot"]
    old_loads = np.asarray(s["loads"], float)
    new_loads = validate_load(body.loads, len(old_loads))
    if body.changed_periods is not None:
        changed = body.changed_periods
    else:
        changed = [t for t in range(len(old_loads)) if abs(old_loads[t] - new_loads[t]) > 1e-9]
    # 重放旧 DispatchResult
    from .dispatch import DispatchResult
    r0 = s["result"]
    names = [u["name"] for u in units]
    P = np.array([r0["generations"][n] for n in names])
    dr0 = DispatchResult(
        P=P, lam=np.asarray(r0["lambda"], float), cost=r0["total_cost"],
        status="optimal", flags={
            "at_high": np.array([[n in set(x["at_max"]) for x in r0["status_per_period"]] for n in names]),
            "at_low": np.array([[n in set(x["at_min"]) for x in r0["status_per_period"]] for n in names]),
            "ramp_up_active": np.array([[n in set(x["ramp_up_blocked"]) for x in r0["status_per_period"]] for n in names]),
            "ramp_down_active": np.array([[n in set(x["ramp_down_blocked"]) for x in r0["status_per_period"]] for n in names]),
        }, multipliers={"ramp_up": np.zeros_like(P), "ramp_down": np.zeros_like(P),
                       "upper": np.zeros_like(P), "lower": np.zeros_like(P)},
        iterations=0)
    pdr, window = local_recompute(dr0, units, old_loads, new_loads, changed)
    kkt = verify_kkt(pdr, units, new_loads, tol=1e-5)
    out = _serialize_result(pdr, units, new_loads, None, kkt)
    out["local_window"] = list(window)
    out["recomputed_from"] = sid
    out["matches_full_resolve"] = pdr.consistent
    out["cost_relative_diff"] = pdr.cost_relative_diff
    nid = __import__("uuid").uuid4().hex
    storage.save_schedule(nid, s["fleet_id"], s["fleet_version"], units, new_loads,
                          __import__("json").dumps(out, ensure_ascii=False), pdr.cost)
    out["schedule_id"] = nid
    return out


# ---------------- 异步作业 ----------------

@app.post("/jobs/dispatch", response_model=Created, status_code=202)
def submit_dispatch_job(body: ScheduleRequest):
    fid = body.fleet_id
    loads = body.loads
    jid = storage.create_job("dispatch", fid, {"loads": loads})

    def fn(progress, cancel):
        if cancel():
            raise jobmgr.JobCancelled()
        result = _run_dispatch(fid, loads, persist=True)
        progress(100)
        return result

    jobmgr.submit(jid, "dispatch", fn)
    return Created(id=jid)


@app.get("/jobs/{jid}", response_model=JobOut)
def get_job(jid: str):
    j = storage.get_job(jid)
    if j is None:
        raise HTTPException(404, "作业不存在")
    return JobOut(job_id=j["job_id"], kind=j["kind"], status=j["status"],
                  progress=j["progress"], result=j["result"], error=j["error"])


@app.post("/jobs/{jid}/cancel", status_code=200)
def cancel_job(jid: str):
    j = storage.get_job(jid)
    if j is None:
        raise HTTPException(404, "作业不存在")
    ok = jobmgr.cancel(jid)
    return {"cancelled": ok, "status": j["status"]}


@app.get("/health")
def health():
    return {"status": "ok"}

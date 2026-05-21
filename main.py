"""
Quantum Trading Backend — Railway.app Edition v3.1
====================================================
Fix: Removed startup IBM connection (causes timeout on Railway free tier)
"""

import numpy as np
import time, os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

IBM_TOKEN    = os.environ.get("IBM_QUANTUM_TOKEN", "")
IBM_INSTANCE = os.environ.get("IBM_INSTANCE", "ibm-q/open/main")
USE_REAL_HW  = os.environ.get("USE_REAL_HW", "false").lower() == "true"

ibm_backend      = None
ibm_backend_name = "aer_simulator"

# Koneksi IBM hanya saat startup jika token & flag tersedia
if IBM_TOKEN and USE_REAL_HW:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        svc = QiskitRuntimeService(
            channel="ibm_quantum",
            token=IBM_TOKEN,
            instance=IBM_INSTANCE
        )
        ibm_backend      = svc.least_busy(operational=True, simulator=False, min_num_qubits=3)
        ibm_backend_name = ibm_backend.name
        print(f"[IBM] Connected: {ibm_backend_name}")
    except Exception as e:
        print(f"[IBM] Skipped: {e}")

app = FastAPI(title="Quantum Trading API", version="3.1")

# ── CORS: izinkan semua origin termasuk Claude artifact ──────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

class AnalysisRequest(BaseModel):
    pair:        str
    timeframe:   str
    prices:      list[float]
    use_real_hw: bool = False

class QuantumResult(BaseModel):
    pair:           str
    timeframe:      str
    signal:         str
    confidence:     float
    position_size:  float
    stop_loss:      float
    take_profit:    float
    risk_score:     float
    backend_used:   str
    quantum_states: dict
    execution_time: float
    rr_ratio:       str

TF_MULT = {
    "1m":0.3,"5m":0.5,"15m":0.7,"30m":0.85,
    "1h":1.0,"4h":1.5,"1d":2.5,"1w":4.0
}

PARAMS = np.array([0.7854,-0.5236,1.0472,-0.7854,0.3927,1.2566,-0.9817,0.6283])

def extract_features(prices: list, tf: str) -> np.ndarray:
    arr  = np.array(prices[-20:], dtype=float)
    mult = TF_MULT.get(tf, 1.0)
    ret  = np.diff(arr) / arr[:-1]
    momentum = (arr[-1] - arr[0]) / arr[0]
    vol      = np.std(ret) * mult
    gains    = ret[ret > 0].mean() if any(ret > 0) else 0
    losses   = abs(ret[ret < 0].mean()) if any(ret < 0) else 1e-9
    rsi      = gains / (gains + losses)
    z        = (arr[-1] - arr.mean()) / (arr.std() + 1e-9)
    mr       = float(np.tanh(z))
    feat     = np.array([momentum, vol, rsi, mr])
    rng      = feat.max() - feat.min() + 1e-9
    return np.pi * 2 * (feat - feat.min()) / rng - np.pi

def build_circuit(features: np.ndarray, params: np.ndarray) -> QuantumCircuit:
    qr = QuantumRegister(3, 'q')
    cr = ClassicalRegister(3, 'c')
    qc = QuantumCircuit(qr, cr)
    qc.h([qr[0], qr[1], qr[2]])
    qc.ry(float(features[0]), qr[0])
    qc.ry(float(features[1]), qr[1])
    qc.cx(qr[0], qr[1])
    qc.rz(float(features[2]), qr[0])
    qc.rx(float(features[3]), qr[1])
    qc.ry(float(params[0]), qr[0])
    qc.ry(float(params[1]), qr[1])
    qc.ry(float(params[2]), qr[2])
    qc.cx(qr[0], qr[2])
    qc.cx(qr[1], qr[2])
    qc.ry(float(params[3]), qr[2])
    qc.rz(float(params[4]), qr[2])
    qc.cx(qr[2], qr[0])
    qc.ry(float(params[5]), qr[0])
    qc.cx(qr[2], qr[1])
    qc.ry(float(params[6]), qr[1])
    qc.rz(float(params[7]), qr[2])
    qc.measure(qr, cr)
    return qc

def run_circuit(qc: QuantumCircuit, use_real: bool, shots: int = 2048) -> dict:
    backend  = ibm_backend if (use_real and ibm_backend) else AerSimulator()
    compiled = transpile(qc, backend, optimization_level=2)
    job      = backend.run(compiled, shots=shots)
    counts   = job.result().get_counts()
    total    = sum(counts.values())
    return {k: round(v / total, 4) for k, v in counts.items()}

def interpret(probs: dict) -> tuple:
    buy  = sum(v for k, v in probs.items() if k[0] == '1')
    sell = 1.0 - buy
    if buy  >= 0.62: return "BUY",  round(buy, 4)
    if sell >= 0.62: return "SELL", round(sell, 4)
    return "HOLD", round(max(buy, sell), 4)

def compute_atr(prices: list) -> float:
    arr = np.array(prices[-14:])
    return float(np.mean(arr * 0.01))

def kelly_size(prob: float) -> float:
    rr = 1.67
    f  = (rr * prob - (1 - prob)) / rr
    return round(min(max(f * 0.25, 0), 0.05) * 100, 2)

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "online",
        "service": "Quantum Trading API v3.1",
        "ibm_connected": ibm_backend is not None,
        "backend": ibm_backend_name
    }

@app.get("/health")
async def health():
    return {
        "status": "online",
        "ibm_connected": ibm_backend is not None,
        "backend": ibm_backend_name
    }

@app.post("/analyze", response_model=QuantumResult)
async def analyze(req: AnalysisRequest):
    if len(req.prices) < 20:
        raise HTTPException(400, "Minimal 20 data harga dibutuhkan")
    t0       = time.time()
    features = extract_features(req.prices, req.timeframe)
    qc       = build_circuit(features, PARAMS)
    probs    = run_circuit(qc, req.use_real_hw)
    signal, conf = interpret(probs)
    price    = req.prices[-1]
    atr      = compute_atr(req.prices)
    pos_size = kelly_size(conf) if signal != "HOLD" else 0.0
    if signal == "BUY":
        sl = round(price - 1.5 * atr, 6)
        tp = round(price + 2.5 * atr, 6)
    elif signal == "SELL":
        sl = round(price + 1.5 * atr, 6)
        tp = round(price - 2.5 * atr, 6)
    else:
        sl = tp = round(price, 6)
    top5 = dict(sorted(probs.items(), key=lambda x: -x[1])[:5])
    used = ibm_backend_name if (req.use_real_hw and ibm_backend) else "aer_simulator"
    return QuantumResult(
        pair=req.pair, timeframe=req.timeframe,
        signal=signal, confidence=conf,
        position_size=pos_size, stop_loss=sl, take_profit=tp,
        risk_score=round(1-conf,4), backend_used=used,
        quantum_states=top5,
        execution_time=round(time.time()-t0,3),
        rr_ratio="1:1.67"
    )

@app.post("/connect-ibm")
async def connect_ibm(token: str, instance: str = "ibm-q/open/main"):
    global ibm_backend, ibm_backend_name
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        svc = QiskitRuntimeService(channel="ibm_quantum", token=token, instance=instance)
        ibm_backend      = svc.least_busy(operational=True, simulator=False, min_num_qubits=3)
        ibm_backend_name = ibm_backend.name
        return {"status": "connected", "backend": ibm_backend_name}
    except Exception as e:
        raise HTTPException(503, f"IBM connection failed: {str(e)}")

@app.get("/backends")
async def list_backends():
    if not ibm_backend:
        return {"backends": ["aer_simulator"], "note": "IBM tidak terkoneksi"}
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        svc = QiskitRuntimeService(channel="ibm_quantum", token=IBM_TOKEN)
        bks = svc.backends(operational=True, simulator=False)
        return {"backends": [{"name": b.name, "qubits": b.num_qubits} for b in bks]}
    except Exception as e:
        raise HTTPException(500, str(e))


cat > /mnt/user-data/outputs/main.py << 'EOF'
"""
Quantum Trading Backend v4.0 — Real-time Price Data
=====================================================
Primary  : Twelve Data API
Backup   : Alpha Vantage API
Fallback : Mock data (jika kedua API gagal)
"""

import numpy as np
import time, os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

# ── Environment Variables ─────────────────────────────────────────────────────
IBM_TOKEN       = os.environ.get("IBM_QUANTUM_TOKEN", "")
IBM_INSTANCE    = os.environ.get("IBM_INSTANCE", "ibm-q/open/main")
USE_REAL_HW     = os.environ.get("USE_REAL_HW", "false").lower() == "true"
TWELVE_API_KEY  = os.environ.get("TWELVE_DATA_API_KEY", "")
ALPHA_API_KEY   = os.environ.get("ALPHA_VANTAGE_API_KEY", "")

ibm_backend      = None
ibm_backend_name = "aer_simulator"

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

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Quantum Trading API", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────────────────────────────
class AnalysisRequest(BaseModel):
    pair:        str          # e.g. "BTC/USDT"
    timeframe:   str          # e.g. "1h"
    use_real_hw: bool = False

class QuantumResult(BaseModel):
    pair:           str
    timeframe:      str
    current_price:  float
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
    data_source:    str       # twelve_data / alpha_vantage / mock

# ── Timeframe mapping ─────────────────────────────────────────────────────────
TF_MULT = {
    "1min":0.3,"5min":0.5,"15min":0.7,"30min":0.85,
    "1h":1.0,"4h":1.5,"1day":2.5,"1week":4.0,
    "1m":0.3,"5m":0.5,"15m":0.7,"30m":0.85,"1d":2.5,"1w":4.0
}

# Twelve Data interval format
TF_TO_TWELVE = {
    "1m":"1min","5m":"5min","15m":"15min","30m":"30min",
    "1h":"1h","4h":"4h","1d":"1day","1w":"1week",
    "1min":"1min","5min":"5min","15min":"15min","30min":"30min",
    "1day":"1day","1week":"1week"
}

# Alpha Vantage interval format
TF_TO_ALPHA = {
    "1m":"1min","5m":"5min","15m":"15min","30m":"30min",
    "1h":"60min","4h":"60min","1d":"daily","1w":"weekly",
    "1min":"1min","5min":"5min","15min":"15min","30min":"30min"
}

PARAMS = np.array([0.7854,-0.5236,1.0472,-0.7854,0.3927,1.2566,-0.9817,0.6283])

# ── Price Fetcher ─────────────────────────────────────────────────────────────

def normalize_pair_twelve(pair: str) -> str:
    """BTC/USDT → BTC/USDT (Twelve Data pakai format ini)"""
    return pair.replace("-", "/").upper()

def normalize_pair_alpha(pair: str) -> tuple:
    """BTC/USDT → (BTC, USDT)"""
    parts = pair.replace("-","/").upper().split("/")
    return (parts[0], parts[1] if len(parts)>1 else "USDT")

async def fetch_twelve_data(pair: str, timeframe: str, limit: int = 50) -> list:
    """Ambil data dari Twelve Data API."""
    if not TWELVE_API_KEY:
        raise ValueError("Twelve Data API key tidak ada")

    symbol   = normalize_pair_twelve(pair)
    interval = TF_TO_TWELVE.get(timeframe, "1h")

    url = (
        f"https://api.twelvedata.com/time_series"
        f"?symbol={symbol}&interval={interval}"
        f"&outputsize={limit}&apikey={TWELVE_API_KEY}&dp=6"
    )

    async with httpx.AsyncClient(timeout=15) as client:
        r    = await client.get(url)
        data = r.json()

    if "values" not in data:
        raise ValueError(f"Twelve Data error: {data.get('message','unknown error')}")

    prices = [float(v["close"]) for v in reversed(data["values"])]
    return prices

async def fetch_alpha_vantage(pair: str, timeframe: str, limit: int = 50) -> list:
    """Ambil data dari Alpha Vantage API sebagai backup."""
    if not ALPHA_API_KEY:
        raise ValueError("Alpha Vantage API key tidak ada")

    from_sym, to_sym = normalize_pair_alpha(pair)
    interval         = TF_TO_ALPHA.get(timeframe, "60min")

    # Tentukan function berdasarkan timeframe
    if timeframe in ["1d","1day","1w","1week"]:
        if timeframe in ["1w","1week"]:
            func = "DIGITAL_CURRENCY_WEEKLY"
        else:
            func = "DIGITAL_CURRENCY_DAILY"
        url = (
            f"https://www.alphavantage.co/query"
            f"?function={func}&symbol={from_sym}&market={to_sym}"
            f"&apikey={ALPHA_API_KEY}"
        )
        key_prefix = "4a. close"
    else:
        func = "CRYPTO_INTRADAY"
        url  = (
            f"https://www.alphavantage.co/query"
            f"?function={func}&symbol={from_sym}&market={to_sym}"
            f"&interval={interval}&outputsize=compact"
            f"&apikey={ALPHA_API_KEY}"
        )
        key_prefix = "4. close"

    async with httpx.AsyncClient(timeout=15) as client:
        r    = await client.get(url)
        data = r.json()

    # Cari key time series
    ts_key = next((k for k in data if "Time Series" in k), None)
    if not ts_key:
        raise ValueError(f"Alpha Vantage error: {list(data.keys())}")

    ts     = data[ts_key]
    prices = []
    for date in sorted(ts.keys())[-limit:]:
        close_key = next((k for k in ts[date] if "close" in k.lower()), None)
        if close_key:
            prices.append(float(ts[date][close_key]))

    if not prices:
        raise ValueError("Alpha Vantage: tidak ada data harga")

    return prices

def generate_mock_prices(pair: str, tf: str, n: int = 50) -> list:
    """Fallback mock prices jika kedua API gagal."""
    base = {
        "BTC/USDT":77000,"ETH/USDT":1800,"SOL/USDT":145,
        "BNB/USDT":580,"XRP/USDT":2.2,"DOGE/USDT":0.17,
        "ADA/USDT":0.35,"MATIC/USDT":0.25,"DOT/USDT":4.5
    }
    p0  = base.get(pair.upper(), 100)
    vol = TF_MULT.get(tf, 1.0) * 0.008
    prices = [p0]
    np.random.seed(int(time.time()) % 1000)
    for _ in range(n-1):
        r = (np.random.randn() * vol)
        prices.append(round(prices[-1] * (1 + r), 6))
    return prices

async def get_prices(pair: str, timeframe: str) -> tuple[list, str]:
    """
    Coba ambil harga:
    1. Twelve Data (primary)
    2. Alpha Vantage (backup)
    3. Mock data (fallback)
    """
    # Primary: Twelve Data
    try:
        prices = await fetch_twelve_data(pair, timeframe, limit=50)
        print(f"[DATA] Twelve Data OK: {pair} {timeframe} ({len(prices)} candles)")
        return prices, "twelve_data"
    except Exception as e:
        print(f"[DATA] Twelve Data failed: {e}")

    # Backup: Alpha Vantage
    try:
        prices = await fetch_alpha_vantage(pair, timeframe, limit=50)
        print(f"[DATA] Alpha Vantage OK: {pair} {timeframe} ({len(prices)} candles)")
        return prices, "alpha_vantage"
    except Exception as e:
        print(f"[DATA] Alpha Vantage failed: {e}")

    # Fallback: Mock
    print(f"[DATA] Using mock data for {pair} {timeframe}")
    prices = generate_mock_prices(pair, timeframe)
    return prices, "mock_data"

# ── Quantum Engine ────────────────────────────────────────────────────────────

def extract_features(prices: list, tf: str) -> np.ndarray:
    arr      = np.array(prices[-20:], dtype=float)
    mult     = TF_MULT.get(tf, 1.0)
    ret      = np.diff(arr) / arr[:-1]
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
    return {k: round(v/total, 4) for k, v in counts.items()}

def interpret(probs: dict) -> tuple:
    buy  = sum(v for k,v in probs.items() if k[0]=='1')
    sell = 1.0 - buy
    if buy  >= 0.62: return "BUY",  round(buy, 4)
    if sell >= 0.62: return "SELL", round(sell, 4)
    return "HOLD", round(max(buy, sell), 4)

def compute_atr(prices: list) -> float:
    arr = np.array(prices[-14:])
    return float(np.mean(arr * 0.012))

def kelly_size(prob: float) -> float:
    rr = 1.67
    f  = (rr * prob - (1-prob)) / rr
    return round(min(max(f*0.25, 0), 0.05)*100, 2)

# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "online",
        "version": "4.0",
        "ibm_connected": ibm_backend is not None,
        "backend": ibm_backend_name,
        "twelve_data": bool(TWELVE_API_KEY),
        "alpha_vantage": bool(ALPHA_API_KEY)
    }

@app.get("/health")
async def health():
    return {
        "status": "online",
        "ibm_connected": ibm_backend is not None,
        "backend": ibm_backend_name,
        "twelve_data": bool(TWELVE_API_KEY),
        "alpha_vantage": bool(ALPHA_API_KEY)
    }

@app.post("/analyze", response_model=QuantumResult)
async def analyze(req: AnalysisRequest):
    t0 = time.time()

    # Ambil harga real-time
    prices, data_source = await get_prices(req.pair, req.timeframe)

    if len(prices) < 20:
        raise HTTPException(400, f"Data tidak cukup: {len(prices)} candles")

    # Quantum analysis
    features     = extract_features(prices, req.timeframe)
    qc           = build_circuit(features, PARAMS)
    probs        = run_circuit(qc, req.use_real_hw)
    signal, conf = interpret(probs)

    price    = prices[-1]
    atr      = compute_atr(prices)
    pos_size = kelly_size(conf) if signal != "HOLD" else 0.0

    if signal == "BUY":
        sl = round(price - 1.5*atr, 6)
        tp = round(price + 2.5*atr, 6)
    elif signal == "SELL":
        sl = round(price + 1.5*atr, 6)
        tp = round(price - 2.5*atr, 6)
    else:
        sl = tp = round(price, 6)

    top5 = dict(sorted(probs.items(), key=lambda x:-x[1])[:5])
    used = ibm_backend_name if (req.use_real_hw and ibm_backend) else "aer_simulator"

    return QuantumResult(
        pair          = req.pair,
        timeframe     = req.timeframe,
        current_price = round(price, 4),
        signal        = signal,
        confidence    = conf,
        position_size = pos_size,
        stop_loss     = sl,
        take_profit   = tp,
        risk_score    = round(1-conf, 4),
        backend_used  = used,
        quantum_states= top5,
        execution_time= round(time.time()-t0, 3),
        rr_ratio      = "1:1.67",
        data_source   = data_source
    )

@app.post("/connect-ibm")
async def connect_ibm(token: str, instance: str = "ibm-q/open/main"):
    global ibm_backend, ibm_backend_name
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        svc              = QiskitRuntimeService(channel="ibm_quantum", token=token, instance=instance)
        ibm_backend      = svc.least_busy(operational=True, simulator=False, min_num_qubits=3)
        ibm_backend_name = ibm_backend.name
        return {"status":"connected","backend":ibm_backend_name}
    except Exception as e:
        raise HTTPException(503, f"IBM connection failed: {str(e)}")
EOF
echo "Done"

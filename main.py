"""
Quantum Trading Backend v5.0 FINAL
====================================
Real-time: Twelve Data + Alpha Vantage backup
Quantum  : IBM Hardware / Aer Simulator
Pairs    : Crypto + XAUUSD + Forex
"""

import numpy as np
import time, os
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator

# ── API Keys (dari environment variable Railway) ──────────────────────────────
IBM_TOKEN    = os.environ.get("IBM_QUANTUM_TOKEN", "")
IBM_INSTANCE = os.environ.get("IBM_INSTANCE", "ibm-q/open/main")
USE_REAL_HW  = os.environ.get("USE_REAL_HW", "false").lower() == "true"
TWELVE_KEY   = os.environ.get("TWELVE_DATA_API_KEY", "5cbbd8c76240448c9e7c1a84ed2532c5")
ALPHA_KEY    = os.environ.get("ALPHA_VANTAGE_API_KEY", "OE6MYY6P1SRF2T9M")

# ── IBM Quantum ───────────────────────────────────────────────────────────────
ibm_backend      = None
ibm_backend_name = "aer_simulator"

if IBM_TOKEN and USE_REAL_HW:
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        svc              = QiskitRuntimeService(channel="ibm_quantum", token=IBM_TOKEN, instance=IBM_INSTANCE)
        ibm_backend      = svc.least_busy(operational=True, simulator=False, min_num_qubits=3)
        ibm_backend_name = ibm_backend.name
        print(f"[IBM] Connected: {ibm_backend_name}")
    except Exception as e:
        print(f"[IBM] Skipped: {e}")

# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="Quantum Trading API", version="5.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET","POST","OPTIONS"],
    allow_headers=["*"],
)

# ── Models ────────────────────────────────────────────────────────────────────
class AnalysisRequest(BaseModel):
    pair:        str
    timeframe:   str
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
    data_source:    str

# ── Constants ─────────────────────────────────────────────────────────────────
TF_MULT   = {"1m":0.3,"5m":0.5,"15m":0.7,"30m":0.85,"1h":1.0,"4h":1.5,"1d":2.5,"1w":4.0}
TF_TWELVE = {"1m":"1min","5m":"5min","15m":"15min","30m":"30min","1h":"1h","4h":"4h","1d":"1day","1w":"1week"}
TF_ALPHA  = {"1m":"1min","5m":"5min","15m":"15min","30m":"30min","1h":"60min","4h":"60min","1d":"daily","1w":"weekly"}
PARAMS    = np.array([0.7854,-0.5236,1.0472,-0.7854,0.3927,1.2566,-0.9817,0.6283])

METAL_FOREX = ["XAUUSD","XAGUSD","EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD"]

def clean_pair(pair: str) -> str:
    return pair.upper().replace("-","").replace("/","")

def is_metal_forex(pair: str) -> bool:
    return clean_pair(pair) in METAL_FOREX

def fmt_twelve(pair: str) -> str:
    p = clean_pair(pair)
    return p[:3]+"/"+p[3:] if len(p)==6 else pair.upper()

def fmt_alpha(pair: str) -> tuple:
    p = clean_pair(pair)
    return (p[:3], p[3:]) if len(p)==6 else (p, "USD")

# ── Data Fetchers ─────────────────────────────────────────────────────────────
async def fetch_twelve(pair: str, tf: str, n: int = 60) -> list:
    symbol   = fmt_twelve(pair)
    interval = TF_TWELVE.get(tf, "1h")
    url      = (f"https://api.twelvedata.com/time_series"
                f"?symbol={symbol}&interval={interval}"
                f"&outputsize={n}&apikey={TWELVE_KEY}&dp=6")
    async with httpx.AsyncClient(timeout=20) as c:
        data = (await c.get(url)).json()
    if "values" not in data:
        raise ValueError(data.get("message","Twelve Data error"))
    return [float(v["close"]) for v in reversed(data["values"])]

async def fetch_alpha(pair: str, tf: str, n: int = 60) -> list:
    fs, ts = fmt_alpha(pair)
    iv     = TF_ALPHA.get(tf,"60min")
    metal  = is_metal_forex(pair)

    if tf in ["1d","1w"]:
        if metal:
            fn  = "FX_DAILY" if tf=="1d" else "FX_WEEKLY"
            url = (f"https://www.alphavantage.co/query?function={fn}"
                   f"&from_symbol={fs}&to_symbol={ts}&outputsize=compact&apikey={ALPHA_KEY}")
        else:
            fn  = "DIGITAL_CURRENCY_DAILY" if tf=="1d" else "DIGITAL_CURRENCY_WEEKLY"
            url = (f"https://www.alphavantage.co/query?function={fn}"
                   f"&symbol={fs}&market={ts}&apikey={ALPHA_KEY}")
    else:
        if metal:
            fn  = "FX_INTRADAY"
            url = (f"https://www.alphavantage.co/query?function={fn}"
                   f"&from_symbol={fs}&to_symbol={ts}&interval={iv}"
                   f"&outputsize=compact&apikey={ALPHA_KEY}")
        else:
            fn  = "CRYPTO_INTRADAY"
            url = (f"https://www.alphavantage.co/query?function={fn}"
                   f"&symbol={fs}&market={ts}&interval={iv}"
                   f"&outputsize=compact&apikey={ALPHA_KEY}")

    async with httpx.AsyncClient(timeout=20) as c:
        data = (await c.get(url)).json()

    ts_key = next((k for k in data if "time series" in k.lower()), None)
    if not ts_key:
        raise ValueError(f"Alpha key not found: {list(data.keys())[:3]}")

    ts     = data[ts_key]
    prices = []
    for d in sorted(ts.keys())[-n:]:
        ck = next((k for k in ts[d] if "close" in k.lower()), None)
        if ck:
            prices.append(float(ts[d][ck]))
    if not prices:
        raise ValueError("Alpha Vantage: empty prices")
    return prices

def mock_prices(pair: str, tf: str, n: int = 60) -> list:
    base = {
        "BTCUSDT":77000,"ETHUSDT":1800,"SOLUSDT":145,"BNBUSDT":580,
        "XRPUSDT":2.2,"DOGEUSDT":0.17,"ADAUSDT":0.35,
        "XAUUSD":3300,"XAGUSD":33,"EURUSD":1.08,"GBPUSD":1.27
    }
    p0  = base.get(clean_pair(pair), 100)
    vol = TF_MULT.get(tf,1.0) * 0.008
    np.random.seed(int(time.time())%9999)
    px  = [p0]
    for _ in range(n-1):
        px.append(round(px[-1]*(1+np.random.randn()*vol),6))
    return px

async def get_prices(pair: str, tf: str) -> tuple:
    try:
        p = await fetch_twelve(pair, tf)
        print(f"[OK] TwelveData {pair} {tf} {len(p)}bars")
        return p, "twelve_data"
    except Exception as e:
        print(f"[FAIL] TwelveData: {e}")
    try:
        p = await fetch_alpha(pair, tf)
        print(f"[OK] AlphaVantage {pair} {tf} {len(p)}bars")
        return p, "alpha_vantage"
    except Exception as e:
        print(f"[FAIL] AlphaVantage: {e}")
    print(f"[WARN] Mock {pair} {tf}")
    return mock_prices(pair, tf), "mock_data"

# ── Quantum Engine ────────────────────────────────────────────────────────────
def extract_features(prices: list, tf: str) -> np.ndarray:
    arr  = np.array(prices[-20:], dtype=float)
    mult = TF_MULT.get(tf, 1.0)
    ret  = np.diff(arr)/arr[:-1]
    mom  = (arr[-1]-arr[0])/arr[0]
    vol  = np.std(ret)*mult
    g    = ret[ret>0].mean() if any(ret>0) else 0
    l    = abs(ret[ret<0].mean()) if any(ret<0) else 1e-9
    rsi  = g/(g+l)
    mr   = float(np.tanh((arr[-1]-arr.mean())/(arr.std()+1e-9)))
    feat = np.array([mom,vol,rsi,mr])
    rng  = feat.max()-feat.min()+1e-9
    return np.pi*2*(feat-feat.min())/rng - np.pi

def build_circuit(f, p) -> QuantumCircuit:
    qr=QuantumRegister(3,'q'); cr=ClassicalRegister(3,'c')
    qc=QuantumCircuit(qr,cr)
    qc.h([qr[0],qr[1],qr[2]])
    qc.ry(float(f[0]),qr[0]); qc.ry(float(f[1]),qr[1])
    qc.cx(qr[0],qr[1])
    qc.rz(float(f[2]),qr[0]); qc.rx(float(f[3]),qr[1])
    qc.ry(float(p[0]),qr[0]); qc.ry(float(p[1]),qr[1]); qc.ry(float(p[2]),qr[2])
    qc.cx(qr[0],qr[2]); qc.cx(qr[1],qr[2])
    qc.ry(float(p[3]),qr[2]); qc.rz(float(p[4]),qr[2])
    qc.cx(qr[2],qr[0]); qc.ry(float(p[5]),qr[0])
    qc.cx(qr[2],qr[1]); qc.ry(float(p[6]),qr[1])
    qc.rz(float(p[7]),qr[2])
    qc.measure(qr,cr)
    return qc

def run_circuit(qc, use_real: bool, shots=2048) -> dict:
    bk  = ibm_backend if (use_real and ibm_backend) else AerSimulator()
    cmp = transpile(qc, bk, optimization_level=2)
    cnt = bk.run(cmp, shots=shots).result().get_counts()
    tot = sum(cnt.values())
    return {k:round(v/tot,4) for k,v in cnt.items()}

def interpret(probs: dict) -> tuple:
    buy  = sum(v for k,v in probs.items() if k[0]=='1')
    sell = 1.0-buy
    if buy  >= 0.62: return "BUY",  round(buy,4)
    if sell >= 0.62: return "SELL", round(sell,4)
    return "HOLD", round(max(buy,sell),4)

def atr(prices): return float(np.mean(np.array(prices[-14:])*0.012))
def kelly(prob):
    f=(1.67*prob-(1-prob))/1.67
    return round(min(max(f*0.25,0),0.05)*100,2)

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/"); 
@app.get("/health")
async def health():
    return {"status":"online","version":"5.0",
            "ibm_connected":ibm_backend is not None,
            "backend":ibm_backend_name,
            "twelve_data":bool(TWELVE_KEY),
            "alpha_vantage":bool(ALPHA_KEY)}

@app.post("/analyze", response_model=QuantumResult)
async def analyze(req: AnalysisRequest):
    t0          = time.time()
    prices, src = await get_prices(req.pair, req.timeframe)
    if len(prices) < 20:
        raise HTTPException(400, f"Data kurang: {len(prices)} candles")
    feat         = extract_features(prices, req.timeframe)
    qc           = build_circuit(feat, PARAMS)
    probs        = run_circuit(qc, req.use_real_hw)
    signal, conf = interpret(probs)
    price        = prices[-1]
    a            = atr(prices)
    pos          = kelly(conf) if signal!="HOLD" else 0.0
    sl = round(price-(1.5*a if signal=="BUY" else -1.5*a),4)
    tp = round(price+(2.5*a if signal=="BUY" else -2.5*a),4)
    if signal=="HOLD": sl=tp=round(price,4)
    used = ibm_backend_name if (req.use_real_hw and ibm_backend) else "aer_simulator"
    return QuantumResult(
        pair=req.pair, timeframe=req.timeframe,
        current_price=round(price,4), signal=signal,
        confidence=conf, position_size=pos,
        stop_loss=sl, take_profit=tp,
        risk_score=round(1-conf,4), backend_used=used,
        quantum_states=dict(sorted(probs.items(),key=lambda x:-x[1])[:5]),
        execution_time=round(time.time()-t0,3),
        rr_ratio="1:1.67", data_source=src)

@app.post("/connect-ibm")
async def connect_ibm(token: str, instance: str="ibm-q/open/main"):
    global ibm_backend, ibm_backend_name
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        svc=QiskitRuntimeService(channel="ibm_quantum",token=token,instance=instance)
        ibm_backend=svc.least_busy(operational=True,simulator=False,min_num_qubits=3)
        ibm_backend_name=ibm_backend.name
        return {"status":"connected","backend":ibm_backend_name}
    except Exception as e:
        raise HTTPException(503,str(e))

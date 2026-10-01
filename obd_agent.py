"""
OBD Smart Library - AI Agent prototype (single file)

SETUP
  pip install fastapi uvicorn python-multipart pypdf requests beautifulsoup4
  export GEMINI_API_KEY="your_free_key_from_aistudio.google.com"   (Windows: set GEMINI_API_KEY=...)
  uvicorn obd_agent:app --reload
  Open http://127.0.0.1:8000 and upload an OBD scan PDF.

FLOW
  PDF -> text -> Gemini extracts {dtc_codes, vin, make, model, engine, year}
      -> fallback: regex for VIN + DTC codes if AI fails
      -> search PartSouq (VIN first, else make+model+engine, else DTC code)
      -> if nothing, search 7zap
      -> JSON result
"""
import io, json, os, re
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pypdf import PdfReader

GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124 Safari/537.36"}

VIN_RE = re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b")
DTC_RE = re.compile(r"\b[PCBU][0-9A-F]{4}\b")

app = FastAPI(title="OBD Smart Library Agent")


# ---------- 1. PDF -> text ----------
def pdf_to_text(data: bytes) -> str:
    reader = PdfReader(io.BytesIO(data))
    return "\n".join((p.extract_text() or "") for p in reader.pages).strip()


# ---------- 2. AI extraction (with regex fallback) ----------
PROMPT = """You are an automotive OBD scan report parser.
From the report text below, extract JSON ONLY (no markdown) with exactly these keys:
{"dtc_codes":[{"code":"P0301","description":"short description from report or null"}],
 "vin":"17-char VIN or null","make":null,"model":null,"year":null,"engine":null,
 "other_identifiers":["any other useful identifiers e.g. ECU name, part numbers"]}
Use null when not present. Never invent values.

REPORT:
"""

def ai_extract(text: str) -> dict | None:
    if not GEMINI_KEY:
        return None
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={GEMINI_KEY}"
    body = {
        "contents": [{"parts": [{"text": PROMPT + text[:30000]}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    try:
        r = requests.post(url, json=body, timeout=60)
        r.raise_for_status()
        raw = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(re.sub(r"^```json|```$", "", raw.strip()).strip())
    except Exception as e:
        print("AI extraction failed:", e)
        return None

def regex_extract(text: str) -> dict:
    vins = [v for v in VIN_RE.findall(text.upper()) if re.search(r"\d", v) and re.search(r"[A-Z]", v)]
    codes = sorted(set(DTC_RE.findall(text.upper())))
    return {
        "dtc_codes": [{"code": c, "description": None} for c in codes],
        "vin": vins[0] if vins else None,
        "make": None, "model": None, "year": None, "engine": None,
        "other_identifiers": [],
    }

def extract(text: str) -> tuple[dict, str]:
    ai = ai_extract(text)
    rx = regex_extract(text)
    if ai:
        # fill any gaps in AI result using regex
        if not ai.get("vin"):
            ai["vin"] = rx["vin"]
        if not ai.get("dtc_codes"):
            ai["dtc_codes"] = rx["dtc_codes"]
        return ai, "ai"
    return rx, "regex-fallback"


# ---------- 3. Build search queries in priority order ----------
def build_queries(info: dict) -> list[tuple[str, str]]:
    q = []
    if info.get("vin"):
        q.append(("vin", info["vin"]))
    parts = [info.get("make"), info.get("model"), info.get("year"), info.get("engine")]
    parts = [str(p) for p in parts if p]
    if len(parts) >= 2:
        q.append(("vehicle", " ".join(parts)))
    for oid in (info.get("other_identifiers") or [])[:3]:
        q.append(("identifier", str(oid)))
    for d in (info.get("dtc_codes") or [])[:3]:
        q.append(("dtc", d["code"]))
    return q


# ---------- 4. Fetch from catalog sites ----------
def scrape(site: str, url: str) -> list[dict]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        out, seen = [], set()
        for a in soup.select("a[href]"):
            title = " ".join(a.get_text(" ", strip=True).split())
            href = a["href"]
            if len(title) < 4 or href in seen:
                continue
            if any(k in href.lower() for k in ("catalog", "vehicle", "part", "unit", "search", "/car")):
                seen.add(href)
                if href.startswith("/"):
                    href = "https://" + url.split("/")[2] + href
                out.append({"title": title[:150], "link": href})
        return out[:15]
    except Exception as e:
        print(f"{site} error:", e)
        return []

def partsouq(q: str) -> list[dict]:
    return scrape("partsouq", f"https://partsouq.com/en/search/all?q={requests.utils.quote(q)}")

def zap7(q: str) -> list[dict]:
    return scrape("7zap", f"https://7zap.com/en/search/?keyword={requests.utils.quote(q)}")

def fetch_catalog(queries: list[tuple[str, str]]) -> dict:
    tried = []
    for kind, q in queries:
        res = partsouq(q)
        tried.append({"site": "partsouq", "kind": kind, "query": q, "hits": len(res)})
        if res:
            return {"source": "partsouq", "query": q, "results": res, "tried": tried}
        res = zap7(q)
        tried.append({"site": "7zap", "kind": kind, "query": q, "hits": len(res)})
        if res:
            return {"source": "7zap", "query": q, "results": res, "tried": tried}
    return {"source": None, "query": None, "results": [], "tried": tried}


# ---------- 5. API ----------
@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    data = await file.read()
    try:
        text = pdf_to_text(data)
    except Exception as e:
        return JSONResponse({"error": f"Could not read PDF: {e}"}, status_code=400)
    if not text:
        return JSONResponse({"error": "No text found. PDF may be scanned/image-only (needs OCR)."}, status_code=422)

    info, method = extract(text)
    queries = build_queries(info)
    catalog = fetch_catalog(queries) if queries else {"source": None, "results": [], "tried": []}
    return {"extraction_method": method, "extracted": info, "catalog": catalog}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>OBD Smart Library</title>
<style>body{font-family:system-ui;margin:0;background:#f4f6fb}nav{background:#1e63d6;color:#fff;padding:14px 20px;font-weight:600}
.card{max-width:720px;margin:24px auto;background:#fff;padding:20px;border-radius:12px;box-shadow:0 2px 8px #0002}
button{background:#1e63d6;color:#fff;border:0;padding:10px 18px;border-radius:8px;cursor:pointer}
pre{background:#0e1525;color:#cfe3ff;padding:12px;border-radius:8px;overflow:auto;max-height:420px}</style></head>
<body><nav>OBD Smart Library</nav><div class="card"><h3>Upload OBD scan PDF</h3>
<input type="file" id="f" accept="application/pdf"> <button onclick="go()">Analyze</button>
<p id="s"></p><pre id="o">Result will appear here</pre></div>
<script>
async function go(){const f=document.getElementById('f').files[0];if(!f)return;
document.getElementById('s').textContent='Analyzing...';
const fd=new FormData();fd.append('file',f);
const r=await fetch('/analyze',{method:'POST',body:fd});
document.getElementById('o').textContent=JSON.stringify(await r.json(),null,2);
document.getElementById('s').textContent='';}
</script></body></html>
"""

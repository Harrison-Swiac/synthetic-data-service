from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd
import numpy as np
import io, uuid
from datetime import datetime

# SDV (fast, CPU-friendly model for tabular data)
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

app = FastAPI(title="Synthetic Data Service", version="0.5.0")

# ---------------- Root & Health ----------------
@app.get("/")
def root():
    return {
        "service": "Synthetic Data Service",
        "endpoints": [
            "/health",
            "/synthesize/tabular",
            "/validate/privacy",
            "/evaluate/tabular",
        ],
    }

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

# ---------------- Helpers ----------------
PII_HINTS = {
    "name", "first_name", "last_name", "surname",
    "dob", "date_of_birth", "birthdate",
    "ssn", "tax", "medicare", "passport",
    "driver", "licence", "license",
    "email", "phone", "mobile",
    "address", "postcode", "zipcode",
    "credit", "card", "iban", "account",
    "mrn", "patient_id", "nhs", "uhid"
}

def df_to_csv_response(df: pd.DataFrame, download_name: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
    )

def coerce_datetimes(df: pd.DataFrame):
    """Try to parse date/time-ish columns to datetime dtype."""
    for col in df.columns:
        lc = col.lower()
        if "date" in lc or "time" in lc:
            with pd.option_context("mode.chained_assignment", None):
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass
    return df

def build_metadata(df: pd.DataFrame) -> SingleTableMetadata:
    """Infer column types, then add hints for better fidelity."""
    meta = SingleTableMetadata()
    meta.detect_from_dataframe(data=df)

    # Treat obvious IDs correctly
    if "patient_id" in df.columns:
        meta.update_column("patient_id", sdtype="id")

    # Low-cardinality strings -> categorical
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            if df[col].nunique(dropna=True) <= max(20, int(0.1 * max(1, len(df)))):
                meta.update_column(col, sdtype="categorical")
        except Exception:
            pass

    # Datetime columns with a consistent format
    for col in df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        meta.update_column(col, sdtype="datetime", datetime_format="%Y-%m-%d")

    return meta

def guardrails(df_in: pd.DataFrame, df_out: pd.DataFrame, id_cols: set | None = None) -> pd.DataFrame:
    """
    Post-process synthetic output to keep values plausible and clean.
    - Clip numeric columns to source min/max.
    - Fence categoricals to source vocab (but never for ID columns).
    - Serialise datetimes to ISO strings.
    """
    id_cols = id_cols or set()

    # Numeric bounds
    num_cols = df_in.select_dtypes(include=["number"]).columns
    for c in num_cols:
        try:
            lo, hi = float(df_in[c].min()), float(df_in[c].max())
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce").clip(lower=lo, upper=hi)
        except Exception:
            pass

    # Categoricals: restrict to source vocab (skip ids)
    for c in df_in.columns:
        if c in id_cols:
            continue
        if c in df_out.columns and df_in[c].dtype == "object":
            vocab = df_in[c].dropna().astype(str).unique().tolist()
            if vocab:
                df_out[c] = df_out[c].astype(str).apply(
                    lambda v: v if v in vocab else vocab[hash(v) % len(vocab)]
                )

    # Datetimes -> ISO strings
    for col in df_out.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        df_out[col] = df_out[col].dt.strftime("%Y-%m-%d")

    return df_out

def psi_numeric(ref: pd.Series, syn: pd.Series, bins: int = 10) -> float:
    """Population Stability Index for numeric columns (rough but useful)."""
    ref_n, syn_n = ref.dropna(), syn.dropna()
    if ref_n.nunique() == 0:
        return 0.0
    try:
        q = np.linspace(0, 1, bins + 1)
        edges = np.unique(ref_n.quantile(q).values)
        if len(edges) < 3:
            edges = np.linspace(ref_n.min(), ref_n.max(), min(3, bins + 1))
    except Exception:
        edges = np.linspace(ref_n.min(), ref_n.max(), min(3, bins + 1))
    if np.all(np.isclose(np.diff(edges), 0)):
        return 0.0
    ref_hist, _ = np.histogram(ref_n, bins=edges)
    syn_hist, _ = np.histogram(syn_n, bins=edges)
    ref_p = (ref_hist + 1e-6) / (ref_hist.sum() + 1e-6 * len(ref_hist))
    syn_p = (syn_hist + 1e-6) / (syn_hist.sum() + 1e-6 * len(syn_hist))
    return float(np.sum((syn_p - ref_p) * np.log(syn_p / ref_p)))

def cat_overlap(ref: pd.Series, syn: pd.Series):
    """Simple Jaccard + coverage for categories."""
    a = set(ref.dropna().astype(str).unique())
    b = set(syn.dropna().astype(str).unique())
    if not a and not b:
        return {"jaccard": 1.0, "coverage": 1.0}
    inter = len(a & b)
    union = len(a | b) or 1
    return {"jaccard": inter / union, "coverage": inter / (len(a) or 1)}

# ---------------- Synthesize (GaussianCopula) ----------------
@app.post("/synthesize/tabular")
async def synthesize_tabular(num_rows: int = Form(1000), file: UploadFile = File(...)):
    """
    Accepts a CSV and returns a *truly synthetic* CSV generated with SDV's GaussianCopula.
    Adds light guardrails and a synthetic_id column.
    """
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content)).copy()
    df = coerce_datetimes(df)

    meta = build_metadata(df)
    synthesizer = GaussianCopulaSynthesizer(meta)
    synthesizer.fit(df)

    rows = max(1, min(int(num_rows), 50000))
    out_df = synthesizer.sample(rows)

    id_cols = {"patient_id"} if "patient_id" in df.columns else set()
    out_df = guardrails(df, out_df, id_cols=id_cols)

    out_df["synthetic_id"] = [str(uuid.uuid4())[:8] for _ in range(len(out_df))]
    return df_to_csv_response(out_df, download_name=f"synthetic_{file.filename}")

# ---------------- Privacy check (heuristic) ----------------
@app.post("/validate/privacy")
async def validate_privacy(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    flagged = []
    for col in df.columns:
        lower = col.lower().replace("-", "_").replace(" ", "_")
        if any(h in lower for h in PII_HINTS):
            flagged.append(col)
    return JSONResponse({
        "columns_flagged": flagged,
        "pii_hint_count": len(flagged),
        "advice": "Avoid uploading direct identifiers. Map IDs to surrogate keys before synthesis."
    })

# ---------------- Quick evaluation ----------------
@app.post("/evaluate/tabular")
async def evaluate_tabular(reference_file: UploadFile = File(...), synthetic_file: UploadFile = File(...)):
    ref_df = pd.read_csv(io.BytesIO(await reference_file.read()))
    syn_df = pd.read_csv(io.BytesIO(await synthetic_file.read()))
    shared = [c for c in ref_df.columns if c in syn_df.columns]
    ref_df, syn_df = ref_df[shared], syn_df[shared]

    nums = [c for c in shared if pd.api.types.is_numeric_dtype(ref_df[c])]
    cats = [c for c in shared if not pd.api.types.is_numeric_dtype(ref_df[c])]

    psi_scores, overlap = {}, {}
    for c in nums:
        try:
            psi_scores[c] = psi_numeric(ref_df[c], syn_df[c])
        except Exception:
            psi_scores[c] = None
    for c in cats:
        try:
            overlap[c] = cat_overlap(ref_df[c], syn_df[c])
        except Exception:
            overlap[c] = None

    return JSONResponse({
        "columns_compared": shared,
        "numeric_psi": psi_scores,
        "categorical_overlap": overlap,
        "notes": "PSI ~0–0.1 small shift, 0.1–0.25 moderate, >0.25 large. Overlap closer to 1.0 is better."
    })

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
import pandas as pd
import numpy as np
import io, uuid
from datetime import datetime

# SDV (CPU-friendly model)
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata

app = FastAPI(title="Synthetic Data Service", version="0.3.0")

# ---------------- Health ----------------
@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

# ---------------- Helpers ----------------
PII_HINTS = {
    "name", "first_name", "last_name", "surname", "dob", "date_of_birth", "birthdate",
    "ssn", "tax", "medicare", "passport", "driver", "license", "email", "phone",
    "mobile", "address", "postcode", "zipcode", "credit", "card", "iban", "account",
    "mrn", "patient_id", "nhs", "uhid"
}

def guess_datetime_cols(df: pd.DataFrame):
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            with pd.option_context("mode.chained_assignment", None):
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass
    return df

def df_to_csv_response(df: pd.DataFrame, download_name: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{download_name}"'}
    )

def psi_numeric(ref: pd.Series, syn: pd.Series, bins: int = 10) -> float:
    ref_n = ref.dropna()
    syn_n = syn.dropna()
    if ref_n.nunique() == 0:
        return 0.0
    try:
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(ref_n.quantile(quantiles).values)
        if len(edges) < 3:
            edges = np.linspace(ref_n.min(), ref_n.max(), min(3, bins + 1))
    except Exception:
        edges = np.linspace(ref_n.min(), ref_n.max(), min(3, bins + 1))
    if np.all(np.isclose(np.diff(edges), 0)):
        return 0.0
    ref_hist, _ = np.histogram(ref_n, bins=edges)
    syn_hist, _ = np.histogram(syn_n, bins=edges)
    ref_prop = (ref_hist + 1e-6) / (ref_hist.sum() + 1e-6 * len(ref_hist))
    syn_prop = (syn_hist + 1e-6) / (syn_hist.sum() + 1e-6 * len(syn_hist))
    psi_vals = (syn_prop - ref_prop) * np.log(syn_prop / ref_prop)
    return float(np.sum(psi_vals))

def cat_overlap(ref: pd.Series, syn: pd.Series):
    a = set(ref.dropna().astype(str).unique())
    b = set(syn.dropna().astype(str).unique())
    if not a and not b:
        return {"jaccard": 1.0, "coverage": 1.0}
    inter = len(a & b)
    union = len(a | b) if a | b else 1
    return {"jaccard": inter / union, "coverage": inter / (len(a) if len(a) else 1)}

# ---------------- Synthesize (real) ----------------
@app.post("/synthesize/tabular")
async def synthesize_tabular(num_rows: int = Form(1000), file: UploadFile = File(...)):
    """
    SDV GaussianCopula with schema hints + postprocessing guardrails.
    """
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content)).copy()

    # --- Basic typing hints ---
    # Dates
    for col in df.columns:
        if "date" in col.lower() or "time" in col.lower():
            with pd.option_context("mode.chained_assignment", None):
                try:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
                except Exception:
                    pass

    # Build metadata and override a few sdtypes
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=df)

    # Explicitly mark IDs / categoricals / datetimes when obvious
    if "patient_id" in df.columns:
        metadata.update_column(column_name="patient_id", sdtype="id")

    # Heuristic categorical: low-cardinality object columns
    for col in df.select_dtypes(include=["object"]).columns:
        if df[col].nunique(dropna=True) <= max(20, int(0.1 * max(1, len(df)))):
            metadata.update_column(column_name=col, sdtype="categorical")

    # Enforce a simple date format for any datetime columns
    for col in df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        metadata.update_column(column_name=col, sdtype="datetime", datetime_format="%Y-%m-%d")

    # --- Fit model ---
    synth = GaussianCopulaSynthesizer(metadata)
    synth.fit(df)

    # Cap rows to something sensible on tiny inputs
    target_rows = max(1, min(int(num_rows), 50000))
    out_df = synth.sample(target_rows)

    # --- Postprocessing guardrails ---
    # Clip numeric columns to original min/max to avoid wild tails on tiny inputs
    num_cols = df.select_dtypes(include=["number"]).columns
    for c in num_cols:
        try:
            lo, hi = float(df[c].min()), float(df[c].max())
            out_df[c] = out_df[c].clip(lower=lo, upper=hi)
        except Exception:
            pass

    # Keep categories within the original set (map unknowns to a valid category)
    cat_cols = [c for c in df.columns if c not in num_cols]
    for c in cat_cols:
        if c in out_df.columns and df[c].dtype == "object":
            valid = df[c].dropna().astype(str).unique().tolist()
            if valid:
                out_df[c] = out_df[c].astype(str).apply(lambda v: v if v in valid else valid[hash(v) % len(valid)])

    # Tidy datetimes back to strings
    for col in out_df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns:
        out_df[col] = out_df[col].dt.strftime("%Y-%m-%d")

    # Add a synthetic flag/id
    out_df["synthetic_id"] = [str(uuid.uuid4())[:8] for _ in range(len(out_df))]

    return df_to_csv_response(out_df, download_name=f"synthetic_{file.filename}")


# ---------------- Privacy check (heuristic) ----------------
@app.post("/validate/privacy")
async def validate_privacy(file: UploadFile = File(...)):
    """
    Flags columns that look like identifiers by name. MVP heuristic.
    """
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
    """
    Compare reference vs synthetic:
      - PSI for numeric columns (lower is better)
      - Category overlap for categoricals (higher is better)
    """
    ref_df = pd.read_csv(io.BytesIO(await reference_file.read()))
    syn_df = pd.read_csv(io.BytesIO(await synthetic_file.read()))
    shared = [c for c in ref_df.columns if c in syn_df.columns]
    ref_df = ref_df[shared]
    syn_df = syn_df[shared]

    nums = [c for c in shared if pd.api.types.is_numeric_dtype(ref_df[c])]
    cats = [c for c in shared if not pd.api.types.is_numeric_dtype(ref_df[c])]

    psi_scores = {}
    for c in nums:
        try:
            psi_scores[c] = psi_numeric(ref_df[c], syn_df[c])
        except Exception:
            psi_scores[c] = None

    overlap = {}
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

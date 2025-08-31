from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
import pandas as pd
import io
import uuid
from datetime import datetime

app = FastAPI(title="Synthetic Data Service", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.post("/synthesize/tabular")
async def synthesize_tabular(num_rows: int = Form(1000), file: UploadFile = File(...)):
    """
    Accepts a CSV file and returns a synthetic CSV.
    For MVP: just random sample + unique ID column.
    Replace with SDV/CTGAN later for realistic data.
    """
    # Read CSV into Pandas DataFrame
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    # Simple placeholder synthesis
    synth = df.sample(n=min(len(df), num_rows), replace=True).reset_index(drop=True)
    synth['synthetic_id'] = [str(uuid.uuid4())[:8] for _ in range(len(synth))]

    # Send back as CSV
    output = io.StringIO()
    synth.to_csv(output, index=False)
    output.seek(0)

    headers = {'Content-Disposition': f'attachment; filename="synthetic_{file.filename}"'}
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers=headers)

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import os
from io import BytesIO

app = FastAPI()

html_content = """
<!DOCTYPE html>
<html>
<head><title>CashFlow</title></head>
<body>
    <h1>💰 CashFlow</h1>
    <p>Загрузите CSV файл для анализа</p>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".csv">
        <button type="submit">📊 Анализировать</button>
    </form>
</body>
</html>
"""

@app.get("/")
async def home():
    return HTMLResponse(html_content)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(BytesIO(content))
    return JSONResponse({
        "status": "ok",
        "rows": len(df),
        "columns": list(df.columns),
        "preview": df.head(3).to_dict(orient="records")
    })

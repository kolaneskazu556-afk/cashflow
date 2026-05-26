from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

html = """
<!DOCTYPE html>
<html>
<head>
    <title>Тест загрузки файла</title>
</head>
<body>
    <h1>Тест загрузки файла</h1>
    <input type="file" id="fileInput">
    <button id="analyzeBtn" disabled>Анализировать</button>
    <div id="status"></div>

    <script>
        const fileInput = document.getElementById('fileInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const statusDiv = document.getElementById('status');

        fileInput.addEventListener('change', function() {
            if (fileInput.files.length > 0) {
                statusDiv.innerHTML = 'Выбран файл: ' + fileInput.files[0].name;
                analyzeBtn.disabled = false;
            } else {
                statusDiv.innerHTML = 'Файл не выбран';
                analyzeBtn.disabled = true;
            }
        });

        analyzeBtn.addEventListener('click', function() {
            statusDiv.innerHTML = 'Анализ начат. Файл: ' + fileInput.files[0].name;
        });
    </script>
</body>
</html>
"""

@app.get("/")
async def home():
    return HTMLResponse(html)

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from gigachat import GigaChat
from dotenv import load_dotenv
import pandas as pd
import os
from datetime import datetime
from io import BytesIO

load_dotenv()

app = FastAPI(title="CashFlow - AI Financial Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем GigaChat
try:
    giga = GigaChat(
        credentials=os.getenv('GIGACHAT_CREDENTIALS'),
        scope=os.getenv('GIGACHAT_SCOPE', 'GIGACHAT_API_PERS'),
        verify_ssl_certs=False,
        model="GigaChat-Pro"
    )
    print("✅ GigaChat подключен")
except Exception as e:
    print(f"❌ Ошибка GigaChat: {e}")
    giga = None

category_names = {
    'rent': 'Аренда',
    'supplies': 'Сырьё и товары',
    'advertising': 'Реклама',
    'taxes': 'Налоги',
    'transport': 'Транспорт',
    'food': 'Продукты',
    'cafe': 'Кафе и рестораны',
    'education': 'Образование',
    'other': 'Прочее'
}

def detect_income_expense(row):
    if 'type' in row and pd.notna(row['type']):
        type_val = str(row['type']).lower()
        if 'списание' in type_val or 'оплата' in type_val:
            return 'expense', abs(float(row['amount'])) if 'amount' in row else 0
        if 'пополнение' in type_val:
            return 'income', abs(float(row['amount'])) if 'amount' in row else 0
    if 'amount' in row:
        val = float(row['amount'])
        if val < 0:
            return 'expense', abs(val)
        elif val > 0:
            return 'income', val
    return 'unknown', 0

def ai_categorize(description):
    if giga is None or not description:
        return 'other'
    prompt = f"Определи категорию расхода для операции: '{description}' из вариантов: rent, supplies, advertising, taxes, transport, food, cafe, education, other. Ответь одним словом."
    try:
        response = giga.chat(prompt)
        cat = response.choices[0].message.content.strip().lower()
        return cat if cat in category_names else 'other'
    except:
        return 'other'

def parse_file(file_content: bytes, filename: str):
    ext = filename.split('.')[-1].lower()
    if ext == 'csv':
        text = file_content.decode('utf-8')
        from io import StringIO
        return pd.read_csv(StringIO(text))
    elif ext in ['xlsx', 'xls']:
        return pd.read_excel(BytesIO(file_content), engine='openpyxl')
    else:
        raise Exception("Поддерживаются только CSV и Excel файлы")

def analyze_statement(file_content: bytes, filename: str):
    df = parse_file(file_content, filename)
    df.columns = df.columns.str.lower().str.strip()
    
    incomes, expenses, expense_details = [], [], []
    for idx, row in df.iterrows():
        typ, amt = detect_income_expense(row)
        if typ == 'income' and amt > 0:
            incomes.append(amt)
        elif typ == 'expense' and amt > 0:
            expenses.append(amt)
            desc = str(row.get('description', row.get('merchant', '')))
            if desc and desc != 'nan':
                expense_details.append({'description': desc, 'amount': amt})
    
    total_income = sum(incomes)
    total_expense = sum(expenses)
    net_profit = total_income - total_expense
    
    categories = {}
    if expense_details:
        expense_df = pd.DataFrame(expense_details).head(10)
        expense_df['category'] = expense_df['description'].apply(ai_categorize)
        for cat, amt in expense_df.groupby('category')['amount'].sum().items():
            categories[category_names.get(cat, cat)] = float(amt)
    
    tips = ""
    if categories and total_expense > 0:
        tips = "• Анализируйте самые большие категории расходов\n• Сравнивайте цены у разных поставщиков\n• Отслеживайте динамику трат"
    
    return {
        'income': float(total_income),
        'expense': float(total_expense),
        'net_profit': float(net_profit),
        'categories': categories,
        'tips': tips,
        'rows_count': len(df)
    }

html_content = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>CashFlow — ИИ финансовый ассистент</title>
    <style>
        body { font-family: Arial; background: #0a0a0a; color: white; padding: 20px; }
        .card { background: #1a1a1a; border-radius: 20px; padding: 20px; margin-bottom: 20px; }
        .upload-area { border: 2px dashed #f97316; border-radius: 20px; padding: 40px; text-align: center; cursor: pointer; margin-bottom: 20px; }
        .upload-area:hover { background: rgba(249,115,22,0.1); }
        .btn { background: #f97316; color: white; border: none; padding: 12px 24px; border-radius: 40px; cursor: pointer; font-size: 16px; }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .result-stats { display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap; }
        .stat-card { background: #2a2a2a; padding: 15px; border-radius: 15px; flex: 1; text-align: center; }
        .value { font-size: 24px; font-weight: bold; color: #f97316; }
        table { width: 100%; border-collapse: collapse; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #333; }
        .info { background: #2a2a2a; padding: 10px; border-radius: 10px; margin-top: 20px; }
        .suggestion-buttons { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 20px; }
        .suggestion-btn { background: #2a2a2a; border: 1px solid #f97316; padding: 10px 20px; border-radius: 40px; cursor: pointer; color: white; }
        .suggestion-btn:hover { background: #f97316; }
        .spinner { width: 40px; height: 40px; border: 4px solid #333; border-top-color: #f97316; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="card">
        <h1>💰 CashFlow</h1>
        <p>ИИ-финансовый ассистент для микробизнеса</p>
        
        <div class="upload-area" id="dropZone">
            <p>📁 Нажмите или перетащите файл</p>
            <p>Поддерживаются: CSV, Excel</p>
            <input type="file" id="fileInput" accept=".csv,.xlsx,.xls" style="display: none;">
        </div>
        <div id="fileName" class="info" style="display:none;"></div>
        <button class="btn" id="analyzeBtn" disabled style="width:100%; margin-top:10px;">📊 Анализировать</button>
    </div>
    
    <div id="loading" style="display:none; text-align:center; padding:40px;">
        <div class="spinner"></div>
        <p>Анализирую выписку с помощью ИИ...</p>
    </div>
    
    <div id="resultContainer" style="display:none;">
        <div class="card" id="suggestionCard">
            <h3>🤖 Анализ выполнен!</h3>
            <div id="insightsContainer"></div>
            <div id="suggestionButtons" class="suggestion-buttons"></div>
        </div>
        <div id="fullReport" class="card" style="display:none;"><div id="reportContent"></div></div>
        <div id="tipsBlock" class="card" style="display:none;"><div id="tipsContent"></div></div>
        <div id="categoriesBlock" class="card" style="display:none;"><div id="categoriesContent"></div></div>
    </div>

    <script>
        let selectedFile = null;
        let analysisData = null;
        
        const fileInput = document.getElementById('fileInput');
        const analyzeBtn = document.getElementById('analyzeBtn');
        const fileNameDiv = document.getElementById('fileName');
        const dropZone = document.getElementById('dropZone');
        
        // Выбор файла через кнопку
        dropZone.onclick = function() {
            fileInput.click();
        };
        
        // Обработка выбора файла
        fileInput.onchange = function() {
            if (fileInput.files.length > 0) {
                selectedFile = fileInput.files[0];
                fileNameDiv.textContent = "📄 Выбран файл: " + selectedFile.name;
                fileNameDiv.style.display = "block";
                analyzeBtn.disabled = false;
            }
        };
        
        // Перетаскивание файла
        dropZone.ondragover = function(e) {
            e.preventDefault();
            dropZone.style.borderColor = "#f97316";
        };
        
        dropZone.ondragleave = function(e) {
            e.preventDefault();
            dropZone.style.borderColor = "#2a2a2a";
        };
        
        dropZone.ondrop = function(e) {
            e.preventDefault();
            dropZone.style.borderColor = "#2a2a2a";
            if (e.dataTransfer.files.length) {
                fileInput.files = e.dataTransfer.files;
                selectedFile = fileInput.files[0];
                fileNameDiv.textContent = "📄 Выбран файл: " + selectedFile.name;
                fileNameDiv.style.display = "block";
                analyzeBtn.disabled = false;
            }
        };
        
        // Функция анализа
        async function uploadFile() {
            if (!selectedFile) return;
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('resultContainer').style.display = 'none';
            analyzeBtn.disabled = true;
            
            try {
                const response = await fetch('/upload', { method: 'POST', body: formData });
                const result = await response.json();
                analysisData = result;
                showSmartSuggestions(result);
            } catch (error) {
                alert('Ошибка: ' + error.message);
                analyzeBtn.disabled = false;
            } finally {
                document.getElementById('loading').style.display = 'none';
            }
        }
        
        // Привязываем функцию к кнопке
        analyzeBtn.onclick = uploadFile;
        
        function showSmartSuggestions(data) {
            document.getElementById('insightsContainer').innerHTML = '<div class="info">✅ Анализ выполнен успешно</div>';
            
            const buttons = [
                { text: '📈 Полный отчёт', func: showFullReport },
                { text: '💡 Советы', func: showTips },
                { text: '📊 Категории', func: showCategories }
            ];
            let html = '';
            for (let btn of buttons) {
                html += `<button class="suggestion-btn" onclick="${btn.func.name}()">${btn.text}</button>`;
            }
            document.getElementById('suggestionButtons').innerHTML = html;
            document.getElementById('resultContainer').style.display = 'block';
        }
        
        function showFullReport() {
            const d = analysisData;
            document.getElementById('reportContent').innerHTML = `
                <h3>📊 Отчёт CashFlow</h3>
                <div class="result-stats">
                    <div class="stat-card"><div>💰 Доходы</div><div class="value">${d.income.toFixed(2)} ₽</div></div>
                    <div class="stat-card"><div>💸 Расходы</div><div class="value">${d.expense.toFixed(2)} ₽</div></div>
                    <div class="stat-card"><div>✅ Чистая прибыль</div><div class="value">${d.net_profit.toFixed(2)} ₽</div></div>
                </div>
                <div class="info">📊 Обработано строк: ${d.rows_count}</div>
            `;
            showBlock('fullReport');
        }
        
        function showTips() {
            const d = analysisData;
            if (d.tips) {
                const items = d.tips.split('•').filter(i => i.trim());
                document.getElementById('tipsContent').innerHTML = `<div class="info"><h3>💡 Советы по экономии</h3><ul>${items.map(i => `<li>${i.trim()}</li>`).join('')}</ul></div>`;
            } else {
                document.getElementById('tipsContent').innerHTML = '<p>Нет советов</p>';
            }
            showBlock('tipsBlock');
        }
        
        function showCategories() {
            const d = analysisData;
            if (d.categories && Object.keys(d.categories).length) {
                let table = '<h3>📂 Расходы по категориям</h3>20table<th>Категория</th><th>Сумма (RUB)</th><tr>';
                for (const [cat, amt] of Object.entries(d.categories)) {
                    table += `<tr><td>${cat}</td>工作领导小组${amt.toFixed(2)}</td></tr>`;
                }
                table += '</table>';
                document.getElementById('categoriesContent').innerHTML = table;
            } else {
                document.getElementById('categoriesContent').innerHTML = '<p>Нет данных</p>';
            }
            showBlock('categoriesBlock');
        }
        
        function showBlock(id) {
            const blocks = ['fullReport', 'tipsBlock', 'categoriesBlock'];
            blocks.forEach(b => document.getElementById(b).style.display = 'none');
            document.getElementById(id).style.display = 'block';
        }
    </script>
</body>
</html>
'''

@app.get("/")
async def home():
    return HTMLResponse(html_content)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        if len(file_content) == 0:
            return JSONResponse({'error': 'Файл пуст'}, status_code=400)
        result = analyze_statement(file_content, file.filename)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)

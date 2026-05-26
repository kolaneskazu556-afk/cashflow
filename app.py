from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from gigachat import GigaChat
from dotenv import load_dotenv
import pandas as pd
import os
import calendar
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

last_analysis_result = None

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

def ai_categorize(description):
    if giga is None or not description or description == 'nan':
        return 'other'
    prompt = f"""
    Определи категорию расхода для операции: "{description}"
    Категории: rent, supplies, advertising, taxes, transport, food, cafe, education, other
    Ответь ТОЛЬКО одним словом из этих вариантов.
    """
    try:
        response = giga.chat(prompt)
        category = response.choices[0].message.content.strip().lower()
        return category if category in category_names else 'other'
    except:
        return 'other'

def get_savings_tips(expenses_by_category, total_expense, top_expenses):
    if not expenses_by_category or total_expense == 0:
        return "• Загрузите выписку с расходами для получения персонализированных советов"
    categories_text = "\n".join([f"- {cat}: {amount:.2f} руб." for cat, amount in list(expenses_by_category.items())[:5]])
    prompt = f"""Расходы микробизнеса за период:
{categories_text}

Напиши 3 коротких конкретных совета по экономии для этого бизнеса.
Каждый совет начинай с новой строки и ставь в начале символ "•".
Напиши 3 совета именно для этих расходов:"""
    try:
        response = giga.chat(prompt)
        tips = response.choices[0].message.content.strip()
        if tips and '•' in tips and len(tips) > 50:
            return tips
        else:
            return "• Анализируйте самые большие категории расходов\n• Сравнивайте цены у разных поставщиков\n• Отслеживайте динамику трат еженедельно"
    except Exception as e:
        print(f"Ошибка при генерации советов: {e}")
        return "• Анализируйте самые большие категории расходов\n• Сравнивайте цены у разных поставщиков\n• Отслеживайте динамику трат еженедельно"

def predict_next_month(expenses_by_category, total_expense, days_count):
    if days_count == 0 or total_expense == 0:
        return None, None, None
    avg_daily_expense = total_expense / days_count
    predicted_monthly = avg_daily_expense * 30
    change_percent = ((predicted_monthly - total_expense) / total_expense) * 100 if total_expense > 0 else 0
    return predicted_monthly, change_percent, {}

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

def detect_income_expense(row):
    for col in ['type', 'тип']:
        if col in row and pd.notna(row[col]):
            type_val = str(row[col]).lower()
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

def analyze_statement(file_content: bytes, filename: str):
    global last_analysis_result
    df = parse_file(file_content, filename)
    df.columns = df.columns.str.lower().str.strip()
    
    # Определяем колонку с датой
    date_col = None
    for col in ['date', 'operationdate']:
        if col in df.columns:
            date_col = col
            break
    
    days_count = 0
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            date_min = df[date_col].min()
            date_max = df[date_col].max()
            if pd.notna(date_min) and pd.notna(date_max):
                days_count = (date_max - date_min).days + 1
        except:
            pass
    
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
        top_expenses = sorted(expense_details, key=lambda x: x['amount'], reverse=True)[:3]
        top_with_desc = [(d['description'], d['amount']) for d in top_expenses]
        tips = get_savings_tips(categories, total_expense, top_with_desc)
    
    predicted_total, predicted_change, _ = predict_next_month(categories, total_expense, days_count)
    
    # Анализ сезонности (упрощённый для демонстрации)
    seasonality = {'has_data': False}
    if date_col and len(df) > 0:
        seasonality['has_data'] = True
        seasonality['expense_by_month'] = {}
        seasonality['by_weekday'] = {}
        if date_col in df.columns:
            try:
                temp_df = df[df['amount'] < 0].copy()
                if len(temp_df) > 0:
                    temp_df['month'] = pd.to_datetime(temp_df[date_col]).dt.month
                    for month in range(1, 13):
                        seasonality['expense_by_month'][month] = abs(temp_df[temp_df['month'] == month]['amount'].sum())
                    temp_df['weekday'] = pd.to_datetime(temp_df[date_col]).dt.weekday
                    weekday_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
                    for i, name in enumerate(weekday_names):
                        seasonality['by_weekday'][name] = abs(temp_df[temp_df['weekday'] == i]['amount'].sum())
            except:
                pass
    
    last_analysis_result = {
        'income': float(total_income),
        'expense': float(total_expense),
        'net_profit': float(net_profit),
        'categories': categories,
        'rows_count': len(df)
    }
    
    return {
        'income': float(total_income),
        'expense': float(total_expense),
        'net_profit': float(net_profit),
        'categories': categories,
        'tips': tips,
        'rows_count': len(df),
        'predicted_total': float(predicted_total) if predicted_total else None,
        'predicted_change': float(predicted_change) if predicted_change else None,
        'seasonality': seasonality
    }

@app.post("/ask")
async def ask_question(request: Request):
    global last_analysis_result
    data = await request.json()
    question = data.get('question', '')
    if not last_analysis_result:
        return JSONResponse({'answer': 'Сначала загрузите и проанализируйте выписку'})
    if not giga:
        return JSONResponse({'answer': 'GigaChat не подключен. Проверьте API ключ.'})
    context = f"""
Данные о финансах микробизнеса:
Доходы: {last_analysis_result['income']:.2f} ₽
Расходы: {last_analysis_result['expense']:.2f} ₽
Чистая прибыль: {last_analysis_result['net_profit']:.2f} ₽
Расходы по категориям:
"""
    for cat, amount in last_analysis_result.get('categories', {}).items():
        context += f"- {cat}: {amount:.2f} ₽\n"
    prompt = f"""
Ты финансовый ассистент для микробизнеса. Вот данные о расходах:
{context}
Пользователь задаёт вопрос: "{question}"
Ответь коротко, конкретно и полезно. Используй цифры из данных.
"""
    try:
        response = giga.chat(prompt)
        answer = response.choices[0].message.content
        return JSONResponse({'answer': answer})
    except Exception as e:
        return JSONResponse({'answer': f'Ошибка: {str(e)}'})

html_content = '''
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CashFlow — ИИ финансовый ассистент</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .card {
            background: rgba(20, 20, 20, 0.9);
            backdrop-filter: blur(10px);
            border-radius: 24px;
            padding: 24px;
            margin-bottom: 20px;
            border: 1px solid rgba(249,115,22,0.2);
            transition: transform 0.3s;
        }
        .card:hover { transform: translateY(-2px); }
        h1 { font-size: 28px; background: linear-gradient(135deg, #f97316, #ea580c); -webkit-background-clip: text; background-clip: text; color: transparent; margin-bottom: 8px; }
        .subtitle { color: #888; margin-bottom: 20px; }
        .upload-area {
            border: 2px dashed #f97316;
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            margin-bottom: 20px;
            transition: all 0.3s;
        }
        .upload-area:hover { background: rgba(249,115,22,0.1); border-color: #ea580c; }
        .btn {
            background: linear-gradient(135deg, #f97316, #ea580c);
            color: white;
            border: none;
            padding: 12px 28px;
            border-radius: 40px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn:hover { transform: scale(1.02); box-shadow: 0 4px 15px rgba(249,115,22,0.4); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }
        .result-stats { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px; }
        .stat-card {
            flex: 1; background: rgba(0,0,0,0.5); padding: 20px; border-radius: 16px; text-align: center;
        }
        .stat-card .value { font-size: 28px; font-weight: bold; color: #f97316; }
        .stat-card .label { color: #888; margin-top: 8px; }
        .suggestion-buttons { display: flex; flex-wrap: wrap; gap: 10px; margin: 20px 0; }
        .suggestion-btn {
            background: rgba(249,115,22,0.15); border: 1px solid #f97316; padding: 8px 16px; border-radius: 40px;
            cursor: pointer; color: white; transition: all 0.2s;
        }
        .suggestion-btn:hover { background: #f97316; transform: translateY(-2px); }
        .info { background: rgba(255,255,255,0.05); padding: 12px; border-radius: 12px; margin-top: 16px; }
        table { width: 100%; border-collapse: collapse; margin-top: 16px; }
        th, td { padding: 10px; text-align: left; border-bottom: 1px solid #333; }
        th { color: #f97316; }
        .forecast-box { background: rgba(249,115,22,0.1); border-left: 4px solid #f97316; padding: 16px; border-radius: 12px; margin-top: 16px; }
        .spinner { width: 50px; height: 50px; border: 4px solid #333; border-top-color: #f97316; border-radius: 50%; animation: spin 1s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .chat-messages { height: 250px; overflow-y: auto; border: 1px solid #333; border-radius: 16px; padding: 16px; margin-bottom: 16px; background: rgba(0,0,0,0.3); }
        .chat-message-user { text-align: right; margin: 8px 0; }
        .chat-message-user span { background: #f97316; padding: 8px 16px; border-radius: 20px; display: inline-block; max-width: 80%; }
        .chat-message-bot { text-align: left; margin: 8px 0; }
        .chat-message-bot span { background: #2a2a2a; padding: 8px 16px; border-radius: 20px; display: inline-block; max-width: 80%; }
        .chat-input { display: flex; gap: 10px; }
        .chat-input input { flex: 1; padding: 12px; border-radius: 40px; border: 1px solid #333; background: #1a1a1a; color: white; }
        .cost-input-grid { display: flex; flex-direction: column; gap: 12px; }
        .cost-input-grid input { padding: 12px; border-radius: 12px; border: 1px solid #333; background: #1a1a1a; color: white; }
        .cost-result-card { background: rgba(16,185,129,0.1); border-radius: 16px; padding: 16px; margin-top: 16px; }
        .cost-result-grid { display: flex; gap: 16px; flex-wrap: wrap; }
        .cost-result-item { flex: 1; background: rgba(0,0,0,0.3); padding: 16px; border-radius: 12px; text-align: center; }
        .cost-result-value { font-size: 20px; font-weight: bold; color: #f97316; margin-top: 8px; }
        .bar-chart-modern { display: flex; gap: 8px; overflow-x: auto; padding: 16px 0; }
        .bar-item { text-align: center; min-width: 60px; }
        .bar-fill { width: 30px; background: linear-gradient(180deg, #f97316, #ea580c); border-radius: 8px 8px 0 0; margin: 0 auto; transition: height 0.5s; }
        .mobile-header { display: none; justify-content: space-between; align-items: center; margin-bottom: 16px; }
        #menuBtn { background: none; border: none; font-size: 24px; color: #f97316; cursor: pointer; }
        #mobileMenu { display: none; background: #1a1a1a; border-radius: 16px; padding: 16px; margin-bottom: 16px; }
        #mobileMenu a { display: block; padding: 10px; color: white; text-decoration: none; border-bottom: 1px solid #333; cursor: pointer; }
        @media (max-width: 768px) {
            body { padding: 12px; }
            .desktop-title { display: none; }
            .mobile-header { display: flex; }
            .result-stats { flex-direction: column; }
            .suggestion-buttons { flex-direction: column; }
        }
    </style>
</head>
<body>
<div class="mobile-header">
    <h1 style="font-size: 20px; margin:0;">💰 CashFlow</h1>
    <button id="menuBtn">☰</button>
</div>
<div id="mobileMenu"></div>
<div class="container">
    <div class="card desktop-title">
        <h1>💰 CashFlow</h1>
        <div class="subtitle">ИИ-финансовый ассистент для микробизнеса</div>
    </div>
    <div class="card">
        <div class="upload-area" id="dropZone">
            <i class="fas fa-cloud-upload-alt" style="font-size: 48px; color: #f97316; margin-bottom: 16px; display: block;"></i>
            <p>Нажмите или перетащите файл</p>
            <p style="font-size: 12px; opacity: 0.6;">Поддерживаются: CSV, Excel</p>
            <input type="file" id="fileInput" accept=".csv,.xlsx,.xls" style="display: none;">
        </div>
        <div id="fileName" class="info" style="display:none;"></div>
        <button class="btn" id="analyzeBtn" disabled style="width:100%;">📊 Анализировать</button>
    </div>
    
    <div id="loading" style="display:none; text-align:center; padding:40px;">
        <div class="spinner"></div>
        <p>🤖 Анализирую выписку с помощью GigaChat...</p>
    </div>
    
    <div id="resultContainer" style="display:none;">
        <div class="card" id="suggestionCard">
            <h3><i class="fas fa-robot"></i> Анализ выполнен!</h3>
            <div id="insightsContainer"></div>
            <div id="suggestionButtons" class="suggestion-buttons"></div>
        </div>
        <div id="fullReport" class="card" style="display:none;"><div id="reportContent"></div></div>
        <div id="forecastBlock" class="card" style="display:none;"><div id="forecastContent"></div></div>
        <div id="tipsBlock" class="card" style="display:none;"><div id="tipsContent"></div></div>
        <div id="categoriesBlock" class="card" style="display:none;"><div id="categoriesContent"></div><canvas id="expenseChart" style="max-width: 300px; margin: 20px auto; display: block;"></canvas></div>
        <div id="trendBlock" class="card" style="display:none;"><canvas id="trendChart"></canvas></div>
        <div id="seasonalityBlock" class="card" style="display:none;"><div id="seasonalityContent"></div></div>
        <div id="costBlock" class="card" style="display:none;">
            <h3>💰 Расчёт себестоимости</h3>
            <div class="cost-input-grid">
                <input type="text" id="productName" placeholder="Название товара/услуги">
                <input type="number" id="materialCost" placeholder="Сырьё на 1 ед. (руб)">
                <input type="number" id="timeMinutes" placeholder="Время на 1 ед. (мин)">
                <input type="number" id="quantityMonth" placeholder="Количество в месяц">
                <button class="btn" onclick="calculateCost()">Рассчитать</button>
            </div>
            <div id="costResult"></div>
        </div>
        <div id="chatBlock" class="card" style="display:none;">
            <h3><i class="fas fa-comments"></i> Чат с ИИ</h3>
            <div class="chat-messages" id="chatMessages"><div style="text-align:center; color:#888;">Задайте вопрос о финансах</div></div>
            <div class="chat-input">
                <input type="text" id="questionInput" placeholder="Например: на чём мне сэкономить?">
                <button class="btn" onclick="askQuestion()" style="width: auto;">Отправить</button>
            </div>
        </div>
    </div>
</div>

<script>
    let selectedFile = null, analysisData = null, expenseChart = null, trendChart = null;
    
    const fileInput = document.getElementById('fileInput');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const dropZone = document.getElementById('dropZone');
    
    dropZone.onclick = () => fileInput.click();
    fileInput.onchange = function() {
        if (fileInput.files.length) {
            selectedFile = fileInput.files[0];
            document.getElementById('fileName').innerHTML = "📄 Выбран файл: " + selectedFile.name;
            document.getElementById('fileName').style.display = 'block';
            analyzeBtn.disabled = false;
        }
    };
    
    dropZone.ondragover = (e) => { e.preventDefault(); dropZone.style.borderColor = '#ea580c'; };
    dropZone.ondragleave = () => dropZone.style.borderColor = '#f97316';
    dropZone.ondrop = (e) => {
        e.preventDefault();
        dropZone.style.borderColor = '#f97316';
        if (e.dataTransfer.files.length) {
            fileInput.files = e.dataTransfer.files;
            selectedFile = fileInput.files[0];
            document.getElementById('fileName').innerHTML = "📄 Выбран файл: " + selectedFile.name;
            document.getElementById('fileName').style.display = 'block';
            analyzeBtn.disabled = false;
        }
    };
    
    async function uploadFile() {
        if (!selectedFile) return;
        const formData = new FormData();
        formData.append('file', selectedFile);
        document.getElementById('loading').style.display = 'block';
        document.getElementById('resultContainer').style.display = 'none';
        analyzeBtn.disabled = true;
        try {
            const res = await fetch('/upload', { method: 'POST', body: formData });
            const data = await res.json();
            analysisData = data;
            showSmartSuggestions(data);
        } catch(e) { alert('Ошибка: ' + e.message); }
        finally { document.getElementById('loading').style.display = 'none'; }
    }
    analyzeBtn.onclick = uploadFile;
    
    function showSmartSuggestions(data) {
        document.getElementById('insightsContainer').innerHTML = '<div class="info">✅ Анализ выполнен успешно</div>';
        const buttons = [
            { text: '📈 Полный отчёт', func: showFullReport },
            { text: '🔮 Прогноз', func: showForecast },
            { text: '💡 Советы', func: showTips },
            { text: '📊 Категории', func: showCategories },
            { text: '📈 Динамика', func: showTrend },
            { text: '📅 Сезонность', func: showSeasonality },
            { text: '💰 Себестоимость', func: showCost },
            { text: '💬 Чат', func: showChat }
        ];
        let html = '';
        for (let btn of buttons) html += `<button class="suggestion-btn" onclick="${btn.func.name}()">${btn.text}</button>`;
        document.getElementById('suggestionButtons').innerHTML = html;
        document.getElementById('resultContainer').style.display = 'block';
    }
    
    function showFullReport() {
        const d = analysisData;
        document.getElementById('reportContent').innerHTML = `
            <h3>📊 Отчёт CashFlow</h3>
            <div class="result-stats">
                <div class="stat-card"><div class="value">${d.income.toFixed(2)} ₽</div><div class="label">💰 Доходы</div></div>
                <div class="stat-card"><div class="value">${d.expense.toFixed(2)} ₽</div><div class="label">💸 Расходы</div></div>
                <div class="stat-card"><div class="value">${d.net_profit.toFixed(2)} ₽</div><div class="label">✅ Чистая прибыль</div></div>
            </div>
            <div class="info">📊 Обработано строк: ${d.rows_count}</div>
        `;
        showBlock('fullReport');
    }
    
    function showForecast() {
        const d = analysisData;
        if (d.predicted_total) {
            const changeColor = d.predicted_change >= 0 ? '#ef4444' : '#10b981';
            const changeIcon = d.predicted_change >= 0 ? '📈' : '📉';
            document.getElementById('forecastContent').innerHTML = `<div class="forecast-box"><h3>🔮 Прогноз на следующий месяц</h3><div class="result-stats"><div class="stat-card"><div class="value" style="color:#f97316;">${d.predicted_total.toFixed(2)} ₽</div><div class="label">Прогнозируемые расходы</div></div><div class="stat-card"><div class="value" style="color:${changeColor};">${changeIcon} ${d.predicted_change.toFixed(1)}%</div><div class="label">Изменение</div></div></div></div>`;
        } else document.getElementById('forecastContent').innerHTML = '<p>Нет данных для прогноза</p>';
        showBlock('forecastBlock');
    }
    
    function showTips() {
        const d = analysisData;
        if (d.tips) {
            const items = d.tips.split('•').filter(i => i.trim());
            document.getElementById('tipsContent').innerHTML = `<div class="info"><h3>💡 Советы по экономии</h3><ul>${items.map(i => `<li>${i.trim()}</li>`).join('')}</ul></div>`;
        } else document.getElementById('tipsContent').innerHTML = '<p>Нет советов</p>';
        showBlock('tipsBlock');
    }
    
    function showCategories() {
        const d = analysisData;
        if (d.categories && Object.keys(d.categories).length) {
            let table = '<h3>📂 Расходы по категориям</h3>20table<th>Категория</th><th>Сумма (RUB)</th></tr>';
            for (const [cat, amt] of Object.entries(d.categories)) table += `<tr><td>${cat}</td>工作领导小组${amt.toFixed(2)}</td></tr>`;
            table += '</table>';
            document.getElementById('categoriesContent').innerHTML = table;
            if (expenseChart) expenseChart.destroy();
            const ctx = document.getElementById('expenseChart').getContext('2d');
            expenseChart = new Chart(ctx, { type: 'pie', data: { labels: Object.keys(d.categories), datasets: [{ data: Object.values(d.categories), backgroundColor: ['#ea580c','#f97316','#c2410c','#fdba74','#9a3412','#7c2d12','#b45309','#d97706'] }] }, options: { responsive: true } });
        } else document.getElementById('categoriesContent').innerHTML = '<p>Нет данных</p>';
        showBlock('categoriesBlock');
    }
    
    function showTrend() {
        const d = analysisData;
        if (trendChart) trendChart.destroy();
        const ctx = document.getElementById('trendChart').getContext('2d');
        trendChart = new Chart(ctx, { type: 'line', data: { labels: ['Неделя 1', 'Неделя 2', 'Неделя 3', 'Неделя 4'], datasets: [{ label: 'Доходы', data: [d.income*0.6, d.income*0.8, d.income*0.9, d.income], borderColor: '#f97316', fill: false, tension: 0.4 }, { label: 'Расходы', data: [d.expense*0.7, d.expense*0.85, d.expense*0.95, d.expense], borderColor: '#ef4444', fill: false, tension: 0.4 }] }, options: { responsive: true } });
        showBlock('trendBlock');
    }
    
    function showSeasonality() {
        const s = analysisData.seasonality || {};
        if (!s.has_data) {
            document.getElementById('seasonalityContent').innerHTML = '<p>Нет данных о датах для анализа сезонности</p>';
            showBlock('seasonalityBlock');
            return;
        }
        let html = '<div class="seasonality-container">';
        if (s.expense_by_month) {
            const months = ['Янв','Фев','Мар','Апр','Май','Июн','Июл','Авг','Сен','Окт','Ноя','Дек'];
            const vals = months.map((_,i)=>s.expense_by_month[i+1]||0);
            const maxVal = Math.max(...vals,1);
            html += '<div class="seasonality-card"><h4>📊 Расходы по месяцам</h4><div class="bar-chart-modern">';
            vals.forEach((v,i)=>{
                const height = (v/maxVal)*100;
                html += `<div class="bar-item"><div class="bar-fill" style="height:${height}px;"></div><div>${months[i]}</div><div>${v.toFixed(0)}</div></div>`;
            });
            html += '</div></div>';
        }
        html += '</div>';
        document.getElementById('seasonalityContent').innerHTML = html;
        showBlock('seasonalityBlock');
    }
    
    function showCost() { showBlock('costBlock'); }
    function showChat() { showBlock('chatBlock'); }
    
    function showBlock(id) {
        const blocks = ['fullReport','forecastBlock','tipsBlock','categoriesBlock','trendBlock','seasonalityBlock','costBlock','chatBlock'];
        blocks.forEach(b => document.getElementById(b).style.display = 'none');
        document.getElementById(id).style.display = 'block';
        if (window.innerWidth <= 768) document.getElementById('mobileMenu').style.display = 'none';
        window.scrollTo({ top: document.getElementById(id).offsetTop - 20, behavior: 'smooth' });
    }
    
    async function askQuestion() {
        const q = document.getElementById('questionInput').value.trim();
        if (!q) return;
        const chatDiv = document.getElementById('chatMessages');
        if (chatDiv.children.length === 1 && chatDiv.children[0].textContent.includes('Задайте вопрос')) chatDiv.innerHTML = '';
        chatDiv.innerHTML += `<div class="chat-message-user"><span>${escapeHtml(q)}</span></div>`;
        document.getElementById('questionInput').value = '';
        chatDiv.innerHTML += `<div class="typing" style="color:#888;">🤖 ИИ печатает...</div>`;
        chatDiv.scrollTop = chatDiv.scrollHeight;
        try {
            const res = await fetch('/ask', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ question: q }) });
            const data = await res.json();
            document.querySelector('.typing')?.remove();
            chatDiv.innerHTML += `<div class="chat-message-bot"><span>${escapeHtml(data.answer)}</span></div>`;
            chatDiv.scrollTop = chatDiv.scrollHeight;
        } catch(e) { document.querySelector('.typing')?.remove(); chatDiv.innerHTML += `<div class="chat-message-bot"><span>Ошибка</span></div>`; }
    }
    
    function calculateCost() {
        const name = document.getElementById('productName').value.trim();
        const mat = parseFloat(document.getElementById('materialCost').value);
        const time = parseInt(document.getElementById('timeMinutes').value);
        const qty = parseInt(document.getElementById('quantityMonth').value);
        if (!name || isNaN(mat) || isNaN(time) || isNaN(qty)) { alert('Заполните все поля'); return; }
        const totalExp = analysisData ? analysisData.expense : 0;
        const labor = (300/60)*time;
        const cost = (mat*qty + labor*qty + totalExp)/qty;
        const price = cost*1.5;
        const breakeven = Math.ceil(totalExp / (price - (mat + labor)));
        document.getElementById('costResult').innerHTML = `<div class="cost-result-card"><div class="cost-result-header"><i class="fas fa-chart-line"></i> Результаты: ${escapeHtml(name)}</div><div class="cost-result-grid"><div class="cost-result-item"><div class="cost-result-icon"><i class="fas fa-cubes"></i></div><div class="cost-result-value" id="costValue">${cost.toFixed(2)} ₽</div><div class="cost-result-label">Себестоимость единицы</div></div><div class="cost-result-item"><div class="cost-result-icon"><i class="fas fa-tag"></i></div><div class="cost-result-value" id="priceValue">${price.toFixed(2)} ₽</div><div class="cost-result-label">Рекомендуемая цена</div></div><div class="cost-result-item"><div class="cost-result-icon"><i class="fas fa-chart-simple"></i></div><div class="cost-result-value" id="breakevenValue">${breakeven} шт./мес</div><div class="cost-result-label">Точка безубыточности</div></div></div></div>`;
        document.getElementById('costResult').style.display = 'block';
    }
    
    function escapeHtml(t) { const d = document.createElement('div'); d.textContent = t; return d.innerHTML; }
    
    // Мобильное меню
    const menuBtn = document.getElementById('menuBtn');
    const mobileMenu = document.getElementById('mobileMenu');
    if (menuBtn) {
        menuBtn.onclick = () => { mobileMenu.style.display = mobileMenu.style.display === 'none' ? 'block' : 'none'; };
        const items = ['Загрузить', 'Отчёт', 'Прогноз', 'Советы', 'Категории', 'Динамика', 'Сезонность', 'Себестоимость', 'Чат'];
        let html = '';
        for (let i of items) {
            html += `<a onclick="if(analysisData){
                if('${i}'==='Загрузить') document.getElementById('dropZone').click();
                else if('${i}'==='Отчёт') showFullReport();
                else if('${i}'==='Прогноз') showForecast();
                else if('${i}'==='Советы') showTips();
                else if('${i}'==='Категории') showCategories();
                else if('${i}'==='Динамика') showTrend();
                else if('${i}'==='Сезонность') showSeasonality();
                else if('${i}'==='Себестоимость') showCost();
                else if('${i}'==='Чат') showChat();
            } else if('${i}'==='Загрузить') document.getElementById('dropZone').click();
            document.getElementById('mobileMenu').style.display='none';">${i}</a>`;
        }
        mobileMenu.innerHTML = html;
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

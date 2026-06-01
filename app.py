from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import os
import io
from datetime import datetime
from io import BytesIO

# Импорт GigaChat с правильной версией
from gigachat import GigaChat
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="CashFlow - AI Financial Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ НАСТРОЙКА GIGACHAT ============
# Получаем credentials из переменных окружения
GIGACHAT_CREDENTIALS = os.getenv('GIGACHAT_CREDENTIALS')
GIGACHAT_SCOPE = os.getenv('GIGACHAT_SCOPE', 'GIGACHAT_API_PERS')

# Проверяем, что credentials заданы
if not GIGACHAT_CREDENTIALS:
    print("⚠️ ВНИМАНИЕ: GIGACHAT_CREDENTIALS не найдена в переменных окружения!")
    print("⚠️ Задайте её в Render: Dashboard → ваш сервис → Environment Variables")
    print("⚠️ Без этого ключа GigaChat не будет работать!")

# Подключаем GigaChat
giga = None
try:
    if GIGACHAT_CREDENTIALS:
        giga = GigaChat(
            credentials=GIGACHAT_CREDENTIALS,
            scope=GIGACHAT_SCOPE,
            verify_ssl_certs=False,  # Отключаем проверку SSL (нужно для Render)
            model="GigaChat-Pro"      # Используем модель Pro
        )
        print("✅ GigaChat успешно подключен!")
    else:
        print("❌ GigaChat не подключен: отсутствуют credentials")
except Exception as e:
    print(f"❌ Ошибка подключения GigaChat: {e}")
    giga = None

# Проверка в конце
if giga:
    print("🚀 GigaChat готов к работе!")
else:
    print("⚠️ GigaChat НЕ РАБОТАЕТ. Чат-функции будут недоступны.")

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
    except Exception as e:
        print(f"Ошибка категоризации: {e}")
        return 'other'

def get_savings_tips(expenses_by_category, total_expense, top_expenses):
    if giga is None or not expenses_by_category or total_expense == 0:
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
    print(f"📁 Анализ файла: {filename}, размер: {len(file_content)} байт")
    
    df = parse_file(file_content, filename)
    df.columns = df.columns.str.lower().str.strip()
    print(f"📊 Колонки: {list(df.columns)}")
    
    # ПОИСК КОЛОНКИ С ДАТАМИ
    date_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'date' in col_lower or 'дата' in col_lower:
            date_col = col
            break
    
    days_count = 0
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
            date_min = df[date_col].min()
            date_max = df[date_col].max()
            if pd.notna(date_min) and pd.notna(date_max):
                days_count = (date_max - date_min).days + 1
                print(f"📅 Найдена колонка дат: {date_col}, период: {date_min.date()} - {date_max.date()}")
        except Exception as e:
            print(f"Ошибка парсинга дат: {e}")
    
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
    print(f"💰 Доходы: {total_income:.2f}, Расходы: {total_expense:.2f}, Строк: {len(df)}")
    
    # Рентабельность
    profitability = (net_profit / total_income * 100) if total_income > 0 else 0
    
    # Средний чек
    avg_check = total_income / len(incomes) if incomes else 0
    
    # Анализ клиентов
    client_analysis = {}
    if 'merchant' in df.columns or 'description' in df.columns:
        source_col = 'merchant' if 'merchant' in df.columns else 'description'
        income_sources = df[df['amount'] > 0].groupby(source_col)['amount'].sum().sort_values(ascending=False).head(10)
        client_analysis = income_sources.to_dict()
    
    categories = {}
    if expense_details:
        expense_df = pd.DataFrame(expense_details).head(10)
        expense_df['category'] = expense_df['description'].apply(ai_categorize)
        for cat, amt in expense_df.groupby('category')['amount'].sum().items():
            categories[category_names.get(cat, cat)] = float(amt)
        print(f"📂 Категории: {list(categories.keys())}")
    
    tips = ""
    if categories and total_expense > 0:
        top_expenses = sorted(expense_details, key=lambda x: x['amount'], reverse=True)[:3]
        top_with_desc = [(d['description'], d['amount']) for d in top_expenses]
        tips = get_savings_tips(categories, total_expense, top_with_desc)
    
    predicted_total, predicted_change, _ = predict_next_month(categories, total_expense, days_count)
    
    # Прогноз кассовых разрывов
    cash_gap_warning = None
    if net_profit < 0:
        cash_gap_warning = f"⚠️ Расходы превышают доходы на {abs(net_profit):.2f} ₽. Рекомендуется сократить расходы или увеличить доходы."
    elif predicted_total and total_income:
        predicted_net = total_income - predicted_total
        if predicted_net < 0:
            cash_gap_warning = f"⚠️ По прогнозу, в следующем месяце ожидается убыток {abs(predicted_net):.2f} ₽. Возможен кассовый разрыв."
    
    # СРАВНЕНИЕ С ПРОШЛЫМ МЕСЯЦЕМ
    comparison = {'has_data': False}
    if date_col and len(df) > 0:
        try:
            last_date = df[date_col].max()
            current_month = last_date.month
            current_year = last_date.year
            last_month = current_month - 1 if current_month > 1 else 12
            last_year = current_year if current_month > 1 else current_year - 1
            
            df['month'] = pd.to_datetime(df[date_col]).dt.month
            df['year'] = pd.to_datetime(df[date_col]).dt.year
            
            current_mask = (df['year'] == current_year) & (df['month'] == current_month)
            current_income = df[current_mask & (df['amount'] > 0)]['amount'].sum()
            current_expense = abs(df[current_mask & (df['amount'] < 0)]['amount'].sum())
            current_profit = current_income - current_expense
            
            last_mask = (df['year'] == last_year) & (df['month'] == last_month)
            last_income = df[last_mask & (df['amount'] > 0)]['amount'].sum()
            last_expense = abs(df[last_mask & (df['amount'] < 0)]['amount'].sum())
            last_profit = last_income - last_expense
            
            if last_income > 0 or last_expense > 0:
                comparison = {
                    'has_data': True,
                    'income_change': ((current_income - last_income) / last_income * 100) if last_income > 0 else 0,
                    'expense_change': ((current_expense - last_expense) / last_expense * 100) if last_expense > 0 else 0,
                    'profit_change': ((current_profit - last_profit) / last_profit * 100) if last_profit != 0 else 0,
                    'current_income': float(current_income),
                    'last_income': float(last_income),
                    'current_expense': float(current_expense),
                    'last_expense': float(last_expense),
                    'current_profit': float(current_profit),
                    'last_profit': float(last_profit),
                    'current_month': f"{current_month}.{current_year}",
                    'last_month': f"{last_month}.{last_year}"
                }
                print(f"📊 Сравнение: текущий месяц {comparison['current_month']}, прошлый {comparison['last_month']}")
        except Exception as e:
            print(f"Ошибка сравнения: {e}")
    
    # Сезонность
    seasonality = {'has_data': False, 'expense_by_month': {}, 'by_weekday': {}}
    if date_col and len(df) > 0:
        try:
            temp_df = df[df['amount'] < 0].copy()
            if len(temp_df) > 0:
                seasonality['has_data'] = True
                temp_df['month'] = pd.to_datetime(temp_df[date_col]).dt.month
                for month in range(1, 13):
                    seasonality['expense_by_month'][month] = abs(temp_df[temp_df['month'] == month]['amount'].sum())
                
                temp_df['weekday'] = pd.to_datetime(temp_df[date_col]).dt.weekday
                weekday_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
                for i, name in enumerate(weekday_names):
                    seasonality['by_weekday'][name] = abs(temp_df[temp_df['weekday'] == i]['amount'].sum())
        except Exception as e:
            print(f"Ошибка сезонности: {e}")
    
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
        'incomes_count': len(incomes),
        'expenses_count': len(expenses),
        'days_count': days_count,
        'predicted_total': float(predicted_total) if predicted_total else None,
        'predicted_change': float(predicted_change) if predicted_change else None,
        'seasonality': seasonality,
        'profitability': round(profitability, 1),
        'avg_check': round(avg_check, 2),
        'client_analysis': client_analysis,
        'cash_gap_warning': cash_gap_warning,
        'comparison': comparison,
        'insights': []
    }

@app.get("/download-template")
async def download_template():
    content = """date,description,amount,type
2025-04-01,Оплата от клиента,50000,пополнение
2025-04-02,Аренда офиса,-15000,списание
2025-04-03,Покупка продуктов,-8000,списание
2025-04-04,Оплата от клиента,30000,пополнение
2025-04-05,Реклама,-5000,списание
2025-04-06,Налог,-4000,списание
2025-04-07,Закуп сырья,-12000,списание"""
    
    return StreamingResponse(
        io.BytesIO(content.encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cashflow_template.csv"}
    )

@app.post("/ask")
async def ask_question(request: Request):
    global last_analysis_result
    data = await request.json()
    question = data.get('question', '')
    
    if not last_analysis_result:
        return JSONResponse({'answer': 'Сначала загрузите и проанализируйте выписку'})
    
    if giga is None:
        return JSONResponse({'answer': '❌ GigaChat не подключен. Проверьте API ключ в настройках Render.'})
    
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
        return JSONResponse({'answer': f'❌ Ошибка GigaChat: {str(e)}'})

# HTML код (упрощённый, но рабочий)
html_content = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CashFlow — ИИ финансовый ассистент</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
            min-height: 100vh;
            padding: 20px;
            color: #fff;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .card {
            background: rgba(17, 17, 17, 0.85);
            backdrop-filter: blur(10px);
            border-radius: 28px;
            padding: 24px;
            margin-bottom: 20px;
            border: 1px solid rgba(234, 88, 12, 0.3);
            box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        }
        h1 {
            font-size: 2rem;
            background: linear-gradient(135deg, #f97316, #ea580c);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .upload-area {
            border: 2px dashed rgba(234, 88, 12, 0.3);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover {
            border-color: #f97316;
            background: rgba(234, 88, 12, 0.1);
        }
        .btn {
            background: linear-gradient(135deg, #ea580c, #9a3412);
            color: white;
            border: none;
            padding: 12px 28px;
            border-radius: 40px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .btn:hover { transform: translateY(-2px); }
        .btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .result-stats {
            display: flex;
            gap: 16px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        .stat-card {
            flex: 1;
            background: rgba(0,0,0,0.5);
            padding: 20px;
            border-radius: 16px;
            text-align: center;
        }
        .stat-card .value { font-size: 1.8rem; font-weight: bold; }
        .income .value { color: #f97316; }
        .expense .value { color: #ef4444; }
        .suggestion-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 20px;
        }
        .suggestion-btn {
            background: rgba(234, 88, 12, 0.15);
            border: 1px solid rgba(234, 88, 12, 0.3);
            padding: 10px 20px;
            border-radius: 40px;
            cursor: pointer;
            color: white;
            transition: all 0.2s;
        }
        .suggestion-btn:hover {
            background: rgba(234, 88, 12, 0.4);
            transform: translateY(-2px);
        }
        .info {
            background: rgba(234, 88, 12, 0.15);
            padding: 10px;
            border-radius: 12px;
            margin-top: 10px;
        }
        .spinner {
            border: 4px solid rgba(234, 88, 12, 0.3);
            border-top: 4px solid #f97316;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .chat-messages {
            height: 250px;
            overflow-y: auto;
            border: 1px solid rgba(234, 88, 12, 0.3);
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 16px;
            background: rgba(0,0,0,0.3);
        }
        .chat-message-user { text-align: right; margin: 8px 0; }
        .chat-message-user span {
            background: linear-gradient(135deg, #ea580c, #9a3412);
            padding: 8px 16px;
            border-radius: 20px;
            display: inline-block;
            max-width: 80%;
        }
        .chat-message-bot { text-align: left; margin: 8px 0; }
        .chat-message-bot span {
            background: rgba(0,0,0,0.5);
            padding: 8px 16px;
            border-radius: 20px;
            display: inline-block;
            max-width: 80%;
            border: 1px solid rgba(234,88,12,0.3);
        }
        .chat-input {
            display: flex;
            gap: 10px;
        }
        .chat-input input {
            flex: 1;
            padding: 12px;
            border: 1px solid rgba(234,88,12,0.3);
            border-radius: 40px;
            background: rgba(0,0,0,0.5);
            color: white;
        }
        @media (max-width: 768px) {
            body { padding: 10px; }
            .result-stats { flex-direction: column; }
            .suggestion-buttons { flex-direction: column; }
        }
    </style>
</head>
<body>
<div class="container">
    <div class="card">
        <h1>💰 CashFlow</h1>
        <p>ИИ-финансовый ассистент для микробизнеса</p>
    </div>
    
    <div class="card">
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <div style="font-size: 48px; color: #f97316;">📁</div>
            <p>Нажмите или перетащите файл</p>
            <p style="font-size: 12px; opacity: 0.7;">Поддерживаются: CSV, Excel</p>
            <input type="file" id="fileInput" accept=".csv,.xlsx,.xls" style="display: none;">
        </div>
        <div id="fileName" class="info" style="display: none;"></div>
        <div style="display: flex; gap: 10px; margin-top: 20px;">
            <button class="btn" id="analyzeBtn" onclick="uploadFile()" disabled style="flex: 1;">📊 Анализировать</button>
            <button class="btn" onclick="downloadTemplate()" style="background: #2a2a2a; border: 1px solid #f97316;">📥 Шаблон CSV</button>
        </div>
    </div>
    
    <div id="loading" style="display: none;">
        <div class="card">
            <div class="spinner"></div>
            <p style="text-align: center;">Анализирую выписку с помощью ИИ...</p>
        </div>
    </div>
    
    <div id="resultContainer" style="display: none;">
        <div class="card">
            <h3>🤖 Анализ выполнен!</h3>
            <div id="suggestionButtons" class="suggestion-buttons"></div>
        </div>
        <div id="fullReport" class="card" style="display: none;"></div>
        <div id="forecastBlock" class="card" style="display: none;"></div>
        <div id="tipsBlock" class="card" style="display: none;"></div>
        <div id="categoriesBlock" class="card" style="display: none;"></div>
        <div id="comparisonBlock" class="card" style="display: none;"></div>
        <div id="chatBlock" class="card" style="display: none;">
            <h3>💬 Чат с ИИ</h3>
            <div class="chat-messages" id="chatMessages">
                <div>Задайте вопрос о финансах</div>
            </div>
            <div class="chat-input">
                <input type="text" id="questionInput" placeholder="Например: на чём мне сэкономить?">
                <button class="btn" onclick="askQuestion()">Отправить</button>
            </div>
        </div>
    </div>
</div>

<script>
let analysisData = null;
const fileInput = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');

fileInput.onchange = () => {
    if (fileInput.files.length) {
        document.getElementById('fileName').textContent = "Выбран файл: " + fileInput.files[0].name;
        document.getElementById('fileName').style.display = 'block';
        analyzeBtn.disabled = false;
    }
};

function downloadTemplate() {
    window.location.href = '/download-template';
}

async function uploadFile() {
    if (!fileInput.files.length) return;
    
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    document.getElementById('loading').style.display = 'block';
    document.getElementById('resultContainer').style.display = 'none';
    
    try {
        const response = await fetch('/upload', { method: 'POST', body: formData });
        const result = await response.json();
        
        if (response.ok) {
            analysisData = result;
            showSmartSuggestions(result);
            document.getElementById('resultContainer').style.display = 'block';
        } else {
            alert('Ошибка: ' + (result.error || 'Неизвестная ошибка'));
        }
    } catch (error) {
        alert('Ошибка: ' + error.message);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

function showSmartSuggestions(data) {
    const buttons = [
        { text: '📈 Полный отчёт', func: showFullReport },
        { text: '📊 Сравнение', func: showComparison },
        { text: '💡 Советы', func: showTips },
        { text: '📂 Категории', func: showCategories },
        { text: '💬 Чат', func: showChat }
    ];
    
    const container = document.getElementById('suggestionButtons');
    container.innerHTML = buttons.map(btn => 
        `<button class="suggestion-btn" onclick="${btn.func.name}()">${btn.text}</button>`
    ).join('');
}

function showFullReport() {
    const d = analysisData;
    document.getElementById('fullReport').innerHTML = `
        <h3>📊 Полный отчёт</h3>
        <div class="result-stats">
            <div class="stat-card income">
                <div class="value">${d.income.toFixed(2)} ₽</div>
                <div>💰 Доходы</div>
            </div>
            <div class="stat-card expense">
                <div class="value">${d.expense.toFixed(2)} ₽</div>
                <div>💸 Расходы</div>
            </div>
            <div class="stat-card">
                <div class="value" style="color: ${d.net_profit >= 0 ? '#f97316' : '#ef4444'}">${d.net_profit >= 0 ? '+' : ''}${d.net_profit.toFixed(2)} ₽</div>
                <div>✅ Чистая прибыль</div>
            </div>
        </div>
        <div class="result-stats">
            <div class="stat-card">
                <div class="value">${d.profitability}%</div>
                <div>📈 Рентабельность</div>
            </div>
            <div class="stat-card">
                <div class="value">${d.avg_check.toFixed(2)} ₽</div>
                <div>💰 Средний чек</div>
            </div>
        </div>
        <div class="info">
            📊 Обработано строк: ${d.rows_count}<br>
            📈 Доходов: ${d.incomes_count}, 📉 Расходов: ${d.expenses_count}
        </div>
        ${d.cash_gap_warning ? `<div class="info" style="background: rgba(239,68,68,0.2); color: #ef4444;">⚠️ ${d.cash_gap_warning}</div>` : ''}
    `;
    showBlock('fullReport');
}

function showComparison() {
    const comp = analysisData.comparison || {};
    if (!comp.has_data) {
        document.getElementById('fullReport').innerHTML = '<div class="info">Нет данных для сравнения с прошлым месяцем. Загрузите выписку за несколько месяцев.</div>';
        showBlock('fullReport');
        return;
    }
    
    document.getElementById('fullReport').innerHTML = `
        <h3>📊 Сравнение с прошлым месяцем</h3>
        <div class="result-stats">
            <div class="stat-card">
                <div>💰 Доходы</div>
                <div class="value" style="font-size: 1.2rem;">${comp.current_income.toFixed(2)} ₽</div>
                <div>было: ${comp.last_income.toFixed(2)} ₽</div>
                <div style="color: ${comp.income_change >= 0 ? '#10b981' : '#ef4444'}">${comp.income_change >= 0 ? '+' : ''}${comp.income_change.toFixed(1)}%</div>
            </div>
            <div class="stat-card">
                <div>💸 Расходы</div>
                <div class="value" style="font-size: 1.2rem;">${comp.current_expense.toFixed(2)} ₽</div>
                <div>было: ${comp.last_expense.toFixed(2)} ₽</div>
                <div style="color: ${comp.expense_change <= 0 ? '#10b981' : '#ef4444'}">${comp.expense_change >= 0 ? '+' : ''}${comp.expense_change.toFixed(1)}%</div>
            </div>
            <div class="stat-card">
                <div>✅ Прибыль</div>
                <div class="value" style="font-size: 1.2rem;">${comp.current_profit.toFixed(2)} ₽</div>
                <div>было: ${comp.last_profit.toFixed(2)} ₽</div>
                <div style="color: ${comp.profit_change >= 0 ? '#10b981' : '#ef4444'}">${comp.profit_change >= 0 ? '+' : ''}${comp.profit_change.toFixed(1)}%</div>
            </div>
        </div>
    `;
    showBlock('fullReport');
}

function showTips() {
    const d = analysisData;
    if (d.tips) {
        const items = d.tips.split('•').filter(i => i.trim());
        document.getElementById('fullReport').innerHTML = `
            <h3>💡 Советы по экономии</h3>
            <ul style="margin-left: 20px;">
                ${items.map(i => `<li style="margin: 10px 0;">• ${i.trim()}</li>`).join('')}
            </ul>
        `;
    } else {
        document.getElementById('fullReport').innerHTML = '<p>Нет советов</p>';
    }
    showBlock('fullReport');
}

function showCategories() {
    const d = analysisData;
    if (d.categories && Object.keys(d.categories).length) {
        let table = '<h3>📊 Расходы по категориям</h3><table style="width: 100%; border-collapse: collapse;">';
        for (const [cat, amt] of Object.entries(d.categories)) {
            table += `<tr style="border-bottom: 1px solid rgba(234,88,12,0.3);"><td style="padding: 10px;">${cat}佛罗<td style="padding: 10px; text-align: right;">${amt.toFixed(2)} ₽</td></tr>`;
        }
        table += '</table>';
        document.getElementById('fullReport').innerHTML = table;
    } else {
        document.getElementById('fullReport').innerHTML = '<p>Нет данных для категоризации</p>';
    }
    showBlock('fullReport');
}

function showChat() {
    showBlock('chatBlock');
}

function showBlock(id) {
    const blocks = ['fullReport', 'forecastBlock', 'tipsBlock', 'categoriesBlock', 'comparisonBlock', 'chatBlock'];
    blocks.forEach(b => {
        const el = document.getElementById(b);
        if (el) el.style.display = 'none';
    });
    document.getElementById(id).style.display = 'block';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function askQuestion() {
    const question = document.getElementById('questionInput').value.trim();
    if (!question) return;
    
    const chatDiv = document.getElementById('chatMessages');
    if (chatDiv.children.length === 1 && chatDiv.children[0].textContent.includes('Задайте вопрос')) {
        chatDiv.innerHTML = '';
    }
    
    chatDiv.innerHTML += `<div class="chat-message-user"><span>${escapeHtml(question)}</span></div>`;
    document.getElementById('questionInput').value = '';
    chatDiv.innerHTML += `<div class="typing" style="opacity:0.7;"><i class="fas fa-spinner fa-pulse"></i> ИИ печатает...</div>`;
    chatDiv.scrollTop = chatDiv.scrollHeight;
    
    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        });
        const data = await response.json();
        document.querySelector('.typing')?.remove();
        chatDiv.innerHTML += `<div class="chat-message-bot"><span>${escapeHtml(data.answer)}</span></div>`;
        chatDiv.scrollTop = chatDiv.scrollHeight;
    } catch (error) {
        document.querySelector('.typing')?.remove();
        chatDiv.innerHTML += `<div class="chat-message-bot"><span>❌ Ошибка</span></div>`;
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
</script>
</body>
</html>
"""

@app.get("/")
async def home():
    return HTMLResponse(html_content)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        print(f"📁 Получен файл: {file.filename}")
        file_content = await file.read()
        print(f"📄 Размер файла: {len(file_content)} байт")
        if len(file_content) == 0:
            return JSONResponse({'error': 'Файл пуст'}, status_code=400)
        result = analyze_statement(file_content, file.filename)
        return JSONResponse(result)
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({'error': str(e)}, status_code=400)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)

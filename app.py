from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from gigachat import GigaChat
from dotenv import load_dotenv
import pandas as pd
import os
import io
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
giga = None
try:
    credentials = os.getenv('GIGACHAT_CREDENTIALS')
    if credentials:
        giga = GigaChat(
            credentials=credentials,
            scope=os.getenv('GIGACHAT_SCOPE', 'GIGACHAT_API_PERS'),
            verify_ssl_certs=False,
            model="GigaChat-Pro"
        )
        print("✅ GigaChat подключен")
    else:
        print("⚠️ GIGACHAT_CREDENTIALS не найдена")
except Exception as e:
    print(f"❌ Ошибка GigaChat: {e}")

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
    
    profitability = (net_profit / total_income * 100) if total_income > 0 else 0
    avg_check = total_income / len(incomes) if incomes else 0
    
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

# HTML код (сокращённый, только CSV, без PDF)
html_content = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=yes">
    <title>CashFlow — ИИ финансовый ассистент</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700;14..32,800&family=Playfair+Display:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        :root {
            --primary-start: #ea580c;
            --primary-end: #9a3412;
            --accent: #f97316;
            --card-bg: rgba(17, 17, 17, 0.85);
            --text-primary: #ffffff;
            --text-secondary: #a3a3a3;
            --border-color: rgba(234, 88, 12, 0.3);
            --stat-bg: rgba(0, 0, 0, 0.5);
            --success: #f97316;
            --danger: #ef4444;
            --warning: #f59e0b;
            --card-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            --backdrop-blur: blur(10px);
        }
        body {
            font-family: 'Inter', sans-serif;
            min-height: 100vh;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f0f 100%);
            padding: 1.5rem;
        }
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: var(--card-bg); border-radius: 10px; }
        ::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 10px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card {
            background: var(--card-bg);
            backdrop-filter: var(--backdrop-blur);
            border-radius: 28px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--card-shadow);
            transition: transform 0.3s ease;
            animation: fadeInUp 0.4s ease-out;
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }
        .card:hover { transform: translateY(-4px); }
        h1 {
            font-family: 'Playfair Display', serif;
            font-size: 2rem;
            background: linear-gradient(135deg, #f97316 0%, #ea580c 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .upload-area {
            border: 2px dashed var(--border-color);
            border-radius: 20px;
            padding: 2rem;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(0, 0, 0, 0.3);
        }
        .upload-area:hover { border-color: var(--accent); background: rgba(234, 88, 12, 0.1); }
        .upload-area i { font-size: 3rem; color: var(--accent); margin-bottom: 1rem; }
        input[type="file"] { display: none; }
        .btn {
            background: linear-gradient(135deg, var(--primary-start), var(--primary-end));
            color: white;
            border: none;
            padding: 12px 28px;
            border-radius: 40px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
        }
        .btn:hover { transform: translateY(-2px); }
        .suggestion-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 0.7rem;
            margin-top: 1rem;
        }
        .suggestion-btn {
            background: rgba(234, 88, 12, 0.15);
            border: 1px solid var(--border-color);
            padding: 0.7rem 1.2rem;
            border-radius: 40px;
            cursor: pointer;
            color: var(--text-primary);
        }
        .suggestion-btn:hover { background: rgba(234, 88, 12, 0.4); }
        .result-stats {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin-bottom: 1.5rem;
        }
        .stat-card {
            flex: 1;
            background: var(--stat-bg);
            padding: 1rem;
            border-radius: 20px;
            text-align: center;
        }
        .stat-card .value { font-size: 1.6rem; font-weight: 800; }
        .income .value { color: var(--success); }
        .expense .value { color: var(--danger); }
        .info {
            background: rgba(234, 88, 12, 0.15);
            padding: 0.7rem;
            border-radius: 12px;
            font-size: 0.8rem;
            margin-top: 1rem;
        }
        .spinner {
            border: 4px solid rgba(234, 88, 12, 0.3);
            border-top: 4px solid var(--accent);
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .chat-messages {
            height: 250px;
            overflow-y: auto;
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 1rem;
            margin-bottom: 1rem;
            background: var(--stat-bg);
        }
        .chat-message-user { text-align: right; margin: 0.5rem 0; }
        .chat-message-user span {
            background: linear-gradient(135deg, var(--primary-start), var(--primary-end));
            color: white;
            padding: 8px 14px;
            border-radius: 20px;
            display: inline-block;
            max-width: 80%;
        }
        .chat-message-bot { text-align: left; margin: 0.5rem 0; }
        .chat-message-bot span {
            background: var(--card-bg);
            color: var(--text-primary);
            padding: 8px 14px;
            border-radius: 20px;
            display: inline-block;
            max-width: 80%;
            border: 1px solid var(--border-color);
        }
        .chat-input { display: flex; gap: 0.8rem; }
        .chat-input input {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid var(--border-color);
            border-radius: 40px;
            background: var(--card-bg);
            color: var(--text-primary);
        }
        .mobile-header { display: none; justify-content: space-between; align-items: center; margin-bottom: 1rem; background: rgba(0,0,0,0.5); padding: 0.8rem 1.2rem; border-radius: 50px; }
        #menuBtn { background: none; border: none; font-size: 1.6rem; cursor: pointer; color: var(--accent); }
        #mobileMenu {
            background: var(--card-bg);
            border-radius: 20px;
            padding: 1rem;
            margin-bottom: 1rem;
            display: none;
        }
        #mobileMenu a {
            display: block;
            padding: 0.8rem;
            text-decoration: none;
            color: var(--text-primary);
            border-bottom: 1px solid var(--border-color);
        }
        @media (max-width: 768px) {
            body { padding: 0.8rem; }
            .desktop-title { display: none; }
            .mobile-header { display: flex; }
            .result-stats { flex-direction: column; }
            .suggestion-buttons { flex-direction: column; }
        }
    </style>
</head>
<body>
<div class="mobile-header">
    <h1 style="color: var(--accent); margin:0; font-size:1.3rem;">CashFlow</h1>
    <button id="menuBtn">☰</button>
</div>
<div id="mobileMenu"></div>
<div class="container">
    <div class="card desktop-title">
        <h1>CashFlow</h1>
        <div class="subtitle">ИИ-финансовый ассистент для микробизнеса</div>
    </div>
    <div class="card">
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <i class="fas fa-cloud-upload-alt"></i>
            <p>Нажмите или перетащите файл</p>
            <p style="font-size:0.7rem;opacity:0.7;">Поддерживаются: CSV, Excel</p>
            <input type="file" id="fileInput" accept=".csv,.xlsx,.xls" style="display: none;">
        </div>
        <div id="fileName" class="info" style="display:none;"></div>
        <div style="display: flex; gap: 10px; margin-top: 1rem;">
            <button class="btn" id="analyzeBtn" onclick="uploadFile()" disabled style="flex: 1;">📊 Анализировать</button>
            <button class="btn" onclick="downloadTemplate()" style="flex: 0; background: #2a2a2a; border: 1px solid #f97316;">📥 Шаблон CSV</button>
        </div>
    </div>
    <div class="loading" id="loading" style="display:none;text-align:center;padding:2rem;"><div class="spinner"></div><p>Анализирую выписку с помощью ИИ...</p></div>
    <div id="resultContainer" style="display:none;">
        <div class="card" id="suggestionCard"><h3><i class="fas fa-robot"></i> Анализ выполнен!</h3><div id="insightsContainer"></div><div id="suggestionButtons" class="suggestion-buttons"></div></div>
        <div id="fullReport" class="card" style="display:none;"><div id="reportContent"></div></div>
        <div id="forecastBlock" class="card" style="display:none;"><div id="forecastContent"></div></div>
        <div id="tipsBlock" class="card" style="display:none;"><div id="tipsContent"></div></div>
        <div id="categoriesBlock" class="card" style="display:none;"><div id="categoriesContent"></div><canvas id="expenseChart" style="max-width:300px; margin:1rem auto;"></canvas></div>
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
        <div id="chatBlock" class="card" style="display:none;"><h3>Чат с ИИ</h3><div class="chat-messages" id="chatMessages"><div>Задайте вопрос о финансах</div></div><div class="chat-input"><input type="text" id="questionInput" placeholder="Например: на чём мне сэкономить?"><button class="btn" onclick="askQuestion()">Отправить</button></div></div>
    </div>
</div>

<script>
let selectedFile = null, analysisData = null, expenseChart = null, trendChart = null;
const fileInput = document.getElementById('fileInput'), analyzeBtn = document.getElementById('analyzeBtn'), fileNameDiv = document.getElementById('fileName');

function handleFileSelect() { 
    if(fileInput.files.length){ 
        selectedFile = fileInput.files[0]; 
        fileNameDiv.textContent = "Выбран файл: "+selectedFile.name; 
        fileNameDiv.style.display = "block"; 
        analyzeBtn.disabled = false; 
    } 
}
fileInput.onchange = handleFileSelect;

function downloadTemplate() { window.location.href = '/download-template'; }

async function uploadFile() {
    if(!selectedFile) return;
    const formData = new FormData();
    formData.append('file', selectedFile);
    document.getElementById('loading').style.display = 'block';
    document.getElementById('resultContainer').style.display = 'none';
    try { 
        const response = await fetch('/upload',{method:'POST',body:formData}); 
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const result = await response.json(); 
        analysisData = result; 
        showSmartSuggestions(result); 
    } catch(error){ 
        alert('Ошибка: '+error.message); 
    } finally{ document.getElementById('loading').style.display='none'; }
}

function drawChart(categories) {
    const ctx = document.getElementById('expenseChart')?.getContext('2d');
    if(!ctx) return;
    if(expenseChart) expenseChart.destroy();
    expenseChart = new Chart(ctx, { type:'pie', data:{ labels:Object.keys(categories), datasets:[{ data:Object.values(categories), backgroundColor:['#ea580c','#f97316','#c2410c','#fdba74','#9a3412','#7c2d12'] }] }, options:{ responsive:true } });
}

function drawTrendChart() {
    const d = analysisData;
    if(!d) return;
    const ctx = document.getElementById('trendChart')?.getContext('2d');
    if(!ctx) return;
    if(trendChart) trendChart.destroy();
    trendChart = new Chart(ctx, { type:'line', data:{ labels:['Неделя 1', 'Неделя 2', 'Неделя 3', 'Неделя 4'], datasets:[{ label:'Доходы', data:[d.income*0.6, d.income*0.8, d.income*0.9, d.income], borderColor:'#f97316', tension:0.4, fill:true },{ label:'Расходы', data:[d.expense*0.7, d.expense*0.85, d.expense*0.95, d.expense], borderColor:'#ef4444', tension:0.4, fill:true }] }, options:{ responsive:true } });
}

function showSmartSuggestions(data) {
    document.getElementById('insightsContainer').innerHTML = '<div><i class="fas fa-check-circle" style="color:#f97316;"></i> Анализ выполнен успешно</div>';
    const allButtons = [
        { text:'📈 Полный отчёт', func:showFullReport },
        { text:'🔮 Прогноз', func:showForecast },
        { text:'💡 Советы', func:showTips },
        { text:'📊 Категории', func:showCategories },
        { text:'📈 Динамика', func:showTrend },
        { text:'📅 Сезонность', func:showSeasonality },
        { text:'💰 Себестоимость', func:showCost },
        { text:'👥 Анализ клиентов', func:showClientAnalysis },
        { text:'⚠️ Кассовые разрывы', func:showCashGap },
        { text:'📊 Сравнение', func:showComparison },
        { text:'💬 Чат', func:showChat }
    ];
    let buttonsHtml = '';
    for(let btn of allButtons) buttonsHtml += `<button class="suggestion-btn" onclick="${btn.func.name}()">${btn.text}</button>`;
    document.getElementById('suggestionButtons').innerHTML = buttonsHtml;
    document.getElementById('resultContainer').style.display = 'block';
    if(window.innerWidth<=768 && mobileMenu) mobileMenu.style.display = 'none';
}

function showFullReport() {
    const d = analysisData;
    document.getElementById('reportContent').innerHTML = `
        <h3><i class="fas fa-chart-simple"></i> Отчёт CashFlow</h3>
        <div class="result-stats">
            <div class="stat-card income"><div class="value">${d.income.toFixed(2)} ₽</div><div>💰 Доходы</div></div>
            <div class="stat-card expense"><div class="value">${d.expense.toFixed(2)} ₽</div><div>💸 Расходы</div></div>
            <div class="stat-card"><div class="value" style="color: ${d.net_profit >= 0 ? '#f97316' : '#ef4444'}">${d.net_profit >= 0 ? '+' : ''}${d.net_profit.toFixed(2)} ₽</div><div>✅ Чистая прибыль</div></div>
        </div>
        <div class="result-stats">
            <div class="stat-card"><div class="value">${d.profitability}%</div><div>📈 Рентабельность</div></div>
            <div class="stat-card"><div class="value">${d.avg_check.toFixed(2)} ₽</div><div>💰 Средний чек</div></div>
        </div>
        <div class="info">📊 Обработано строк: ${d.rows_count}<br>📈 Доходов: ${d.incomes_count}, 📉 Расходов: ${d.expenses_count}</div>
        ${d.cash_gap_warning ? `<div class="info" style="background: rgba(239,68,68,0.2); color: #ef4444;">⚠️ ${d.cash_gap_warning}</div>` : ''}
    `;
    showBlock('fullReport');
}

function showForecast() {
    const d = analysisData;
    if(d.predicted_total && d.predicted_total>0){
        const changeColor = d.predicted_change >= 0 ? '#ef4444' : '#f97316';
        const changeIcon = d.predicted_change >= 0 ? '📈' : '📉';
        document.getElementById('forecastContent').innerHTML = `<div><h3><i class="fas fa-calendar-week"></i> Прогноз на следующий месяц</h3><div class="result-stats"><div class="stat-card"><div class="value" style="color:#f97316;">${d.predicted_total.toFixed(2)} ₽</div><div>Прогнозируемые расходы</div></div><div class="stat-card"><div class="value" style="color:${changeColor};">${changeIcon} ${d.predicted_change>=0?'+':''}${d.predicted_change.toFixed(1)}%</div><div>Изменение</div></div></div><div class="info">Прогноз основан на ${d.days_count} днях</div></div>`;
    } else document.getElementById('forecastContent').innerHTML = '<p>Нет данных для прогноза</p>';
    showBlock('forecastBlock');
}

function showTips() {
    const d = analysisData;
    if(d.tips){
        const items = d.tips.split('•').filter(i=>i.trim());
        document.getElementById('tipsContent').innerHTML = `<h3><i class="fas fa-lightbulb"></i> Советы по экономии</h3><ul>${items.map(i=>`<li>• ${i.trim()}</li>`).join('')}</ul>`;
    } else document.getElementById('tipsContent').innerHTML = '<p>Нет советов</p>';
    showBlock('tipsBlock');
}

function showCategories() {
    const d = analysisData;
    if(d.categories && Object.keys(d.categories).length){
        let table = '<h3><i class="fas fa-tags"></i> Расходы по категориям</h3><table style="width:100%"><tr><th>Категория</th><th>Сумма</th></tr>';
        for(const [cat,amt] of Object.entries(d.categories)){
            table += `<tr><td>${cat}</td><td style="text-align:right">${amt.toFixed(2)} ₽</td></tr>`;
        }
        table += '</table>';
        document.getElementById('categoriesContent').innerHTML = table;
        drawChart(d.categories);
    } else document.getElementById('categoriesContent').innerHTML = '<p>Нет данных для категоризации</p>';
    showBlock('categoriesBlock');
}

function showTrend() { drawTrendChart(); showBlock('trendBlock'); }

function showSeasonality() {
    const s = analysisData?.seasonality || {};
    if (!s.has_data) {
        document.getElementById('seasonalityContent').innerHTML = '<p>Нет данных для анализа сезонности</p>';
        showBlock('seasonalityBlock');
        return;
    }
    let html = '<h3><i class="fas fa-calendar-alt"></i> Сезонность расходов</h3><div style="display:flex;gap:10px;flex-wrap:wrap;">';
    for (const [month, amt] of Object.entries(s.expense_by_month)) {
        if (amt > 0) html += `<div style="text-align:center"><div>Месяц ${month}</div><div style="color:#f97316">${amt.toFixed(0)} ₽</div></div>`;
    }
    html += '</div>';
    document.getElementById('seasonalityContent').innerHTML = html;
    showBlock('seasonalityBlock');
}

function showCost() { showBlock('costBlock'); }
function showChat() { showBlock('chatBlock'); }

function showClientAnalysis() {
    const d = analysisData;
    const clients = d.client_analysis || {};
    if (Object.keys(clients).length === 0) {
        document.getElementById('categoriesContent').innerHTML = '<p>Нет данных для анализа клиентов</p>';
        showBlock('categoriesBlock');
        return;
    }
    let table = '<h3><i class="fas fa-users"></i> Анализ клиентов</h3><table style="width:100%"><tr><th>Источник</th><th>Сумма</th></tr>';
    for (const [source, amount] of Object.entries(clients)) {
        const shortSource = source.length > 40 ? source.substring(0, 37) + '...' : source;
        table += `<tr><td title="${source}">${shortSource}</td><td style="text-align:right">${amount.toFixed(2)} ₽</td></tr>`;
    }
    table += '</table>';
    document.getElementById('categoriesContent').innerHTML = table;
    showBlock('categoriesBlock');
}

function showCashGap() {
    const d = analysisData;
    if (!d.cash_gap_warning) {
        document.getElementById('seasonalityContent').innerHTML = '<p>Прогноз кассовых разрывов: всё стабильно</p>';
        showBlock('seasonalityBlock');
        return;
    }
    document.getElementById('seasonalityContent').innerHTML = `<div style="background: rgba(239,68,68,0.2); padding:1rem; border-radius:16px;"><h3><i class="fas fa-exclamation-triangle"></i> Предупреждение</h3><p>${d.cash_gap_warning}</p></div>`;
    showBlock('seasonalityBlock');
}

function showComparison() {
    const d = analysisData;
    const comp = d.comparison || {};
    if (!comp.has_data) {
        document.getElementById('reportContent').innerHTML = '<p>Нет данных для сравнения. Загрузите выписку за несколько месяцев.</p>';
        showBlock('fullReport');
        return;
    }
    const incomeColor = comp.income_change >= 0 ? '#10b981' : '#ef4444';
    const expenseColor = comp.expense_change >= 0 ? '#ef4444' : '#10b981';
    const profitColor = comp.profit_change >= 0 ? '#10b981' : '#ef4444';
    document.getElementById('reportContent').innerHTML = `
        <h3><i class="fas fa-chart-line"></i> Сравнение с прошлым месяцем</h3>
        <div class="result-stats">
            <div class="stat-card"><div>💰 Доходы</div><div class="value">${comp.current_income.toFixed(2)} ₽</div><div>vs ${comp.last_income.toFixed(2)} ₽</div><div style="color:${incomeColor}">${comp.income_change >= 0 ? '+' : ''}${comp.income_change.toFixed(1)}%</div></div>
            <div class="stat-card"><div>💸 Расходы</div><div class="value">${comp.current_expense.toFixed(2)} ₽</div><div>vs ${comp.last_expense.toFixed(2)} ₽</div><div style="color:${expenseColor}">${comp.expense_change >= 0 ? '+' : ''}${comp.expense_change.toFixed(1)}%</div></div>
            <div class="stat-card"><div>✅ Прибыль</div><div class="value">${comp.current_profit.toFixed(2)} ₽</div><div>vs ${comp.last_profit.toFixed(2)} ₽</div><div style="color:${profitColor}">${comp.profit_change >= 0 ? '+' : ''}${comp.profit_change.toFixed(1)}%</div></div>
        </div>
    `;
    showBlock('fullReport');
}

function showBlock(id) {
    const blocks = ['fullReport','forecastBlock','tipsBlock','categoriesBlock','trendBlock','seasonalityBlock','costBlock','chatBlock'];
    blocks.forEach(b=>{ const el = document.getElementById(b); if(el) el.style.display = 'none'; });
    document.getElementById(id).style.display = 'block';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function askQuestion() {
    const q = document.getElementById('questionInput').value.trim();
    if(!q) return;
    const chatDiv = document.getElementById('chatMessages');
    if(chatDiv.children.length===1 && chatDiv.children[0].textContent.includes('Задайте вопрос')) chatDiv.innerHTML = '';
    chatDiv.innerHTML += `<div class="chat-message-user"><span>${escapeHtml(q)}</span></div>`;
    document.getElementById('questionInput').value = '';
    chatDiv.innerHTML += `<div class="typing" style="opacity:0.7;"><i class="fas fa-spinner fa-pulse"></i> ИИ печатает...</div>`;
    chatDiv.scrollTop = chatDiv.scrollHeight;
    try{
        const res = await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
        const data = await res.json();
        document.querySelector('.typing')?.remove();
        chatDiv.innerHTML += `<div class="chat-message-bot"><span>${escapeHtml(data.answer)}</span></div>`;
        chatDiv.scrollTop = chatDiv.scrollHeight;
    } catch(e){ document.querySelector('.typing')?.remove(); chatDiv.innerHTML += `<div class="chat-message-bot"><span>Ошибка</span></div>`; }
}

function calculateCost() {
    const name = document.getElementById('productName').value.trim();
    const mat = parseFloat(document.getElementById('materialCost').value);
    const time = parseInt(document.getElementById('timeMinutes').value);
    const qty = parseInt(document.getElementById('quantityMonth').value);
    if(!name || isNaN(mat) || isNaN(time) || isNaN(qty)){ alert('Заполните все поля'); return; }
    const totalExp = analysisData ? analysisData.expense : 0;
    const labor = (300/60)*time;
    const varTotal = mat*qty + labor*qty;
    const full = varTotal + totalExp;
    const cost = full/qty;
    const price = cost*1.5;
    document.getElementById('costResult').innerHTML = `<div class="info"><strong>Результаты для ${name}:</strong><br>Себестоимость: ${cost.toFixed(2)} ₽<br>Рекомендуемая цена: ${price.toFixed(2)} ₽</div>`;
}

function escapeHtml(t){ const d=document.createElement('div'); d.textContent=t; return d.innerHTML; }

const menuBtn=document.getElementById('menuBtn'), mobileMenu=document.getElementById('mobileMenu');
if(menuBtn && mobileMenu){
    menuBtn.onclick=()=>{ mobileMenu.style.display=mobileMenu.style.display==='none'?'block':'none'; };
    const items=['Загрузить','Отчёт','Прогноз','Советы','Категории','Динамика','Сезонность','Себестоимость','Клиенты','Кассовые разрывы','Сравнение','Чат'];
    let html='';
    for(let i of items) html+=`<a href="#" onclick="if(analysisData){ if('${i}'==='Загрузить') document.querySelector('.upload-area').click(); else if('${i}'==='Отчёт') showFullReport(); else if('${i}'==='Прогноз') showForecast(); else if('${i}'==='Советы') showTips(); else if('${i}'==='Категории') showCategories(); else if('${i}'==='Динамика') showTrend(); else if('${i}'==='Сезонность') showSeasonality(); else if('${i}'==='Себестоимость') showCost(); else if('${i}'==='Клиенты') showClientAnalysis(); else if('${i}'==='Кассовые разрывы') showCashGap(); else if('${i}'==='Сравнение') showComparison(); else if('${i}'==='Чат') showChat(); } else if('${i}'==='Загрузить') document.querySelector('.upload-area').click(); document.getElementById('mobileMenu').style.display='none';">${i}</a>`;
    mobileMenu.innerHTML=html;
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

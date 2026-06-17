from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import openai
import pandas as pd
import os
import calendar
import io
import json
import sqlite3
from datetime import datetime
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import requests

load_dotenv()

app = FastAPI(title="CashFlow - AI Financial Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ БАЗА ДАННЫХ ============
def init_db():
    conn = sqlite3.connect('cashflow_history.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            filename TEXT NOT NULL,
            income REAL,
            expense REAL,
            net_profit REAL,
            categories TEXT,
            months_data TEXT,
            forecast TEXT,
            full_data TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS budgets (
            category TEXT,
            limit_amount REAL,
            month TEXT,
            PRIMARY KEY (category, month)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def save_analysis_to_db(filename, result):
    conn = sqlite3.connect('cashflow_history.db')
    cursor = conn.cursor()
    
    # Преобразуем все numpy типы в обычные Python
    def convert_to_python(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python(v) for v in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        else:
            return obj
    
    clean_result = convert_to_python(result)
    
    cursor.execute('''
        INSERT INTO analyses (date, filename, income, expense, net_profit, categories, months_data, forecast, full_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        filename,
        float(result.get('income', 0)),
        float(result.get('expense', 0)),
        float(result.get('net_profit', 0)),
        json.dumps(convert_to_python(result.get('categories', {}))),
        json.dumps(convert_to_python(result.get('months_data', {}))),
        json.dumps(convert_to_python(result.get('forecast_3months', []))),
        json.dumps(clean_result)
    ))
    conn.commit()
    conn.close()

def get_history_from_db(limit=10):
    conn = sqlite3.connect('cashflow_history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, date, filename, income, expense, net_profit FROM analyses ORDER BY date DESC LIMIT ?', (limit,))
    rows = cursor.fetchall()
    conn.close()
    return [{'id': row[0], 'date': row[1], 'filename': row[2], 'income': row[3], 'expense': row[4], 'net_profit': row[5]} for row in rows]

def get_analysis_by_id(analysis_id):
    conn = sqlite3.connect('cashflow_history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT full_data FROM analyses WHERE id = ?', (analysis_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None

def get_budget_limits(month=None):
    if month is None:
        month = datetime.now().strftime('%Y-%m')
    conn = sqlite3.connect('cashflow_history.db')
    cursor = conn.cursor()
    cursor.execute('SELECT category, limit_amount FROM budgets WHERE month = ?', (month,))
    rows = cursor.fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}

def set_budget_limit(category, limit_amount, month=None):
    if month is None:
        month = datetime.now().strftime('%Y-%m')
    conn = sqlite3.connect('cashflow_history.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM budgets WHERE category = ? AND month = ?', (category, month))
    cursor.execute('INSERT INTO budgets (category, limit_amount, month) VALUES (?, ?, ?)', (category, limit_amount, month))
    conn.commit()
    conn.close()

def check_budget_alerts(expenses_by_category):
    current_month = datetime.now().strftime('%Y-%m')
    budgets = get_budget_limits(current_month)
    alerts = []
    for category, spent in expenses_by_category.items():
        if category in budgets:
            limit = budgets[category]
            if spent > limit:
                percent = (spent - limit) / limit * 100
                alerts.append({
                    'category': category,
                    'spent': spent,
                    'limit': limit,
                    'percent': round(percent, 1),
                    'message': f"⚠️ Превышен лимит по категории '{category}': {spent:.2f} ₽ / {limit:.2f} ₽ (+{percent:.1f}%)"
                })
            elif spent > limit * 0.8:
                alerts.append({
                    'category': category,
                    'spent': spent,
                    'limit': limit,
                    'percent': round((spent / limit) * 100, 1),
                    'message': f"⚠️ Лимит по категории '{category}' почти исчерпан: {spent:.2f} ₽ / {limit:.2f} ₽ ({round(spent/limit*100)}%)"
                })
    return alerts
    # ============ YANDEXGPT ============
YANDEX_CLOUD_FOLDER = "b1g41a3v1qrmkt4rccos"
YANDEX_CLOUD_API_KEY = os.getenv('YANDEX_API_KEY')

def ask_yandex(prompt: str) -> str:
    """Запрос к YandexGPT"""
    if not YANDEX_CLOUD_API_KEY:
        return None
    
    try:
        client = openai.OpenAI(
            api_key=YANDEX_CLOUD_API_KEY,
            base_url="https://ai.api.cloud.yandex.net/v1",
            project=YANDEX_CLOUD_FOLDER
        )
        
        response = client.responses.create(
            model=f"gpt://{YANDEX_CLOUD_FOLDER}/yandexgpt-5.1/latest",
            temperature=0.7,
            instructions="Ты финансовый ассистент для микробизнеса. Отвечай на русском, коротко и по делу.",
            input=prompt,
            max_output_tokens=500
        )
        return response.output_text
    except Exception as e:
        print(f"❌ Ошибка YandexGPT: {e}")
        return None

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
    """Категоризация без ИИ (fallback)"""
    # Если нет YandexGPT, используем простую логику
    if not description or description == 'nan':
        return 'other'
    # Простая категоризация по ключевым словам
    keywords = {
        'rent': ['аренда', 'офис', 'помещение'],
        'supplies': ['сырьё', 'товар', 'закуп', 'материал'],
        'advertising': ['реклама', 'продвижение', 'маркетинг'],
        'taxes': ['налог', 'сбор', 'пошлина'],
        'transport': ['транспорт', 'доставка', 'логистика'],
        'food': ['продукт', 'еда', 'питание'],
        'cafe': ['кафе', 'ресторан', 'обед'],
        'education': ['образование', 'курс', 'обучение']
    }
    desc_lower = description.lower()
    for cat, words in keywords.items():
        for word in words:
            if word in desc_lower:
                return cat
    return 'other'

def get_savings_tips(expenses_by_category, total_expense, top_expenses):
    if not expenses_by_category or total_expense == 0:
        return "• Загрузите выписку с расходами для получения персонализированных советов\n• Анализируйте самые большие категории расходов\n• Сравнивайте цены у разных поставщиков"
    
    categories_text = "\n".join([f"- {cat}: {amount:.2f} руб." for cat, amount in list(expenses_by_category.items())[:5]])
    
    # Пробуем получить советы от YandexGPT
    if YANDEX_CLOUD_API_KEY:
        prompt = f"""Расходы микробизнеса за период:
{categories_text}

Напиши 3 коротких конкретных совета по экономии для этого бизнеса.
Каждый совет начинай с новой строки и ставь в начале символ "•".
Напиши 3 совета именно для этих расходов:"""
        try:
            tips = ask_yandex(prompt)
            if tips and '•' in tips and len(tips) > 50:
                return tips
        except:
            pass
    
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
    
    date_col = None
    for col in df.columns:
        col_lower = col.lower()
        if 'date' in col_lower or 'дата' in col_lower:
            date_col = col
            break
    
    days_count = 0
    months_data = {}
    
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
            date_min = df[date_col].min()
            date_max = df[date_col].max()
            if pd.notna(date_min) and pd.notna(date_max):
                days_count = (date_max - date_min).days + 1
                print(f"📅 Найдена колонка дат: {date_col}, период: {date_min.date()} - {date_max.date()}")
            
            df['year_month'] = df[date_col].dt.strftime('%Y-%m')
            for month in df['year_month'].unique():
                month_df = df[df['year_month'] == month]
                month_income = month_df[month_df['amount'] > 0]['amount'].sum()
                month_expense = abs(month_df[month_df['amount'] < 0]['amount'].sum())
                months_data[month] = {
                    'income': float(month_income),
                    'expense': float(month_expense),
                    'profit': float(month_income - month_expense)
                }
            print(f"📊 Данных по месяцам: {len(months_data)} месяцев")
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
    
    tips = get_savings_tips(categories, total_expense, [])
    predicted_total, predicted_change, _ = predict_next_month(categories, total_expense, days_count)
    
    forecast_3months = []
    cash_gaps = []
    
    if len(months_data) >= 1:
        sorted_months = sorted(months_data.keys())
        monthly_expenses = [months_data[m]['expense'] for m in sorted_months]
        
        if len(monthly_expenses) >= 3:
            trend = (monthly_expenses[-1] - monthly_expenses[0]) / len(monthly_expenses)
        else:
            trend = 0
        
        avg_expense = sum(monthly_expenses) / len(monthly_expenses)
        last_expense = monthly_expenses[-1] if monthly_expenses else total_expense
        
        for i in range(1, 4):
            predicted_expense = last_expense + (trend * i)
            if predicted_expense <= 0:
                predicted_expense = avg_expense
            
            if len(monthly_expenses) >= 2:
                income_trend = (months_data[sorted_months[-1]]['income'] - months_data[sorted_months[0]]['income']) / len(monthly_expenses)
                predicted_income = months_data[sorted_months[-1]]['income'] + (income_trend * i)
            else:
                predicted_income = total_income
            
            predicted_profit = predicted_income - predicted_expense
            
            risk_level = "low"
            risk_text = "🟢 Низкий"
            if predicted_profit < 0:
                risk_level = "critical"
                risk_text = "🔴 Критический"
                cash_gaps.append({
                    'month': i,
                    'shortage': abs(predicted_profit),
                    'advice': f"Ожидается нехватка {abs(predicted_profit):.2f} ₽"
                })
            elif predicted_profit < last_expense * 0.1:
                risk_level = "medium"
                risk_text = "🟡 Средний"
            
            forecast_3months.append({
                'month': i,
                'income': round(predicted_income, 2),
                'expense': round(predicted_expense, 2),
                'profit': round(predicted_profit, 2),
                'risk_level': risk_level,
                'risk_text': risk_text
            })
    
    comparison = {'has_data': False}
    monthly_comparison = {}
    
    if len(months_data) >= 2:
        sorted_months = sorted(months_data.keys())
        current_month_key = sorted_months[-1]
        previous_month_key = sorted_months[-2]
        
        current = months_data[current_month_key]
        previous = months_data[previous_month_key]
        
        income_change = ((current['income'] - previous['income']) / previous['income'] * 100) if previous['income'] > 0 else 0
        expense_change = ((current['expense'] - previous['expense']) / previous['expense'] * 100) if previous['expense'] > 0 else 0
        profit_change = ((current['profit'] - previous['profit']) / previous['profit'] * 100) if previous['profit'] != 0 else 0
        
        comparison = {
            'has_data': True,
            'current_month': current_month_key,
            'previous_month': previous_month_key,
            'income': {
                'current': current['income'],
                'previous': previous['income'],
                'change': round(income_change, 1),
                'change_abs': round(current['income'] - previous['income'], 2)
            },
            'expense': {
                'current': current['expense'],
                'previous': previous['expense'],
                'change': round(expense_change, 1),
                'change_abs': round(current['expense'] - previous['expense'], 2)
            },
            'profit': {
                'current': current['profit'],
                'previous': previous['profit'],
                'change': round(profit_change, 1),
                'change_abs': round(current['profit'] - previous['profit'], 2)
            }
        }
        
        for month in sorted_months[-6:]:
            monthly_comparison[month] = months_data[month]
    
    budget_alerts = check_budget_alerts(categories)
    
    cash_gap_warning = None
    if cash_gaps:
        first_gap = cash_gaps[0]
        cash_gap_warning = f"⚠️ {first_gap['advice']}. Рекомендуется сократить расходы или увеличить доходы."
    elif net_profit < 0:
        cash_gap_warning = f"⚠️ Расходы превышают доходы на {abs(net_profit):.2f} ₽. Рекомендуется сократить расходы или увеличить доходы."
    
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
    
    result = {
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
        'monthly_comparison': monthly_comparison,
        'forecast_3months': forecast_3months,
        'budget_alerts': budget_alerts,
        'months_data': months_data,
        'insights': []
    }
    
    last_analysis_result = result
    save_analysis_to_db(filename, result)
    
    return result
    # ============ ЭНДПОИНТЫ ============

@app.get("/history")
async def get_history():
    history = get_history_from_db()
    return JSONResponse(history)

@app.get("/history/{analysis_id}")
async def get_history_item(analysis_id: int):
    result = get_analysis_by_id(analysis_id)
    if result:
        return JSONResponse(result)
    return JSONResponse({'error': 'Не найдено'}, status_code=404)

@app.post("/set-budget")
async def set_budget(request: Request):
    data = await request.json()
    category = data.get('category')
    limit = data.get('limit')
    month = data.get('month')
    if not category or limit is None:
        return JSONResponse({'error': 'Укажите категорию и лимит'}, status_code=400)
    set_budget_limit(category, float(limit), month)
    return JSONResponse({'success': True, 'message': f'Лимит для {category} установлен: {limit} ₽'})

@app.get("/get-budgets")
async def get_budgets():
    budgets = get_budget_limits()
    return JSONResponse(budgets)

@app.get("/monthly-comparison")
async def get_monthly_comparison():
    global last_analysis_result
    if not last_analysis_result:
        return JSONResponse({'error': 'Нет данных. Сначала загрузите выписку.'}, status_code=400)
    
    return JSONResponse({
        'comparison': last_analysis_result.get('comparison', {}),
        'monthly_data': last_analysis_result.get('monthly_comparison', {}),
        'forecast': last_analysis_result.get('forecast_3months', [])
    })

@app.get("/cash-gap-forecast")
async def get_cash_gap_forecast():
    global last_analysis_result
    if not last_analysis_result:
        return JSONResponse({'error': 'Нет данных. Сначала загрузите выписку.'}, status_code=400)
    
    forecast = last_analysis_result.get('forecast_3months', [])
    cash_gaps = last_analysis_result.get('cash_gaps', [])
    
    return JSONResponse({
        'forecast': forecast,
        'cash_gaps': cash_gaps,
        'has_warning': len(cash_gaps) > 0
    })

@app.post("/ask")
async def ask_question(request: Request):
    global last_analysis_result
    data = await request.json()
    question = data.get('question', '')
    
    if not last_analysis_result:
        return JSONResponse({'answer': 'Сначала загрузите и проанализируйте выписку'})
    
    context = f"""
Данные о финансах микробизнеса:
Доходы: {last_analysis_result['income']:.2f} ₽
Расходы: {last_analysis_result['expense']:.2f} ₽
Чистая прибыль: {last_analysis_result['net_profit']:.2f} ₽
Рентабельность: {last_analysis_result.get('profitability', 0)}%
Расходы по категориям:
"""
    for cat, amount in last_analysis_result.get('categories', {}).items():
        context += f"- {cat}: {amount:.2f} ₽\n"
    
    prompt = f"""
Пользователь задаёт вопрос: "{question}"

Вот данные о финансах:
{context}

Ответь коротко, конкретно и полезно. Используй цифры из данных.
"""
    
    answer = ask_yandex(prompt)
    if answer:
        return JSONResponse({'answer': answer})
    else:
        return JSONResponse({'answer': '❌ Ошибка YandexGPT. Проверьте API ключ в настройках Render.'})

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

# ============ HTML (УПРОЩЁННАЯ ВЕРСИЯ) ============
# Полный HTML-код здесь (я даю минимальную версию, чтобы код поместился)

html_content = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CashFlow — ИИ финансовый ассистент</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400;14..32,600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0a0a0a, #1a1a1a);
            min-height: 100vh;
            padding: 20px;
            color: #fff;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .card {
            background: rgba(17,17,17,0.85);
            backdrop-filter: blur(10px);
            border-radius: 28px;
            padding: 24px;
            margin-bottom: 20px;
            border: 1px solid rgba(234,88,12,0.3);
        }
        h1 {
            font-size: 2rem;
            background: linear-gradient(135deg, #f97316, #ea580c);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }
        .upload-area {
            border: 2px dashed rgba(234,88,12,0.3);
            border-radius: 20px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
        }
        .upload-area:hover { border-color: #f97316; background: rgba(234,88,12,0.1); }
        .btn {
            background: linear-gradient(135deg, #ea580c, #9a3412);
            color: white;
            border: none;
            padding: 12px 28px;
            border-radius: 40px;
            cursor: pointer;
        }
        .result-stats { display: flex; gap: 16px; flex-wrap: wrap; }
        .stat-card { flex: 1; background: rgba(0,0,0,0.5); padding: 20px; border-radius: 16px; text-align: center; }
        .stat-card .value { font-size: 1.8rem; font-weight: bold; }
        .income .value { color: #f97316; }
        .expense .value { color: #ef4444; }
        .suggestion-btn {
            background: rgba(234,88,12,0.15);
            border: 1px solid rgba(234,88,12,0.3);
            padding: 10px 20px;
            border-radius: 40px;
            cursor: pointer;
            color: white;
        }
        .chat-messages { height: 250px; overflow-y: auto; border: 1px solid rgba(234,88,12,0.3); border-radius: 16px; padding: 16px; margin-bottom: 16px; background: rgba(0,0,0,0.3); }
        .chat-message-user { text-align: right; margin: 8px 0; }
        .chat-message-user span { background: linear-gradient(135deg, #ea580c, #9a3412); padding: 8px 16px; border-radius: 20px; display: inline-block; max-width: 80%; }
        .chat-message-bot { text-align: left; margin: 8px 0; }
        .chat-message-bot span { background: rgba(0,0,0,0.5); padding: 8px 16px; border-radius: 20px; display: inline-block; max-width: 80%; border: 1px solid rgba(234,88,12,0.3); }
        .chat-input { display: flex; gap: 10px; }
        .chat-input input { flex: 1; padding: 12px; border: 1px solid rgba(234,88,12,0.3); border-radius: 40px; background: rgba(0,0,0,0.5); color: white; }
        @media (max-width: 768px) { .result-stats { flex-direction: column; } }
    </style>
</head>
<body>
<div class="container">
    <div class="card"><h1>💰 CashFlow</h1><p>ИИ-финансовый ассистент</p></div>
    
    <div class="card">
        <div class="upload-area" onclick="document.getElementById('fileInput').click()">
            <div style="font-size:48px;">📁</div>
            <p>Нажмите или перетащите файл</p>
            <input type="file" id="fileInput" accept=".csv,.xlsx,.xls" style="display:none;">
        </div>
        <div id="fileName" style="margin-top:10px;display:none;"></div>
        <div style="display:flex;gap:10px;margin-top:20px;">
            <button class="btn" id="analyzeBtn" onclick="uploadFile()" disabled>📊 Анализировать</button>
            <button class="btn" onclick="downloadTemplate()" style="background:#2a2a2a;">📥 Шаблон CSV</button>
        </div>
    </div>
    
    <div id="loading" style="display:none;text-align:center;padding:20px;"><div style="border:4px solid rgba(234,88,12,0.3);border-top:4px solid #f97316;border-radius:50%;width:50px;height:50px;animation:spin 1s linear infinite;margin:0 auto;"></div><p>Анализирую...</p></div>
    
    <div id="resultContainer" style="display:none;">
        <div class="card"><h3>🤖 Анализ выполнен!</h3><div id="suggestionButtons"></div></div>
        <div id="fullReport" class="card" style="display:none;"></div>
        <div id="comparisonBlock" class="card" style="display:none;"></div>
        <div id="forecastBlock" class="card" style="display:none;"></div>
        <div id="tipsBlock" class="card" style="display:none;"></div>
        <div id="categoriesBlock" class="card" style="display:none;"></div>
        <div id="chatBlock" class="card" style="display:none;">
            <h3>💬 Чат с ИИ</h3>
            <div class="chat-messages" id="chatMessages"><div>Задайте вопрос о финансах</div></div>
            <div class="chat-input"><input type="text" id="questionInput" placeholder="Например: на чём мне сэкономить?"><button class="btn" onclick="askQuestion()">Отправить</button></div>
        </div>
    </div>
</div>

<script>
let analysisData = null;
const fileInput = document.getElementById('fileInput'), analyzeBtn = document.getElementById('analyzeBtn');

fileInput.onchange = () => {
    if (fileInput.files.length) {
        document.getElementById('fileName').textContent = "Выбран: " + fileInput.files[0].name;
        document.getElementById('fileName').style.display = 'block';
        analyzeBtn.disabled = false;
    }
};

function downloadTemplate() { window.location.href = '/download-template'; }

async function uploadFile() {
    if (!fileInput.files.length) return;
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    document.getElementById('loading').style.display = 'block';
    document.getElementById('resultContainer').style.display = 'none';
    try {
        const res = await fetch('/upload', { method: 'POST', body: formData });
        const data = await res.json();
        analysisData = data;
        showSmartSuggestions(data);
        document.getElementById('resultContainer').style.display = 'block';
    } catch(e) { alert('Ошибка: ' + e.message); }
    finally { document.getElementById('loading').style.display = 'none'; }
}

function showSmartSuggestions(data) {
    const buttons = [
        { text: '📈 Полный отчёт', func: showFullReport },
        { text: '📊 Сравнение', func: showComparison },
        { text: '⚠️ Прогноз', func: showForecast },
        { text: '💡 Советы', func: showTips },
        { text: '📊 Категории', func: showCategories },
        { text: '💬 Чат', func: showChat }
    ];
    document.getElementById('suggestionButtons').innerHTML = buttons.map(b => `<button class="suggestion-btn" onclick="${b.func}()">${b.text}</button>`).join('');
}

function showFullReport() {
    const d = analysisData;
    document.getElementById('fullReport').innerHTML = `
        <h3>📊 Отчёт</h3>
        <div class="result-stats">
            <div class="stat-card income"><div class="value">${d.income.toFixed(2)} ₽</div><div>💰 Доходы</div></div>
            <div class="stat-card expense"><div class="value">${d.expense.toFixed(2)} ₽</div><div>💸 Расходы</div></div>
            <div class="stat-card"><div class="value" style="color:${d.net_profit>=0?'#f97316':'#ef4444'}">${d.net_profit>=0?'+':''}${d.net_profit.toFixed(2)} ₽</div><div>✅ Прибыль</div></div>
        </div>
        <div class="result-stats">
            <div class="stat-card"><div class="value">${d.profitability}%</div><div>📈 Рентабельность</div></div>
            <div class="stat-card"><div class="value">${d.avg_check.toFixed(2)} ₽</div><div>💰 Средний чек</div></div>
        </div>
        <div style="background:rgba(234,88,12,0.15);padding:10px;border-radius:12px;margin-top:10px;">📊 Обработано строк: ${d.rows_count}</div>
        ${d.cash_gap_warning ? `<div style="background:rgba(239,68,68,0.2);color:#ef4444;padding:10px;border-radius:12px;margin-top:10px;">⚠️ ${d.cash_gap_warning}</div>` : ''}
    `;
    showBlock('fullReport');
}

function showComparison() {
    const comp = analysisData.comparison || {};
    if (!comp.has_data) {
        document.getElementById('fullReport').innerHTML = '<p>Нет данных для сравнения</p>';
        showBlock('fullReport');
        return;
    }
    document.getElementById('fullReport').innerHTML = `
        <h3>📊 Сравнение с прошлым месяцем</h3>
        <div class="result-stats">
            <div class="stat-card"><div>💰 Доходы</div><div class="value">${comp.current_income.toFixed(2)} ₽</div><div>было: ${comp.last_income.toFixed(2)} ₽</div><div style="color:${comp.income_change>=0?'#10b981':'#ef4444'}">${comp.income_change>=0?'+':''}${comp.income_change.toFixed(1)}%</div></div>
            <div class="stat-card"><div>💸 Расходы</div><div class="value">${comp.current_expense.toFixed(2)} ₽</div><div>было: ${comp.last_expense.toFixed(2)} ₽</div><div style="color:${comp.expense_change<=0?'#10b981':'#ef4444'}">${comp.expense_change>=0?'+':''}${comp.expense_change.toFixed(1)}%</div></div>
            <div class="stat-card"><div>✅ Прибыль</div><div class="value">${comp.current_profit.toFixed(2)} ₽</div><div>было: ${comp.last_profit.toFixed(2)} ₽</div><div style="color:${comp.profit_change>=0?'#10b981':'#ef4444'}">${comp.profit_change>=0?'+':''}${comp.profit_change.toFixed(1)}%</div></div>
        </div>
    `;
    showBlock('fullReport');
}

function showForecast() {
    const forecast = analysisData.forecast_3months || [];
    if (!forecast.length) {
        document.getElementById('forecastBlock').innerHTML = '<p>Нет данных для прогноза</p>';
        showBlock('forecastBlock');
        return;
    }
    let html = '<h3>⚠️ Прогноз на 3 месяца</h3>';
    for (const m of forecast) {
        html += `<div style="background:rgba(0,0,0,0.3);padding:12px;border-radius:16px;margin-bottom:10px;border-left:4px solid ${m.profit>=0?'#10b981':'#ef4444'}">
            <strong>📅 Месяц ${m.month}</strong>
            <div>💰 Доходы: ${m.income.toFixed(2)} ₽</div>
            <div>💸 Расходы: ${m.expense.toFixed(2)} ₽</div>
            <div>✅ Прибыль: ${m.profit>=0?'+':''}${m.profit.toFixed(2)} ₽</div>
            <div style="background:rgba(234,88,12,0.15);padding:6px;border-radius:8px;margin-top:6px;">${m.risk_text} риск</div>
        </div>`;
    }
    document.getElementById('forecastBlock').innerHTML = html;
    showBlock('forecastBlock');
}

function showTips() {
    const d = analysisData;
    if (d.tips) {
        const items = d.tips.split('•').filter(i=>i.trim());
        document.getElementById('fullReport').innerHTML = `<h3>💡 Советы</h3><ul>${items.map(i=>`<li>• ${i.trim()}</li>`).join('')}</ul>`;
        showBlock('fullReport');
    } else {
        document.getElementById('fullReport').innerHTML = '<p>Нет советов</p>';
        showBlock('fullReport');
    }
}

function showCategories() {
    const d = analysisData;
    if (d.categories && Object.keys(d.categories).length) {
        let html = '<h3>📊 Категории</h3><table style="width:100%"><tr><th>Категория</th><th>Сумма</th></tr>';
        for (const [cat, amt] of Object.entries(d.categories)) {
            html += `<tr><td>${cat}</td><td>${amt.toFixed(2)} ₽</td></tr>`;
        }
        html += '</table>';
        document.getElementById('fullReport').innerHTML = html;
        showBlock('fullReport');
    } else {
        document.getElementById('fullReport').innerHTML = '<p>Нет данных</p>';
        showBlock('fullReport');
    }
}

function showChat() {
    document.getElementById('resultContainer').style.display = 'block';
    showBlock('chatBlock');
}

function showBlock(id) {
    const blocks = ['fullReport','comparisonBlock','forecastBlock','tipsBlock','categoriesBlock','chatBlock'];
    blocks.forEach(b => { const el = document.getElementById(b); if (el) el.style.display = 'none'; });
    document.getElementById(id).style.display = 'block';
}

async function askQuestion() {
    const q = document.getElementById('questionInput').value.trim();
    if (!q) return;
    const chatDiv = document.getElementById('chatMessages');
    chatDiv.innerHTML += `<div class="chat-message-user"><span>${escapeHtml(q)}</span></div>`;
    document.getElementById('questionInput').value = '';
    chatDiv.innerHTML += `<div class="chat-message-bot"><span>🤔 ИИ думает...</span></div>`;
    chatDiv.scrollTop = chatDiv.scrollHeight;
    try {
        const res = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: q })
        });
        const data = await res.json();
        chatDiv.innerHTML = chatDiv.innerHTML.replace('<div class="chat-message-bot"><span>🤔 ИИ думает...</span></div>', '');
        chatDiv.innerHTML += `<div class="chat-message-bot"><span>${escapeHtml(data.answer)}</span></div>`;
        chatDiv.scrollTop = chatDiv.scrollHeight;
    } catch(e) {
        chatDiv.innerHTML = chatDiv.innerHTML.replace('<div class="chat-message-bot"><span>🤔 ИИ думает...</span></div>', '');
        chatDiv.innerHTML += `<div class="chat-message-bot"><span>❌ Ошибка</span></div>`;
    }
}

function escapeHtml(t) { const d=document.createElement('div'); d.textContent=t; return d.innerHTML; }
</script>
<style>
@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}
</style>
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

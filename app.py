from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv
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
    cursor.execute('''
        INSERT INTO analyses (date, filename, income, expense, net_profit, categories, months_data, forecast, full_data)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        filename,
        result.get('income'),
        result.get('expense'),
        result.get('net_profit'),
        json.dumps(result.get('categories', {})),
        json.dumps(result.get('months_data', {})),
        json.dumps(result.get('forecast_3months', [])),
        json.dumps(result)
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

# ============ DEEPSEEK (ИСПРАВЛЕННАЯ ВЕРСИЯ - БЕЗ PROXIES) ============
# ============ DEEPSEEK ============
deepseek_client = None
try:
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if api_key:
        deepseek_client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        print("✅ DeepSeek API подключен")
    else:
        print("⚠️ DEEPSEEK_API_KEY не найдена")
except Exception as e:
    print(f"❌ Ошибка DeepSeek: {e}")

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
    if deepseek_client is None or not description or description == 'nan':
        return 'other'
    prompt = f"""
    Определи категорию расхода для операции: "{description}"
    Категории: rent, supplies, advertising, taxes, transport, food, cafe, education, other
    Ответь ТОЛЬКО одним словом из этих вариантов.
    """
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=20
        )
        category = response.choices[0].message.content.strip().lower()
        return category if category in category_names else 'other'
    except Exception as e:
        print(f"Ошибка категоризации: {e}")
        return 'other'

def get_savings_tips(expenses_by_category, total_expense, top_expenses):
    if deepseek_client is None or not expenses_by_category or total_expense == 0:
        return "• Загрузите выписку с расходами для получения персонализированных советов\n• Анализируйте самые большие категории расходов\n• Сравнивайте цены у разных поставщиков"
    
    categories_text = "\n".join([f"- {cat}: {amount:.2f} руб." for cat, amount in list(expenses_by_category.items())[:5]])
    prompt = f"""Расходы микробизнеса за период:
{categories_text}

Напиши 3 коротких конкретных совета по экономии для этого бизнеса.
Каждый совет начинай с новой строки и ставь в начале символ "•".
Напиши 3 совета именно для этих расходов:"""
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
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
    
    if deepseek_client is None:
        return JSONResponse({'answer': '❌ DeepSeek не подключен. Проверьте API ключ в настройках Render.'})
    
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
    try:
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Ты финансовый ассистент для микробизнеса. Отвечай на русском, коротко и по делу. Используй цифры из данных."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        answer = response.choices[0].message.content
        return JSONResponse({'answer': answer})
    except Exception as e:
        error_msg = str(e)
        if "insufficient_quota" in error_msg.lower() or "billing" in error_msg.lower():
            return JSONResponse({'answer': '⚠️ Закончились бесплатные токены DeepSeek. Пополните баланс или получите новый API-ключ.'})
        return JSONResponse({'answer': f'❌ Ошибка DeepSeek: {error_msg}'})

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

# ============ HTML (полный старый дизайн) ============
html_content = """<!DOCTYPE html>
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
            --hover-shadow: 0 12px 40px rgba(234, 88, 12, 0.2);
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
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); background: linear-gradient(135deg, #ea580c 0%, #f97316 100%); }
            100% { transform: scale(1); }
        }
        @keyframes glow {
            0% { box-shadow: 0 0 5px rgba(249,115,22,0.5); }
            100% { box-shadow: 0 0 20px rgba(249,115,22,0.8); }
        }
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: var(--card-bg); border-radius: 10px; }
        ::-webkit-scrollbar-thumb { background: var(--accent); border-radius: 10px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--primary-start); }
        .container { max-width: 1200px; margin: 0 auto; position: relative; z-index: 1; }
        .card {
            background: var(--card-bg);
            backdrop-filter: var(--backdrop-blur);
            border-radius: 28px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--card-shadow);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            animation: fadeInUp 0.4s ease-out;
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }
        .card:hover { transform: translateY(-4px); box-shadow: var(--hover-shadow); }
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
        .upload-area:hover { border-color: var(--accent); background: rgba(234, 88, 12, 0.1); transform: scale(1.01); }
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
            box-shadow: 0 4px 15px rgba(234, 88, 12, 0.3);
        }
        .btn:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 8px 25px rgba(234, 88, 12, 0.5);
            background: linear-gradient(135deg, var(--primary-end), var(--primary-start));
            animation: glow 0.5s ease;
        }
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
            transition: all 0.2s;
            color: var(--text-primary);
            font-size: 0.9rem;
        }
        .suggestion-btn:hover { 
            background: rgba(234, 88, 12, 0.4); 
            transform: translateY(-2px);
            animation: pulse 0.4s ease;
        }
        .result-stats {
            display: flex;
            gap: 1rem;
            flex-wrap: wrap;
            margin-bottom: 1.5rem;
            animation: fadeIn 0.5s ease-out;
        }
        @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
        .stat-card {
            flex: 1;
            background: var(--stat-bg);
            padding: 1rem;
            border-radius: 20px;
            text-align: center;
            transition: transform 0.2s;
        }
        .stat-card:hover { transform: translateY(-4px); }
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
        .progress-container {
            height: 8px;
            background: var(--border-color);
            border-radius: 4px;
            margin-top: 1rem;
            overflow: hidden;
            display: none;
        }
        .progress-bar {
            width: 0%;
            height: 100%;
            background: linear-gradient(90deg, var(--accent), var(--primary-start));
            border-radius: 4px;
            transition: width 0.3s ease;
        }
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
        .chat-input { display: flex; gap: 0.8rem; flex-wrap: wrap; }
        .chat-input input {
            flex: 1;
            padding: 12px 16px;
            border: 1px solid var(--border-color);
            border-radius: 40px;
            background: var(--card-bg);
            color: var(--text-primary);
        }
        .mobile-header { display: none; justify-content: space-between; align-items: center; margin-bottom: 1rem; background: rgba(0,0,0,0.5); backdrop-filter: blur(10px); padding: 0.8rem 1.2rem; border-radius: 50px; }
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
        .seasonality-container { display: flex; flex-direction: column; gap: 1.5rem; }
        .seasonality-card { background: rgba(0,0,0,0.3); border-radius: 20px; padding: 1rem; }
        .seasonality-card h4 { margin-bottom: 1rem; font-size: 1rem; display: flex; align-items: center; gap: 0.5rem; }
        .bar-chart-modern {
            display: flex;
            justify-content: space-around;
            align-items: flex-end;
            gap: 0.5rem;
            overflow-x: auto;
            padding: 0.5rem 0;
        }
        .bar-item { text-align: center; min-width: 60px; }
        .bar-label { font-size: 0.7rem; margin-bottom: 0.3rem; }
        .bar-wrapper { height: 120px; display: flex; align-items: flex-end; justify-content: center; margin-bottom: 0.3rem; }
        .bar-fill { width: 30px; border-radius: 12px 12px 0 0; transition: height 0.6s ease-out; }
        .bar-value { font-size: 0.7rem; font-weight: bold; color: #f97316; }
        .cost-input-grid {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .cost-input-grid input {
            background: rgba(0,0,0,0.5);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 12px 16px;
            color: var(--text-primary);
            font-size: 0.9rem;
        }
        .cost-input-grid input:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(249,115,22,0.2);
        }
        .cost-result-card {
            background: rgba(234, 88, 12, 0.1);
            border: 1px solid rgba(249,115,22,0.3);
            border-radius: 20px;
            padding: 1.2rem;
            margin-top: 1rem;
            backdrop-filter: blur(5px);
        }
        .cost-result-header {
            font-size: 1rem;
            font-weight: bold;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--accent);
            border-bottom: 1px solid rgba(249,115,22,0.3);
            padding-bottom: 0.5rem;
        }
        .cost-result-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            justify-content: space-between;
        }
        .cost-result-item {
            flex: 1;
            min-width: 140px;
            background: rgba(0,0,0,0.4);
            border-radius: 16px;
            padding: 1rem;
            text-align: center;
            transition: transform 0.2s;
        }
        .cost-result-item:hover { transform: translateY(-2px); }
        .cost-result-icon { font-size: 1.8rem; color: var(--accent); margin-bottom: 0.5rem; }
        .cost-result-label { font-size: 0.7rem; opacity: 0.8; margin-top: 0.3rem; }
        .cost-result-value { font-size: 1.2rem; font-weight: bold; margin-top: 0.3rem; color: var(--accent); }
        .forecast-card {
            background: rgba(0,0,0,0.3);
            border-radius: 16px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .risk-critical { border-left: 4px solid #ef4444; }
        .risk-medium { border-left: 4px solid #f59e0b; }
        .risk-low { border-left: 4px solid #10b981; }
        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .comparison-item {
            text-align: center;
            padding: 1rem;
            background: rgba(0,0,0,0.3);
            border-radius: 16px;
        }
        .change-positive { color: #10b981; }
        .change-negative { color: #ef4444; }
        .alert-warning {
            background: rgba(239, 68, 68, 0.2);
            border-left: 4px solid #ef4444;
            padding: 12px;
            margin: 10px 0;
            border-radius: 12px;
        }
        .budget-input {
            display: flex;
            gap: 10px;
            margin: 10px 0;
            flex-wrap: wrap;
        }
        .budget-input input, .budget-input select {
            padding: 10px;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            background: var(--stat-bg);
            color: var(--text-primary);
        }
        .history-item {
            background: rgba(0,0,0,0.3);
            padding: 12px;
            margin: 8px 0;
            border-radius: 16px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .history-item:hover {
            background: rgba(234,88,12,0.2);
            transform: translateX(5px);
        }
        @media (max-width: 768px) {
            body { padding: 0.8rem; }
            .desktop-title { display: none; }
            .mobile-header { display: flex; }
            .result-stats { flex-direction: column; }
            .suggestion-buttons { flex-direction: column; }
            .bar-item { min-width: 45px; }
            .bar-fill { width: 25px; }
            .cost-result-grid { flex-direction: column; }
            .comparison-grid { grid-template-columns: 1fr; }
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
        <div class="progress-container" id="progressContainer"><div class="progress-bar" id="progressBar"></div></div>
        <div style="display: flex; gap: 10px; margin-top: 1rem;">
            <button class="btn" id="analyzeBtn" onclick="uploadFile()" disabled style="flex: 1;">📊 Анализировать</button>
            <button class="btn" onclick="downloadTemplate()" style="flex: 0; background: #2a2a2a; border: 1px solid #f97316;">📥 Шаблон CSV</button>
        </div>
    </div>
    <div id="loading" style="display:none;text-align:center;padding:2rem;"><div class="spinner"></div><p>Анализирую выписку с помощью ИИ...</p></div>
    <div id="resultContainer" style="display:none;">
        <div class="card" id="suggestionCard">
            <h3><i class="fas fa-robot"></i> Анализ выполнен!</h3>
            <div id="budgetAlerts"></div>
            <div id="insightsContainer"></div>
            <div id="suggestionButtons" class="suggestion-buttons"></div>
        </div>
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
        <div id="budgetBlock" class="card" style="display:none;">
            <h3><i class="fas fa-chart-line"></i> Планирование бюджета</h3>
            <div class="budget-input">
                <select id="budgetCategory"></select>
                <input type="number" id="budgetLimit" placeholder="Лимит в ₽">
                <button class="btn" onclick="setBudget()">Установить лимит</button>
            </div>
            <div id="budgetList"></div>
        </div>
        <div id="historyBlock" class="card" style="display:none;">
            <h3><i class="fas fa-history"></i> История анализов</h3>
            <div id="historyList"></div>
        </div>
        <div id="chatBlock" class="card" style="display:none;"><h3>Чат с ИИ</h3><div class="chat-messages" id="chatMessages"><div>Задайте вопрос о финансах</div></div><div class="chat-input"><input type="text" id="questionInput" placeholder="Например: на чём мне сэкономить?"><button class="btn" onclick="askQuestion()">Отправить</button></div></div>
    </div>
</div>

<script>
let selectedFile = null, analysisData = null, expenseChart = null, trendChart = null;
const fileInput = document.getElementById('fileInput'), analyzeBtn = document.getElementById('analyzeBtn'), fileNameDiv = document.getElementById('fileName');
const progressContainer = document.getElementById('progressContainer'), progressBar = document.getElementById('progressBar');

function handleFileSelect() { 
    if(fileInput.files.length){ 
        selectedFile = fileInput.files[0]; 
        fileNameDiv.textContent = "Выбран файл: "+selectedFile.name; 
        fileNameDiv.style.display = "block"; 
        analyzeBtn.disabled = false; 
    } 
}
fileInput.onchange = handleFileSelect;

const dropZone = document.querySelector('.upload-area');
dropZone.ondragover = (e) => { e.preventDefault(); dropZone.style.borderColor = '#f97316'; };
dropZone.ondragleave = () => dropZone.style.borderColor = 'var(--border-color)';
dropZone.ondrop = (e) => { 
    e.preventDefault(); 
    dropZone.style.borderColor = 'var(--border-color)'; 
    if(e.dataTransfer.files.length){ 
        fileInput.files = e.dataTransfer.files; 
        handleFileSelect(); 
    } 
};

function downloadTemplate() {
    window.location.href = '/download-template';
}

async function uploadFile() {
    if(!selectedFile) return;
    
    const formData = new FormData();
    formData.append('file', selectedFile);
    progressContainer.style.display = 'block'; progressBar.style.width = '0%';
    document.getElementById('loading').style.display = 'block';
    document.getElementById('resultContainer').style.display = 'none';
    let progress = 0; const interval = setInterval(() => { progress += 10; if(progress>=90) clearInterval(interval); progressBar.style.width = Math.min(progress,90)+'%'; }, 200);
    try { 
        const response = await fetch('/upload',{method:'POST',body:formData}); 
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const result = await response.json(); 
        progressBar.style.width='100%'; 
        setTimeout(()=>{progressContainer.style.display='none';},500);
        analysisData = result; 
        showBudgetAlerts(result.budget_alerts);
        showSmartSuggestions(result); 
    }
    catch(error){ 
        alert('Ошибка: '+error.message); 
        progressContainer.style.display='none';
    }
    finally{ clearInterval(interval); document.getElementById('loading').style.display='none'; }
}

function showBudgetAlerts(alerts) {
    const container = document.getElementById('budgetAlerts');
    if (!alerts || alerts.length === 0) {
        container.innerHTML = '';
        return;
    }
    container.innerHTML = alerts.map(alert => 
        `<div class="alert-warning"><i class="fas fa-exclamation-triangle"></i> ${alert.message}</div>`
    ).join('');
}

function showSmartSuggestions(data) {
    document.getElementById('insightsContainer').innerHTML = '<div class="insight-item"><i class="fas fa-check-circle" style="color:#f97316;"></i> Анализ выполнен успешно</div>';
    const allButtons = [
        { key:'full', text:'📈 Полный отчёт', func:showFullReport },
        { key:'comparison', text:'📊 Сравнение с прошлым месяцем', func:showComparison },
        { key:'forecast', text:'⚠️ Прогноз кассовых разрывов', func:showCashGapForecast },
        { key:'savings', text:'💡 Советы', func:showTips },
        { key:'categories', text:'📊 Категории', func:showCategories },
        { key:'trend', text:'📈 Динамика', func:showTrend },
        { key:'seasonality', text:'📅 Сезонность', func:showSeasonality },
        { key:'cost', text:'💰 Себестоимость', func:showCost },
        { key:'clients', text:'👥 Анализ клиентов', func:showClientAnalysis },
        { key:'budget', text:'💰 Бюджет', func:showBudget },
        { key:'history', text:'📜 История', func:showHistory },
        { key:'chat', text:'💬 Чат', func:showChat }
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
            <div class="stat-card income"><div class="value">${d.income.toFixed(2)} ₽</div><div class="label">💰 Доходы</div></div>
            <div class="stat-card expense"><div class="value">${d.expense.toFixed(2)} ₽</div><div class="label">💸 Расходы</div></div>
            <div class="stat-card"><div class="value" style="color: ${d.net_profit >= 0 ? '#f97316' : '#ef4444'}">${d.net_profit >= 0 ? '+' : ''}${d.net_profit.toFixed(2)} ₽</div><div class="label">✅ Чистая прибыль</div></div>
        </div>
        <div class="result-stats">
            <div class="stat-card"><div class="value">${d.profitability}%</div><div class="label">📈 Рентабельность</div></div>
            <div class="stat-card"><div class="value">${d.avg_check.toFixed(2)} ₽</div><div class="label">💰 Средний чек</div></div>
        </div>
        <div class="info"><i class="fas fa-info-circle"></i> Обработано строк: ${d.rows_count}<br><i class="fas fa-arrow-up"></i> Доходов: ${d.incomes_count}, <i class="fas fa-arrow-down"></i> Расходов: ${d.expenses_count}</div>
        ${d.cash_gap_warning ? `<div class="info" style="background: rgba(239,68,68,0.2); color: #ef4444;"><i class="fas fa-exclamation-triangle"></i> ${d.cash_gap_warning}</div>` : ''}
    `;
    showBlock('fullReport');
}

async function showComparison() {
    const res = await fetch('/monthly-comparison');
    const data = await res.json();
    const comp = data.comparison;
    if (!comp.has_data) {
        document.getElementById('reportContent').innerHTML = '<div class="info"><i class="fas fa-chart-line"></i> Нет данных для сравнения с прошлым месяцем. Загрузите выписку за несколько месяцев.</div>';
        showBlock('fullReport');
        return;
    }
    const incomeColor = comp.income.change >= 0 ? '#10b981' : '#ef4444';
    const expenseColor = comp.expense.change <= 0 ? '#10b981' : '#ef4444';
    const profitColor = comp.profit.change >= 0 ? '#10b981' : '#ef4444';
    document.getElementById('reportContent').innerHTML = `
        <h3><i class="fas fa-chart-line"></i> Сравнение с прошлым месяцем</h3>
        <div class="result-stats">
            <div class="stat-card"><div>💰 Доходы</div><div class="value" style="font-size:1.2rem;">${comp.income.current.toFixed(2)} ₽</div><div>было: ${comp.income.previous.toFixed(2)} ₽</div><div style="color:${incomeColor}">${comp.income.change >= 0 ? '+' : ''}${comp.income.change}%</div></div>
            <div class="stat-card"><div>💸 Расходы</div><div class="value" style="font-size:1.2rem;">${comp.expense.current.toFixed(2)} ₽</div><div>было: ${comp.expense.previous.toFixed(2)} ₽</div><div style="color:${expenseColor}">${comp.expense.change >= 0 ? '+' : ''}${comp.expense.change}%</div></div>
            <div class="stat-card"><div>✅ Прибыль</div><div class="value" style="font-size:1.2rem;">${comp.profit.current.toFixed(2)} ₽</div><div>было: ${comp.profit.previous.toFixed(2)} ₽</div><div style="color:${profitColor}">${comp.profit.change >= 0 ? '+' : ''}${comp.profit.change}%</div></div>
        </div>
    `;
    showBlock('fullReport');
}

async function showCashGapForecast() {
    const res = await fetch('/cash-gap-forecast');
    const data = await res.json();
    if (!data.forecast || data.forecast.length === 0) {
        document.getElementById('forecastContent').innerHTML = '<p>Нет данных для прогноза</p>';
        showBlock('forecastBlock');
        return;
    }
    let html = '<h3><i class="fas fa-chart-line"></i> Прогноз кассовых разрывов на 3 месяца</h3>';
    for (const m of data.forecast) {
        let riskClass = m.risk_level === 'critical' ? 'risk-critical' : (m.risk_level === 'medium' ? 'risk-medium' : 'risk-low');
        let profitColor = m.profit >= 0 ? '#10b981' : '#ef4444';
        let profitSign = m.profit >= 0 ? '+' : '';
        html += `
            <div class="forecast-card ${riskClass}">
                <div style="font-weight: bold; margin-bottom: 8px;">📅 Месяц ${m.month}</div>
                <div class="result-stats" style="margin-bottom: 0;">
                    <div class="stat-card" style="padding: 8px;"><div>💰 Доходы</div><div style="font-weight: bold;">${m.income.toFixed(2)} ₽</div></div>
                    <div class="stat-card" style="padding: 8px;"><div>💸 Расходы</div><div style="font-weight: bold;">${m.expense.toFixed(2)} ₽</div></div>
                    <div class="stat-card" style="padding: 8px;"><div>✅ Прибыль</div><div style="font-weight: bold; color: ${profitColor};">${profitSign}${m.profit.toFixed(2)} ₽</div></div>
                </div>
                <div class="info" style="margin-top: 8px;">${m.risk_text} уровень риска</div>
            </div>
        `;
    }
    document.getElementById('forecastContent').innerHTML = html;
    showBlock('forecastBlock');
}

function showTips() {
    const d = analysisData;
    if(d.tips){
        const items = d.tips.split('•').filter(i=>i.trim());
        document.getElementById('tipsContent').innerHTML = `<div class="tips-box"><h3><i class="fas fa-lightbulb"></i> Советы по экономии</h3><ul>${items.map(i=>`<li><i class="fas fa-check-circle" style="color:#f97316;"></i> ${escapeHtml(i.trim())}</li>`).join('')}</ul></div>`;
    } else document.getElementById('tipsContent').innerHTML = '<p>Нет советов</p>';
    showBlock('tipsBlock');
}

function showCategories() {
    const d = analysisData;
    if(d.categories && Object.keys(d.categories).length){
        let table = '<h3><i class="fas fa-tags"></i> Расходы по категориям</h3><table style="width:100%"><tr><th>Категория</th><th>Сумма (RUB)</th></tr>';
        for(const [cat,amt] of Object.entries(d.categories)){
            const icon = {'Аренда':'🏠','Сырьё и товары':'📦','Реклама':'📢','Налоги':'📄','Транспорт':'🚗','Продукты':'🍎','Кафе и рестораны':'🍽️','Образование':'📚','Прочее':'📌'}[cat] || '💰';
            table += `<tr><td><span class="category-icon">${icon}</span> ${cat}佛罗<td style="text-align:right">${amt.toFixed(2)} ₽</td></tr>`;
        }
        table += '</table>';
        document.getElementById('categoriesContent').innerHTML = table;
        const ctx = document.getElementById('expenseChart').getContext('2d');
        if(expenseChart) expenseChart.destroy();
        expenseChart = new Chart(ctx, { type:'pie', data:{ labels:Object.keys(d.categories), datasets:[{ data:Object.values(d.categories), backgroundColor:['#ea580c','#f97316','#c2410c','#fdba74','#9a3412','#7c2d12','#b45309','#d97706','#a16207'] }] }, options:{ responsive:true } });
    } else document.getElementById('categoriesContent').innerHTML = '<p>Нет данных для категоризации</p>';
    showBlock('categoriesBlock');
}

function showTrend() {
    const d = analysisData;
    const ctx = document.getElementById('trendChart').getContext('2d');
    if(trendChart) trendChart.destroy();
    trendChart = new Chart(ctx, { 
        type:'line', 
        data:{ 
            labels:['Неделя 1', 'Неделя 2', 'Неделя 3', 'Неделя 4'], 
            datasets:[
                { label:'Доходы', data:[d.income*0.6, d.income*0.8, d.income*0.9, d.income], borderColor:'#f97316', backgroundColor:'rgba(249,115,22,0.1)', tension:0.4, fill:true, pointBackgroundColor:'#ea580c', pointBorderColor:'#fff', pointRadius:5, pointHoverRadius:7 },
                { label:'Расходы', data:[d.expense*0.7, d.expense*0.85, d.expense*0.95, d.expense], borderColor:'#ef4444', backgroundColor:'rgba(239,68,68,0.1)', tension:0.4, fill:true, pointBackgroundColor:'#dc2626', pointBorderColor:'#fff', pointRadius:5, pointHoverRadius:7 }
            ] 
        }, 
        options:{ responsive:true, maintainAspectRatio:true, animation:{ duration:1000, easing:'easeOutCubic' }, plugins:{ legend:{ position:'top', labels:{ color:'#ffffff' } } } } 
    });
    showBlock('trendBlock');
}

function showSeasonality() {
    const s = analysisData?.seasonality || {};
    if (!s.has_data || !s.expense_by_month || Object.keys(s.expense_by_month).length === 0) {
        document.getElementById('seasonalityContent').innerHTML = '<div class="info"><i class="fas fa-chart-line"></i> Нет данных для анализа сезонности. Загрузите файл с датами и расходами.</div>';
        showBlock('seasonalityBlock');
        return;
    }
    let html = '<div class="seasonality-container">';
    if(s.expense_by_month && Object.keys(s.expense_by_month).length > 0){
        const months = ['Янв','Фев','Мар','Апр','Май','Июн','Июл','Авг','Сен','Окт','Ноя','Дек'];
        const vals = months.map((_,i)=>s.expense_by_month[i+1]||0);
        const maxVal = Math.max(...vals,1);
        html += '<div class="seasonality-card"><h4><i class="fas fa-calendar-alt"></i> Расходы по месяцам (₽)</h4><div class="bar-chart-modern">';
        vals.forEach((v,i)=>{
            const percent = (v / maxVal) * 100;
            html += `<div class="bar-item">
                        <div class="bar-label">${months[i]}</div>
                        <div class="bar-wrapper">
                            <div class="bar-fill" style="height: ${percent}%; width: 100%; background: linear-gradient(180deg, #f97316, #ea580c);"></div>
                        </div>
                        <div class="bar-value">${v.toFixed(0)} ₽</div>
                    </div>`;
        });
        html += '</div></div>';
    }
    if(s.by_weekday && Object.keys(s.by_weekday).length > 0){
        const days = ['Пн','Вт','Ср','Чт','Пт','Сб','Вс'];
        const vals = days.map(d=>s.by_weekday[d]||0);
        const maxVal = Math.max(...vals,1);
        html += '<div class="seasonality-card"><h4><i class="fas fa-calendar-week"></i> Расходы по дням недели (₽)</h4><div class="bar-chart-modern">';
        vals.forEach((v,i)=>{
            const percent = (v / maxVal) * 100;
            html += `<div class="bar-item">
                        <div class="bar-label">${days[i]}</div>
                        <div class="bar-wrapper">
                            <div class="bar-fill" style="height: ${percent}%; width: 100%; background: linear-gradient(180deg, #3b82f6, #1d4ed8);"></div>
                        </div>
                        <div class="bar-value">${v.toFixed(0)} ₽</div>
                    </div>`;
        });
        html += '</div></div>';
    }
    html += '</div>';
    document.getElementById('seasonalityContent').innerHTML = html;
    showBlock('seasonalityBlock');
}

function showCost() { showBlock('costBlock'); }

function showClientAnalysis() {
    const d = analysisData;
    const clients = d.client_analysis || {};
    if (Object.keys(clients).length === 0) {
        document.getElementById('categoriesContent').innerHTML = '<p>Нет данных для анализа клиентов</p>';
        showBlock('categoriesBlock');
        return;
    }
    let table = '<h3><i class="fas fa-users"></i> Анализ клиентов (источники дохода)</h3><table style="width:100%"><tr><th>Источник</th><th>Сумма (RUB)</th></tr>';
    for (const [source, amount] of Object.entries(clients)) {
        const shortSource = source.length > 40 ? source.substring(0, 37) + '...' : source;
        table += `<tr><td title="${escapeHtml(source)}">${escapeHtml(shortSource)}</td><td style="text-align:right">${amount.toFixed(2)} ₽</td></tr>`;
    }
    table += '</table>';
    document.getElementById('categoriesContent').innerHTML = table;
    showBlock('categoriesBlock');
}

async function showBudget() {
    await loadBudgets();
    showBlock('budgetBlock');
}

async function loadBudgets() {
    const res = await fetch('/get-budgets');
    const budgets = await res.json();
    const select = document.getElementById('budgetCategory');
    if (analysisData && analysisData.categories) {
        select.innerHTML = Object.keys(analysisData.categories).map(cat => `<option value="${cat}">${cat}</option>`).join('');
    }
    const listDiv = document.getElementById('budgetList');
    if (Object.keys(budgets).length === 0) {
        listDiv.innerHTML = '<p>Лимиты не установлены</p>';
    } else {
        listDiv.innerHTML = '<h4>Текущие лимиты:</h4>' + Object.entries(budgets).map(([cat, limit]) => `<div class="info">${cat}: ${limit} ₽</div>`).join('');
    }
}

async function setBudget() {
    const category = document.getElementById('budgetCategory').value;
    const limit = document.getElementById('budgetLimit').value;
    if (!category || !limit) {
        alert('Заполните все поля');
        return;
    }
    await fetch('/set-budget', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ category, limit: parseFloat(limit) })
    });
    alert('Лимит установлен!');
    await loadBudgets();
    location.reload();
}

async function showHistory() {
    const res = await fetch('/history');
    const history = await res.json();
    const listDiv = document.getElementById('historyList');
    if (history.length === 0) {
        listDiv.innerHTML = '<p>Нет сохранённых анализов</p>';
    } else {
        listDiv.innerHTML = history.map(item => `
            <div class="history-item" onclick="loadHistoryItem(${item.id})">
                <strong>${item.date.substring(0, 16)}</strong><br>
                📁 ${item.filename}<br>
                💰 Доходы: ${item.income.toFixed(2)} ₽ | 💸 Расходы: ${item.expense.toFixed(2)} ₽
            </div>
        `).join('');
    }
    showBlock('historyBlock');
}

async function loadHistoryItem(id) {
    const res = await fetch(`/history/${id}`);
    const data = await res.json();
    analysisData = data;
    showBudgetAlerts(data.budget_alerts);
    showSmartSuggestions(data);
    alert('Загружен анализ от ' + data.date);
}

function showChat() { showBlock('chatBlock'); }

async function askQuestion() {
    const q = document.getElementById('questionInput').value.trim();
    if(!q) return;
    const chatDiv = document.getElementById('chatMessages');
    if(chatDiv.children.length===1 && chatDiv.children[0].textContent.includes('Задайте вопрос')) chatDiv.innerHTML = '';
    chatDiv.innerHTML += `<div class="chat-message-user"><span>${escapeHtml(q)}</span></div>`;
    document.getElementById('questionInput').value = '';
    chatDiv.innerHTML += `<div class="typing" style="opacity:0.7;font-style:italic;"><i class="fas fa-spinner fa-pulse"></i> ИИ печатает...</div>`;
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
    const breakeven = Math.ceil(totalExp / (price - (mat + labor)));
    const resultDiv = document.getElementById('costResult');
    resultDiv.style.display = 'block';
    resultDiv.innerHTML = `
        <div class="cost-result-card">
            <div class="cost-result-header"><i class="fas fa-chart-line"></i> Результаты: ${escapeHtml(name)}</div>
            <div class="cost-result-grid">
                <div class="cost-result-item"><div class="cost-result-icon"><i class="fas fa-cubes"></i></div><div class="cost-result-label">Себестоимость единицы</div><div class="cost-result-value" id="costValue">0 ₽</div></div>
                <div class="cost-result-item"><div class="cost-result-icon"><i class="fas fa-tag"></i></div><div class="cost-result-label">Рекомендуемая цена</div><div class="cost-result-value" id="priceValue">0 ₽</div></div>
                <div class="cost-result-item"><div class="cost-result-icon"><i class="fas fa-chart-simple"></i></div><div class="cost-result-label">Точка безубыточности</div><div class="cost-result-value" id="breakevenValue">0 шт./мес</div></div>
            </div>
        </div>
    `;
    function animateValue(id, start, end, suffix) {
        const el = document.getElementById(id);
        if(!el) return;
        const range = end - start;
        const startTime = performance.now();
        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / 1000, 1);
            el.textContent = Math.round(start + (range * progress)) + suffix;
            if(progress < 1) requestAnimationFrame(update);
        }
        requestAnimationFrame(update);
    }
    animateValue('costValue', 0, cost, ' ₽');
    animateValue('priceValue', 0, price, ' ₽');
    animateValue('breakevenValue', 0, breakeven, ' шт./мес');
}

function showBlock(id) {
    const blocks = ['fullReport','forecastBlock','tipsBlock','categoriesBlock','trendBlock','seasonalityBlock','costBlock','chatBlock','budgetBlock','historyBlock'];
    blocks.forEach(b=>{ const el = document.getElementById(b); if(el) el.style.display = 'none'; });
    document.getElementById(id).style.display = 'block';
    if(window.innerWidth<=768 && mobileMenu) mobileMenu.style.display='none';
    window.scrollTo({ top: document.getElementById(id).offsetTop-20, behavior:'smooth' });
}

function escapeHtml(t){ const d=document.createElement('div'); d.textContent=t; return d.innerHTML; }

const menuBtn=document.getElementById('menuBtn'), mobileMenu=document.getElementById('mobileMenu');
if(menuBtn && mobileMenu){
    menuBtn.onclick=()=>{ mobileMenu.style.display=mobileMenu.style.display==='none'?'block':'none'; };
    const items=['Загрузить','Отчёт','Сравнение','Прогноз','Советы','Категории','Динамика','Сезонность','Себестоимость','Клиенты','Бюджет','История','Чат'];
    let html='';
    for(let i of items) html+=`<a href="#" onclick="if(analysisData){ if('${i}'==='Загрузить') document.querySelector('.upload-area').click(); else if('${i}'==='Отчёт') showFullReport(); else if('${i}'==='Сравнение') showComparison(); else if('${i}'==='Прогноз') showCashGapForecast(); else if('${i}'==='Советы') showTips(); else if('${i}'==='Категории') showCategories(); else if('${i}'==='Динамика') showTrend(); else if('${i}'==='Сезонность') showSeasonality(); else if('${i}'==='Себестоимость') showCost(); else if('${i}'==='Клиенты') showClientAnalysis(); else if('${i}'==='Бюджет') showBudget(); else if('${i}'==='История') showHistory(); else if('${i}'==='Чат') showChat(); } else if('${i}'==='Загрузить') document.querySelector('.upload-area').click(); document.getElementById('mobileMenu').style.display='none';">${i}</a>`;
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
        # Очищаем имя файла от проблемных символов
        clean_filename = file.filename.replace(' ', '_').replace('(', '').replace(')', '').replace('[', '').replace(']', '')
        
        print(f"📁 Получен файл: {file.filename} -> {clean_filename}")
        file_content = await file.read()
        
        if len(file_content) == 0:
            return JSONResponse({'error': 'Файл пуст'}, status_code=400)
        
        # Проверка расширения
        if not file.filename.lower().endswith(('.csv', '.xlsx', '.xls')):
            return JSONResponse({'error': 'Поддерживаются только CSV и Excel файлы'}, status_code=400)
        
        print(f"📄 Размер файла: {len(file_content)} байт")
        
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

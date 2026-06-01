from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from gigachat import GigaChat
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
from reportlab.lib.units import mm

load_dotenv()

app = FastAPI(title="CashFlow - AI Financial Assistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ БАЗА ДАННЫХ ДЛЯ ИСТОРИИ ============
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
            category TEXT PRIMARY KEY,
            limit_amount REAL,
            month TEXT
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
    
    tips = ""
    if categories and total_expense > 0:
        top_expenses = sorted(expense_details, key=lambda x: x['amount'], reverse=True)[:3]
        top_with_desc = [(d['description'], d['amount']) for d in top_expenses]
        tips = get_savings_tips(categories, total_expense, top_with_desc)
    
    predicted_total, predicted_change, _ = predict_next_month(categories, total_expense, days_count)
    
    # Прогноз на 3 месяца
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
    
    # Сравнение с прошлым месяцем
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
    
    # Бюджетные алерты
    budget_alerts = check_budget_alerts(categories)
    
    cash_gap_warning = None
    if cash_gaps:
        first_gap = cash_gaps[0]
        cash_gap_warning = f"⚠️ {first_gap['advice']}. Рекомендуется сократить расходы или увеличить доходы."
    elif net_profit < 0:
        cash_gap_warning = f"⚠️ Расходы превышают доходы на {abs(net_profit):.2f} ₽. Рекомендуется сократить расходы или увеличить доходы."
    
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

# ============ НОВЫЕ ЭНДПОИНТЫ ============

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

@app.get("/export-pdf")
async def export_pdf():
    global last_analysis_result
    if not last_analysis_result:
        return JSONResponse({'error': 'Нет данных для экспорта'}, status_code=400)
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []
    
    # Заголовок
    title_style = ParagraphStyle('CustomTitle', parent=styles['Title'], fontSize=24, textColor=colors.orange)
    elements.append(Paragraph("CashFlow - Финансовый отчёт", title_style))
    elements.append(Spacer(1, 20))
    
    # Основные показатели
    data = [
        ['Показатель', 'Сумма (₽)'],
        ['Доходы', f"{last_analysis_result['income']:.2f}"],
        ['Расходы', f"{last_analysis_result['expense']:.2f}"],
        ['Чистая прибыль', f"{last_analysis_result['net_profit']:.2f}"],
        ['Рентабельность', f"{last_analysis_result.get('profitability', 0)}%"],
        ['Средний чек', f"{last_analysis_result.get('avg_check', 0):.2f} ₽"]
    ]
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.orange),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))
    
    # Категории расходов
    if last_analysis_result.get('categories'):
        elements.append(Paragraph("Расходы по категориям", styles['Heading2']))
        elements.append(Spacer(1, 10))
        cat_data = [['Категория', 'Сумма (₽)']]
        for cat, amt in last_analysis_result['categories'].items():
            cat_data.append([cat, f"{amt:.2f}"])
        cat_table = Table(cat_data)
        cat_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.orange),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
        ]))
        elements.append(cat_table)
    
    doc.build(elements)
    buffer.seek(0)
    
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=cashflow_report.pdf"}
    )

@app.get("/export-excel")
async def export_excel():
    global last_analysis_result
    if not last_analysis_result:
        return JSONResponse({'error': 'Нет данных для экспорта'}, status_code=400)
    
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Основные показатели
        summary_df = pd.DataFrame([
            ['Доходы', last_analysis_result['income']],
            ['Расходы', last_analysis_result['expense']],
            ['Чистая прибыль', last_analysis_result['net_profit']],
            ['Рентабельность', f"{last_analysis_result.get('profitability', 0)}%"],
            ['Средний чек', last_analysis_result.get('avg_check', 0)]
        ], columns=['Показатель', 'Значение'])
        summary_df.to_excel(writer, sheet_name='Основное', index=False)
        
        # Категории
        if last_analysis_result.get('categories'):
            cat_df = pd.DataFrame([
                [cat, amt] for cat, amt in last_analysis_result['categories'].items()
            ], columns=['Категория', 'Сумма'])
            cat_df.to_excel(writer, sheet_name='Категории', index=False)
        
        # Прогноз
        if last_analysis_result.get('forecast_3months'):
            forecast_df = pd.DataFrame(last_analysis_result['forecast_3months'])
            forecast_df.to_excel(writer, sheet_name='Прогноз', index=False)
    
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=cashflow_report.xlsx"}
    )

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

# HTML КОД (полный - я сокращу для читаемости, но он такой же как был с новыми кнопками)
# Добавляем новые кнопки в интерфейс

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
        /* Стили такие же как были, добавляем новые */
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
            --info: #3b82f6;
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
        /* Остальные стили такие же как в предыдущей версии */
        .card {
            background: var(--card-bg);
            backdrop-filter: var(--backdrop-blur);
            border-radius: 28px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: var(--card-shadow);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }
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
            font-size: 0.9rem;
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
        .alert-warning {
            background: rgba(239, 68, 68, 0.2);
            border-left: 4px solid #ef4444;
            padding: 12px;
            margin: 10px 0;
            border-radius: 12px;
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
    
    <div id="loading" style="display:none;text-align:center;padding:2rem;"><div class="spinner"></div><p>Анализирую выписку с помощью ИИ...</p></div>
    
    <div id="resultContainer" style="display:none;">
        <div class="card" id="suggestionCard">
            <h3><i class="fas fa-robot"></i> Анализ выполнен!</h3>
            <div id="budgetAlerts"></div>
            <div id="suggestionButtons" class="suggestion-buttons"></div>
        </div>
        
        <!-- НОВЫЙ БЛОК БЮДЖЕТА -->
        <div class="card" id="budgetBlock" style="display:none;">
            <h3><i class="fas fa-chart-line"></i> Планирование бюджета</h3>
            <div class="budget-input">
                <select id="budgetCategory"></select>
                <input type="number" id="budgetLimit" placeholder="Лимит в ₽">
                <button class="btn" onclick="setBudget()">Установить лимит</button>
            </div>
            <div id="budgetList"></div>
        </div>
        
        <!-- НОВЫЙ БЛОК ИСТОРИИ -->
        <div class="card" id="historyBlock" style="display:none;">
            <h3><i class="fas fa-history"></i> История анализов</h3>
            <div id="historyList"></div>
        </div>
        
        <div id="fullReport" class="card" style="display:none;"><div id="reportContent"></div></div>
        <div id="forecastBlock" class="card" style="display:none;"><div id="forecastContent"></div></div>
        <div id="tipsBlock" class="card" style="display:none;"><div id="tipsContent"></div></div>
        <div id="categoriesBlock" class="card" style="display:none;"><div id="categoriesContent"></div><canvas id="expenseChart" style="max-width:300px; margin:1rem auto;"></canvas></div>
        <div id="trendBlock" class="card" style="display:none;"><canvas id="trendChart"></canvas></div>
        <div id="seasonalityBlock" class="card" style="display:none;"><div id="seasonalityContent"></div></div>
        <div id="costBlock" class="card" style="display:none;"><div id="costContent"></div></div>
        <div id="chatBlock" class="card" style="display:none;">
            <h3>Чат с ИИ</h3>
            <div class="chat-messages" id="chatMessages"><div>Задайте вопрос о финансах</div></div>
            <div class="chat-input"><input type="text" id="questionInput" placeholder="Например: на чём мне сэкономить?"><button class="btn" onclick="askQuestion()">Отправить</button></div>
        </div>
    </div>
</div>

<script>
let analysisData = null;
let expenseChart = null;
let trendChart = null;

function uploadFile() {
    const file = document.getElementById('fileInput').files[0];
    if (!file) return;
    
    const formData = new FormData();
    formData.append('file', file);
    
    document.getElementById('loading').style.display = 'block';
    
    fetch('/upload', { method: 'POST', body: formData })
        .then(res => res.json())
        .then(data => {
            analysisData = data;
            showSmartSuggestions(data);
            showBudgetAlerts(data.budget_alerts);
            document.getElementById('resultContainer').style.display = 'block';
            document.getElementById('loading').style.display = 'none';
            loadBudgets();
            loadHistory();
        })
        .catch(err => {
            alert('Ошибка: ' + err.message);
            document.getElementById('loading').style.display = 'none';
        });
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
    const buttons = [
        { text: '📈 Полный отчёт', func: 'showFullReport' },
        { text: '📊 Сравнение', func: 'showComparison' },
        { text: '⚠️ Прогноз', func: 'showCashGapForecast' },
        { text: '💡 Советы', func: 'showTips' },
        { text: '📂 Категории', func: 'showCategories' },
        { text: '📈 Динамика', func: 'showTrend' },
        { text: '📅 Сезонность', func: 'showSeasonality' },
        { text: '💰 Себестоимость', func: 'showCost' },
        { text: '👥 Клиенты', func: 'showClientAnalysis' },
        { text: '💰 Бюджет', func: 'showBudget' },
        { text: '📜 История', func: 'showHistory' },
        { text: '📄 Экспорт PDF', func: 'exportPDF' },
        { text: '📊 Экспорт Excel', func: 'exportExcel' },
        { text: '💬 Чат', func: 'showChat' }
    ];
    
    const container = document.getElementById('suggestionButtons');
    container.innerHTML = buttons.map(btn => 
        `<button class="suggestion-btn" onclick="${btn.func}()">${btn.text}</button>`
    ).join('');
}

function showFullReport() {
    const d = analysisData;
    document.getElementById('reportContent').innerHTML = `
        <h3>📊 Полный отчёт</h3>
        <div class="result-stats">
            <div class="stat-card income"><div class="value">${d.income.toFixed(2)} ₽</div><div>💰 Доходы</div></div>
            <div class="stat-card expense"><div class="value">${d.expense.toFixed(2)} ₽</div><div>💸 Расходы</div></div>
            <div class="stat-card"><div class="value" style="color: ${d.net_profit >= 0 ? '#f97316' : '#ef4444'}">${d.net_profit >= 0 ? '+' : ''}${d.net_profit.toFixed(2)} ₽</div><div>✅ Чистая прибыль</div></div>
        </div>
        <div class="result-stats">
            <div class="stat-card"><div class="value">${d.profitability}%</div><div>📈 Рентабельность</div></div>
            <div class="stat-card"><div class="value">${d.avg_check.toFixed(2)} ₽</div><div>💰 Средний чек</div></div>
        </div>
        ${d.cash_gap_warning ? `<div class="info">⚠️ ${d.cash_gap_warning}</div>` : ''}
    `;
    showBlock('fullReport');
}

async function showComparison() {
    const res = await fetch('/monthly-comparison');
    const data = await res.json();
    const comp = data.comparison;
    if (!comp.has_data) {
        document.getElementById('reportContent').innerHTML = '<p>Нет данных для сравнения</p>';
        showBlock('fullReport');
        return;
    }
    document.getElementById('reportContent').innerHTML = `
        <h3>📊 Сравнение с прошлым месяцем</h3>
        <div class="result-stats">
            <div class="stat-card"><div>💰 Доходы</div><div class="value">${comp.income.current.toFixed(2)} ₽</div><div>было: ${comp.income.previous.toFixed(2)} ₽</div><div style="color: ${comp.income.change >= 0 ? '#10b981' : '#ef4444'}">${comp.income.change >= 0 ? '+' : ''}${comp.income.change}%</div></div>
            <div class="stat-card"><div>💸 Расходы</div><div class="value">${comp.expense.current.toFixed(2)} ₽</div><div>было: ${comp.expense.previous.toFixed(2)} ₽</div><div style="color: ${comp.expense.change <= 0 ? '#10b981' : '#ef4444'}">${comp.expense.change >= 0 ? '+' : ''}${comp.expense.change}%</div></div>
            <div class="stat-card"><div>✅ Прибыль</div><div class="value">${comp.profit.current.toFixed(2)} ₽</div><div>было: ${comp.profit.previous.toFixed(2)} ₽</div><div style="color: ${comp.profit.change >= 0 ? '#10b981' : '#ef4444'}">${comp.profit.change >= 0 ? '+' : ''}${comp.profit.change}%</div></div>
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
    let html = '<h3>⚠️ Прогноз на 3 месяца</h3>';
    for (const m of data.forecast) {
        html += `<div class="forecast-card risk-${m.risk_level}">
            <strong>📅 Месяц ${m.month}</strong>
            <div>💰 Доходы: ${m.income.toFixed(2)} ₽</div>
            <div>💸 Расходы: ${m.expense.toFixed(2)} ₽</div>
            <div>✅ Прибыль: ${m.profit.toFixed(2)} ₽</div>
            <div class="info">${m.risk_text} уровень риска</div>
        </div>`;
    }
    document.getElementById('forecastContent').innerHTML = html;
    showBlock('forecastBlock');
}

function showTips() {
    const d = analysisData;
    if (d.tips) {
        const items = d.tips.split('•').filter(i => i.trim());
        document.getElementById('tipsContent').innerHTML = `<ul>${items.map(i => `<li>• ${i.trim()}</li>`).join('')}</ul>`;
    } else {
        document.getElementById('tipsContent').innerHTML = '<p>Нет советов</p>';
    }
    showBlock('tipsBlock');
}

function showCategories() {
    const d = analysisData;
    if (d.categories) {
        let html = '<h3>📊 Расходы по категориям</h3><table style="width:100%"><tr><th>Категория</th><th>Сумма</th></tr>';
        for (const [cat, amt] of Object.entries(d.categories)) {
            html += `<tr><td>${cat}</td><td>${amt.toFixed(2)} ₽</td></tr>`;
        }
        html += '</table>';
        document.getElementById('categoriesContent').innerHTML = html;
        
        const ctx = document.getElementById('expenseChart').getContext('2d');
        if (expenseChart) expenseChart.destroy();
        expenseChart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: Object.keys(d.categories),
                datasets: [{ data: Object.values(d.categories), backgroundColor: ['#ea580c','#f97316','#c2410c','#fdba74','#9a3412','#7c2d12'] }]
            }
        });
        showBlock('categoriesBlock');
    }
}

function showTrend() {
    const d = analysisData;
    const ctx = document.getElementById('trendChart').getContext('2d');
    if (trendChart) trendChart.destroy();
    trendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['Неделя 1', 'Неделя 2', 'Неделя 3', 'Неделя 4'],
            datasets: [
                { label: 'Доходы', data: [d.income*0.6, d.income*0.8, d.income*0.9, d.income], borderColor: '#f97316' },
                { label: 'Расходы', data: [d.expense*0.7, d.expense*0.85, d.expense*0.95, d.expense], borderColor: '#ef4444' }
            ]
        }
    });
    showBlock('trendBlock');
}

function showSeasonality() {
    const s = analysisData?.seasonality || {};
    if (!s.has_data) {
        document.getElementById('seasonalityContent').innerHTML = '<p>Нет данных</p>';
        showBlock('seasonalityBlock');
        return;
    }
    let html = '<h3>📅 Сезонность</h3><div style="display:flex;gap:10px;flex-wrap:wrap;">';
    for (const [month, amt] of Object.entries(s.expense_by_month)) {
        if (amt > 0) html += `<div style="text-align:center"><div>Месяц ${month}</div><div style="color:#f97316">${amt.toFixed(0)} ₽</div></div>`;
    }
    html += '</div>';
    document.getElementById('seasonalityContent').innerHTML = html;
    showBlock('seasonalityBlock');
}

function showCost() {
    document.getElementById('costContent').innerHTML = `
        <h3>💰 Расчёт себестоимости</h3>
        <div class="cost-input-grid">
            <input type="text" id="productName" placeholder="Название товара">
            <input type="number" id="materialCost" placeholder="Сырьё на 1 ед. (₽)">
            <input type="number" id="timeMinutes" placeholder="Время на 1 ед. (мин)">
            <input type="number" id="quantityMonth" placeholder="Количество в месяц">
            <button class="btn" onclick="calculateCost()">Рассчитать</button>
        </div>
        <div id="costResult"></div>
    `;
    showBlock('costBlock');
}

function showClientAnalysis() {
    const clients = analysisData?.client_analysis || {};
    if (Object.keys(clients).length === 0) {
        document.getElementById('categoriesContent').innerHTML = '<p>Нет данных</p>';
        showBlock('categoriesBlock');
        return;
    }
    let html = '<h3>👥 Анализ клиентов</h3><table style="width:100%"><tr><th>Источник</th><th>Сумма</th></tr>';
    for (const [source, amount] of Object.entries(clients)) {
        html += `<tr><td>${source.substring(0, 40)}</td><td>${amount.toFixed(2)} ₽</td></tr>`;
    }
    html += '</table>';
    document.getElementById('categoriesContent').innerHTML = html;
    showBlock('categoriesBlock');
}

async function showBudget() {
    await loadBudgets();
    showBlock('budgetBlock');
}

async function loadBudgets() {
    const res = await fetch('/get-budgets');
    const budgets = await res.json();
    
    // Заполняем select категориями
    const select = document.getElementById('budgetCategory');
    if (analysisData && analysisData.categories) {
        select.innerHTML = Object.keys(analysisData.categories).map(cat => `<option value="${cat}">${cat}</option>`).join('');
    }
    
    // Показываем установленные лимиты
    const listDiv = document.getElementById('budgetList');
    if (Object.keys(budgets).length === 0) {
        listDiv.innerHTML = '<p>Лимиты не установлены</p>';
    } else {
        listDiv.innerHTML = '<h4>Текущие лимиты:</h4>' + 
            Object.entries(budgets).map(([cat, limit]) => 
                `<div class="info">${cat}: ${limit} ₽</div>`
            ).join('');
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
    // Перезагружаем анализ для обновления алертов
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
    showSmartSuggestions(data);
    showBudgetAlerts(data.budget_alerts);
    document.getElementById('resultContainer').style.display = 'block';
    alert('Загружен анализ от ' + data.analysis_date);
}

function exportPDF() {
    window.open('/export-pdf', '_blank');
}

function exportExcel() {
    window.open('/export-excel', '_blank');
}

function showChat() {
    showBlock('chatBlock');
}

function downloadTemplate() {
    window.location.href = '/download-template';
}

async function askQuestion() {
    const q = document.getElementById('questionInput').value;
    if (!q) return;
    const chatDiv = document.getElementById('chatMessages');
    chatDiv.innerHTML += `<div class="chat-message-user"><span>${q}</span></div>`;
    document.getElementById('questionInput').value = '';
    chatDiv.innerHTML += `<div class="typing">ИИ печатает...</div>`;
    const res = await fetch('/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q })
    });
    const data = await res.json();
    document.querySelector('.typing')?.remove();
    chatDiv.innerHTML += `<div class="chat-message-bot"><span>${data.answer}</span></div>`;
}

function calculateCost() {
    const name = document.getElementById('productName')?.value;
    const mat = parseFloat(document.getElementById('materialCost')?.value);
    const time = parseInt(document.getElementById('timeMinutes')?.value);
    const qty = parseInt(document.getElementById('quantityMonth')?.value);
    if (!name || isNaN(mat) || isNaN(time) || isNaN(qty)) {
        alert('Заполните все поля');
        return;
    }
    const totalExp = analysisData ? analysisData.expense : 0;
    const labor = (300 / 60) * time;
    const varTotal = mat * qty + labor * qty;
    const full = varTotal + totalExp;
    const cost = full / qty;
    const price = cost * 1.5;
    document.getElementById('costResult').innerHTML = `
        <div class="info">
            <strong>Результаты для ${name}:</strong><br>
            Себестоимость: ${cost.toFixed(2)} ₽<br>
            Рекомендуемая цена: ${price.toFixed(2)} ₽
        </div>
    `;
}

function showBlock(id) {
    const blocks = ['fullReport', 'forecastBlock', 'tipsBlock', 'categoriesBlock', 'trendBlock', 'seasonalityBlock', 'costBlock', 'chatBlock', 'budgetBlock', 'historyBlock'];
    blocks.forEach(b => {
        const el = document.getElementById(b);
        if (el) el.style.display = 'none';
    });
    document.getElementById(id).style.display = 'block';
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Инициализация
document.getElementById('fileInput').onchange = () => {
    const btn = document.getElementById('analyzeBtn');
    btn.disabled = false;
    document.getElementById('fileName').innerHTML = 'Выбран: ' + document.getElementById('fileInput').files[0].name;
    document.getElementById('fileName').style.display = 'block';
};
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

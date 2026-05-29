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
    months_data = {}  # Для хранения данных по месяцам
    
    if date_col:
        try:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce', dayfirst=True)
            date_min = df[date_col].min()
            date_max = df[date_col].max()
            if pd.notna(date_min) and pd.notna(date_max):
                days_count = (date_max - date_min).days + 1
                print(f"📅 Найдена колонка дат: {date_col}, период: {date_min.date()} - {date_max.date()}")
            
            # Группировка по месяцам для сравнения
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
                expense_details.append({'description': desc, 'amount': amt, 'date': row.get(date_col, datetime.now()) if date_col else datetime.now()})
    
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
        expense_df = pd.DataFrame(expense_details).head(20)
        expense_df['category'] = expense_df['description'].apply(ai_categorize)
        for cat, amt in expense_df.groupby('category')['amount'].sum().items():
            categories[category_names.get(cat, cat)] = float(amt)
        print(f"📂 Категории: {list(categories.keys())}")
    
    tips = ""
    if categories and total_expense > 0:
        top_expenses = sorted(expense_details, key=lambda x: x['amount'], reverse=True)[:3]
        top_with_desc = [(d['description'], d['amount']) for d in top_expenses]
        tips = get_savings_tips(categories, total_expense, top_with_desc)
    
    # ПРОГНОЗ НА 3 МЕСЯЦА
    forecast_3months = []
    cash_gaps = []
    
    if len(months_data) >= 1:
        # Сортируем месяцы по дате
        sorted_months = sorted(months_data.keys())
        monthly_expenses = [months_data[m]['expense'] for m in sorted_months]
        
        # Простой прогноз на основе среднего и тренда
        if len(monthly_expenses) >= 3:
            # Линейный тренд
            trend = (monthly_expenses[-1] - monthly_expenses[0]) / len(monthly_expenses)
        else:
            trend = 0
        
        avg_expense = sum(monthly_expenses) / len(monthly_expenses)
        last_expense = monthly_expenses[-1] if monthly_expenses else total_expense
        
        # Прогноз на 3 месяца
        for i in range(1, 4):
            predicted_expense = last_expense + (trend * i)
            if predicted_expense <= 0:
                predicted_expense = avg_expense
            
            # Прогноз доходов
            if len(monthly_expenses) >= 2:
                income_trend = (months_data[sorted_months[-1]]['income'] - months_data[sorted_months[0]]['income']) / len(monthly_expenses)
                predicted_income = months_data[sorted_months[-1]]['income'] + (income_trend * i)
            else:
                predicted_income = total_income
            
            predicted_profit = predicted_income - predicted_expense
            
            # Определяем уровень риска
            risk_level = "low"
            risk_text = "🟢 Низкий"
            if predicted_profit < 0:
                risk_level = "critical"
                risk_text = "🔴 Критический"
                cash_gaps.append({
                    'month': i,
                    'shortage': abs(predicted_profit),
                    'advice': f"Ожидается нехватка {abs(predicted_profit):.2f} ₽. Рекомендуется сократить расходы или найти дополнительный доход."
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
    
    # СРАВНЕНИЕ С ПРОШЛЫМ МЕСЯЦЕМ (улучшенное)
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
        
        # Создаём помесячную историю для графика
        for month in sorted_months[-6:]:  # последние 6 месяцев
            monthly_comparison[month] = months_data[month]
    
    # Сравнение с прошлым месяцем (старый формат для совместимости)
    old_comparison = {'has_data': comparison['has_data']}
    if comparison['has_data']:
        old_comparison = {
            'has_data': True,
            'income_change': comparison['income']['change'],
            'expense_change': comparison['expense']['change'],
            'profit_change': comparison['profit']['change'],
            'current_income': comparison['income']['current'],
            'last_income': comparison['income']['previous'],
            'current_expense': comparison['expense']['current'],
            'last_expense': comparison['expense']['previous'],
            'current_profit': comparison['profit']['current'],
            'last_profit': comparison['profit']['previous'],
            'current_month': comparison['current_month'],
            'last_month': comparison['previous_month']
        }
    
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
    
    # Прогноз кассового разрыва (общий)
    cash_gap_warning = None
    if cash_gaps:
        first_gap = cash_gaps[0]
        cash_gap_warning = f"⚠️ {first_gap['advice']}"
    elif net_profit < 0:
        cash_gap_warning = f"⚠️ Расходы превышают доходы на {abs(net_profit):.2f} ₽. Рекомендуется сократить расходы или увеличить доходы."
    
    last_analysis_result = {
        'income': float(total_income),
        'expense': float(total_expense),
        'net_profit': float(net_profit),
        'categories': categories,
        'rows_count': len(df),
        'months_data': months_data,
        'forecast_3months': forecast_3months,
        'cash_gaps': cash_gaps
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
        'predicted_total': forecast_3months[0]['expense'] if forecast_3months else None,
        'predicted_change': ((forecast_3months[0]['expense'] - total_expense) / total_expense * 100) if forecast_3months and total_expense > 0 else None,
        'seasonality': seasonality,
        'profitability': round(profitability, 1),
        'avg_check': round(avg_check, 2),
        'client_analysis': client_analysis,
        'cash_gap_warning': cash_gap_warning,
        'comparison': old_comparison,
        'detailed_comparison': comparison,
        'monthly_comparison': monthly_comparison,
        'forecast_3months': forecast_3months,
        'insights': []
    }

@app.get("/monthly-comparison")
async def get_monthly_comparison():
    """Возвращает сравнение по месяцам"""
    global last_analysis_result
    if not last_analysis_result:
        return JSONResponse({'error': 'Нет данных. Сначала загрузите выписку.'}, status_code=400)
    
    return JSONResponse({
        'comparison': last_analysis_result.get('detailed_comparison', {}),
        'monthly_data': last_analysis_result.get('monthly_comparison', {}),
        'forecast': last_analysis_result.get('forecast_3months', [])
    })

@app.get("/cash-gap-forecast")
async def get_cash_gap_forecast():
    """Возвращает прогноз кассовых разрывов на 3 месяца"""
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

# HTML код (упрощённая версия - я дам отдельно, так как он очень длинный)
# ПОЛНЫЙ HTML КОД НИЖЕ (скопируйте его полностью)

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
            font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 100%);
            min-height: 100vh;
            padding: 20px;
            color: #fff;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .card {
            background: rgba(17, 17, 17, 0.85);
            backdrop-filter: blur(10px);
            border-radius: 24px;
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
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 20px;
        }
        .stat-card {
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
        .loading {
            text-align: center;
            padding: 40px;
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
        .info {
            background: rgba(234, 88, 12, 0.15);
            padding: 12px;
            border-radius: 12px;
            margin-top: 16px;
        }
        .forecast-card {
            background: rgba(0,0,0,0.5);
            padding: 16px;
            border-radius: 16px;
            margin-bottom: 12px;
        }
        .risk-critical { border-left: 4px solid #ef4444; }
        .risk-medium { border-left: 4px solid #f59e0b; }
        .risk-low { border-left: 4px solid #10b981; }
        .comparison-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 16px;
            margin-bottom: 20px;
        }
        .comparison-item {
            text-align: center;
            padding: 12px;
            background: rgba(0,0,0,0.3);
            border-radius: 12px;
        }
        .change-positive { color: #10b981; }
        .change-negative { color: #ef4444; }
        .chat-messages {
            height: 300px;
            overflow-y: auto;
            border: 1px solid rgba(234, 88, 12, 0.3);
            border-radius: 16px;
            padding: 16px;
            margin-bottom: 16px;
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
        }
        .chat-input {
            display: flex;
            gap: 10px;
        }
        .chat-input input {
            flex: 1;
            padding: 12px;
            border: 1px solid rgba(234, 88, 12, 0.3);
            border-radius: 40px;
            background: rgba(0,0,0,0.5);
            color: white;
        }
        @media (max-width: 768px) {
            body { padding: 10px; }
            .result-stats { grid-template-columns: 1fr; }
            .comparison-grid { grid-template-columns: 1fr; }
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
            <button class="btn" onclick="downloadTemplate()" style="background: #2a2a2a;">📥 Шаблон CSV</button>
        </div>
    </div>
    
    <div id="loading" style="display: none;">
        <div class="card">
            <div class="loading">
                <div class="spinner"></div>
                <p>Анализирую выписку с помощью ИИ...</p>
            </div>
        </div>
    </div>
    
    <div id="resultContainer" style="display: none;">
        <div class="card" id="insightsCard">
            <h3>🤖 Анализ выполнен!</h3>
            <div id="suggestionButtons" class="suggestion-buttons"></div>
        </div>
        <div id="fullReport" class="card" style="display: none;"></div>
        <div id="comparisonBlock" class="card" style="display: none;"></div>
        <div id="forecastBlock" class="card" style="display: none;"></div>
        <div id="tipsBlock" class="card" style="display: none;"></div>
        <div id="categoriesBlock" class="card" style="display: none;"></div>
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

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
let analysisData = null;
let expenseChart = null;

const fileInput = document.getElementById('fileInput');
const analyzeBtn = document.getElementById('analyzeBtn');

function handleFileSelect() {
    if (fileInput.files.length) {
        const file = fileInput.files[0];
        document.getElementById('fileName').textContent = `📄 Выбран файл: ${file.name}`;
        document.getElementById('fileName').style.display = 'block';
        analyzeBtn.disabled = false;
    }
}

fileInput.onchange = handleFileSelect;

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
        alert('Ошибка при загрузке файла: ' + error.message);
    } finally {
        document.getElementById('loading').style.display = 'none';
    }
}

function showSmartSuggestions(data) {
    const buttons = [
        { text: '📈 Полный отчёт', func: showFullReport },
        { text: '📊 Сравнение с прошлым месяцем', func: showComparison },
        { text: '⚠️ Прогноз кассовых разрывов', func: showForecast },
        { text: '💡 Советы', func: showTips },
        { text: '📂 Категории', func: showCategories },
        { text: '💬 Чат с ИИ', func: showChat }
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
    hideAllBlocks();
    document.getElementById('fullReport').style.display = 'block';
}

async function showComparison() {
    try {
        const response = await fetch('/monthly-comparison');
        const data = await response.json();
        const comp = data.comparison;
        
        if (!comp.has_data) {
            document.getElementById('comparisonBlock').innerHTML = '<p>❌ Нет данных для сравнения. Загрузите файл с данными за несколько месяцев.</p>';
        } else {
            const incomeClass = comp.income.change >= 0 ? 'change-positive' : 'change-negative';
            const incomeSign = comp.income.change >= 0 ? '+' : '';
            const expenseClass = comp.expense.change >= 0 ? 'change-negative' : 'change-positive';
            const expenseSign = comp.expense.change >= 0 ? '+' : '';
            const profitClass = comp.profit.change >= 0 ? 'change-positive' : 'change-negative';
            const profitSign = comp.profit.change >= 0 ? '+' : '';
            
            document.getElementById('comparisonBlock').innerHTML = `
                <h3>📊 Сравнение с прошлым месяцем</h3>
                <div class="comparison-grid">
                    <div class="comparison-item">
                        <div>💰 Доходы</div>
                        <div class="value" style="font-size: 1.3rem;">${comp.income.current.toFixed(2)} ₽</div>
                        <div>было: ${comp.income.previous.toFixed(2)} ₽</div>
                        <div class="${incomeClass}">${incomeSign}${comp.income.change}% (${incomeSign}${comp.income.change_abs.toFixed(2)} ₽)</div>
                    </div>
                    <div class="comparison-item">
                        <div>💸 Расходы</div>
                        <div class="value" style="font-size: 1.3rem;">${comp.expense.current.toFixed(2)} ₽</div>
                        <div>было: ${comp.expense.previous.toFixed(2)} ₽</div>
                        <div class="${expenseClass}">${expenseSign}${comp.expense.change}% (${expenseSign}${comp.expense.change_abs.toFixed(2)} ₽)</div>
                    </div>
                    <div class="comparison-item">
                        <div>✅ Прибыль</div>
                        <div class="value" style="font-size: 1.3rem;">${comp.profit.current.toFixed(2)} ₽</div>
                        <div>было: ${comp.profit.previous.toFixed(2)} ₽</div>
                        <div class="${profitClass}">${profitSign}${comp.profit.change}% (${profitSign}${comp.profit.change_abs.toFixed(2)} ₽)</div>
                    </div>
                </div>
                <div class="info">
                    📅 ${comp.previous_month} → ${comp.current_month}
                </div>
            `;
        }
        hideAllBlocks();
        document.getElementById('comparisonBlock').style.display = 'block';
    } catch (error) {
        document.getElementById('comparisonBlock').innerHTML = '<p>❌ Ошибка загрузки данных</p>';
        hideAllBlocks();
        document.getElementById('comparisonBlock').style.display = 'block';
    }
}

async function showForecast() {
    try {
        const response = await fetch('/cash-gap-forecast');
        const data = await response.json();
        const forecast = data.forecast;
        
        if (!forecast || forecast.length === 0) {
            document.getElementById('forecastBlock').innerHTML = '<p>❌ Нет данных для прогноза. Загрузите файл с данными за несколько месяцев.</p>';
        } else {
            let html = '<h3>⚠️ Прогноз кассовых разрывов на 3 месяца</h3>';
            
            for (const month of forecast) {
                let riskClass = '';
                if (month.risk_level === 'critical') riskClass = 'risk-critical';
                else if (month.risk_level === 'medium') riskClass = 'risk-medium';
                else riskClass = 'risk-low';
                
                const profitColor = month.profit >= 0 ? '#10b981' : '#ef4444';
                const profitSign = month.profit >= 0 ? '+' : '';
                
                html += `
                    <div class="forecast-card ${riskClass}">
                        <div style="font-weight: bold; margin-bottom: 8px;">📅 Месяц ${month.month}</div>
                        <div class="result-stats" style="margin-bottom: 0;">
                            <div class="stat-card" style="padding: 8px;"><div>💰 Доходы</div><div style="font-weight: bold;">${month.income.toFixed(2)} ₽</div></div>
                            <div class="stat-card" style="padding: 8px;"><div>💸 Расходы</div><div style="font-weight: bold;">${month.expense.toFixed(2)} ₽</div></div>
                            <div class="stat-card" style="padding: 8px;"><div>✅ Прибыль</div><div style="font-weight: bold; color: ${profitColor};">${profitSign}${month.profit.toFixed(2)} ₽</div></div>
                        </div>
                        <div class="info" style="margin-top: 8px;">${month.risk_text} уровень риска</div>
                    </div>
                `;
            }
            
            if (data.has_warning) {
                html += `<div class="info" style="background: rgba(239,68,68,0.2); color: #ef4444; margin-top: 16px;">
                    ⚠️ ВНИМАНИЕ! Обнаружены риски кассовых разрывов. Рекомендуется принять меры.
                </div>`;
            }
            
            document.getElementById('forecastBlock').innerHTML = html;
        }
        hideAllBlocks();
        document.getElementById('forecastBlock').style.display = 'block';
    } catch (error) {
        document.getElementById('forecastBlock').innerHTML = '<p>❌ Ошибка загрузки данных</p>';
        hideAllBlocks();
        document.getElementById('forecastBlock').style.display = 'block';
    }
}

function showTips() {
    const d = analysisData;
    if (d.tips) {
        const tipsList = d.tips.split('•').filter(t => t.trim());
        document.getElementById('tipsBlock').innerHTML = `
            <h3>💡 Советы по экономии</h3>
            <ul style="margin-left: 20px;">
                ${tipsList.map(t => `<li style="margin: 10px 0;">• ${t.trim()}</li>`).join('')}
            </ul>
        `;
    } else {
        document.getElementById('tipsBlock').innerHTML = '<p>❌ Нет советов</p>';
    }
    hideAllBlocks();
    document.getElementById('tipsBlock').style.display = 'block';
}

function showCategories() {
    const d = analysisData;
    if (d.categories && Object.keys(d.categories).length) {
        let table = '<h3>📊 Расходы по категориям</h3><table style="width: 100%; border-collapse: collapse;">';
        for (const [cat, amt] of Object.entries(d.categories)) {
            table += `<tr style="border-bottom: 1px solid rgba(234,88,12,0.3);"><td style="padding: 10px;">${cat}</td><td style="padding: 10px; text-align: right;">${amt.toFixed(2)} ₽</td></tr>`;
        }
        table += '</table>';
        document.getElementById('categoriesBlock').innerHTML = table;
        
        const ctx = document.createElement('canvas');
        ctx.id = 'expenseChart';
        document.getElementById('categoriesBlock').appendChild(ctx);
        
        if (expenseChart) expenseChart.destroy();
        expenseChart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: Object.keys(d.categories),
                datasets: [{ data: Object.values(d.categories), backgroundColor: ['#ea580c','#f97316','#c2410c','#fdba74','#9a3412','#7c2d12'] }]
            }
        });
    } else {
        document.getElementById('categoriesBlock').innerHTML = '<p>❌ Нет данных для категоризации</p>';
    }
    hideAllBlocks();
    document.getElementById('categoriesBlock').style.display = 'block';
}

function showChat() {
    hideAllBlocks();
    document.getElementById('chatBlock').style.display = 'block';
}

function hideAllBlocks() {
    const blocks = ['fullReport', 'comparisonBlock', 'forecastBlock', 'tipsBlock', 'categoriesBlock', 'chatBlock'];
    blocks.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = 'none';
    });
}

async function askQuestion() {
    const question = document.getElementById('questionInput').value.trim();
    if (!question) return;
    
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.innerHTML += `<div class="chat-message-user"><span>${escapeHtml(question)}</span></div>`;
    document.getElementById('questionInput').value = '';
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    try {
        const response = await fetch('/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: question })
        });
        const data = await response.json();
        chatMessages.innerHTML += `<div class="chat-message-bot"><span>${escapeHtml(data.answer)}</span></div>`;
        chatMessages.scrollTop = chatMessages.scrollHeight;
    } catch (error) {
        chatMessages.innerHTML += `<div class="chat-message-bot"><span>❌ Ошибка: ${error.message}</span></div>`;
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

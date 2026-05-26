from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from gigachat import GigaChat
from dotenv import load_dotenv
import pandas as pd
import os
import calendar
import tempfile
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
    
    predicted_categories = {}
    for cat, amount in expenses_by_category.items():
        avg_daily_cat = amount / days_count
        predicted_categories[cat] = avg_daily_cat * 30
    
    change_percent = ((predicted_monthly - total_expense) / total_expense) * 100 if total_expense > 0 else 0
    
    return predicted_monthly, change_percent, predicted_categories

def analyze_trends(categories, total_income, total_expense, net_profit, days_count):
    recommendations = []
    insights = []
    
    if net_profit < 0:
        recommendations.append("savings")
        insights.append(f"⚠️ Расходы превышают доходы на {abs(net_profit):.2f} ₽")
    elif net_profit < total_income * 0.1 and total_income > 0:
        recommendations.append("savings")
        insights.append(f"📉 Маржинальность ниже 10% ({net_profit/total_income*100:.1f}%)")
    
    if 'Прочее' in categories and total_expense > 0:
        other_percent = (categories['Прочее'] / total_expense) * 100
        if other_percent > 15:
            recommendations.append("categories")
            insights.append(f"🔍 Категория «Прочее» — {other_percent:.1f}% расходов")
    
    if 'Транспорт' in categories and total_expense > 0:
        transport = categories['Транспорт']
        if transport > total_expense * 0.15:
            recommendations.append("forecast")
            insights.append(f"🚗 Высокие транспортные расходы: {transport:.2f} ₽")
    
    recommendations.append("chat")
    recommendations.append("seasonality")
    
    if len(recommendations) < 2:
        recommendations.insert(0, "forecast")
    
    recommendations = list(dict.fromkeys(recommendations))
    
    return recommendations, insights

def analyze_seasonality(df, date_col):
    result = {
        'by_month': {},
        'by_weekday': {},
        'by_hour': {},
        'expense_by_month': {},
        'income_by_month': {},
        'has_data': False
    }
    
    if date_col not in df.columns or date_col is None:
        return result
    
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        
        if len(df) == 0:
            return result
        
        result['has_data'] = True
        
        incomes = []
        expenses = []
        for idx, row in df.iterrows():
            typ, amount = detect_income_expense(row)
            if typ == 'income' and amount > 0:
                incomes.append((row[date_col], amount))
            elif typ == 'expense' and amount > 0:
                expenses.append((row[date_col], amount))
        
        income_df = pd.DataFrame(incomes, columns=['date', 'amount']) if incomes else pd.DataFrame()
        expense_df = pd.DataFrame(expenses, columns=['date', 'amount']) if expenses else pd.DataFrame()
        
        if not expense_df.empty:
            expense_df['month'] = expense_df['date'].dt.month
            monthly_expense = expense_df.groupby('month')['amount'].sum().to_dict()
            result['expense_by_month'] = monthly_expense
            
            if monthly_expense:
                max_month = max(monthly_expense, key=monthly_expense.get)
                min_month = min(monthly_expense, key=monthly_expense.get)
                result['max_expense_month'] = {
                    'month': max_month,
                    'name': calendar.month_name[max_month],
                    'amount': monthly_expense[max_month]
                }
                result['min_expense_month'] = {
                    'month': min_month,
                    'name': calendar.month_name[min_month],
                    'amount': monthly_expense[min_month]
                }
        
        if not income_df.empty:
            income_df['month'] = income_df['date'].dt.month
            monthly_income = income_df.groupby('month')['amount'].sum().to_dict()
            result['income_by_month'] = monthly_income
        
        if not expense_df.empty:
            expense_df['weekday'] = expense_df['date'].dt.weekday
            weekday_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
            weekday_expense = expense_df.groupby('weekday')['amount'].sum().to_dict()
            result['by_weekday'] = {weekday_names[k]: v for k, v in weekday_expense.items()}
            
            if weekday_expense:
                max_weekday = max(weekday_expense, key=weekday_expense.get)
                result['max_weekday'] = {
                    'name': weekday_names[max_weekday],
                    'amount': weekday_expense[max_weekday]
                }
                min_weekday = min(weekday_expense, key=weekday_expense.get)
                result['min_weekday'] = {
                    'name': weekday_names[min_weekday],
                    'amount': weekday_expense[min_weekday]
                }
        
        if not expense_df.empty and expense_df['date'].dt.hour.nunique() > 1:
            expense_df['hour'] = expense_df['date'].dt.hour
            hourly_expense = expense_df.groupby('hour')['amount'].sum().to_dict()
            result['by_hour'] = hourly_expense
            
            if hourly_expense:
                max_hour = max(hourly_expense, key=hourly_expense.get)
                result['max_hour'] = {
                    'hour': max_hour,
                    'amount': hourly_expense[max_hour]
                }
    except Exception as e:
        print(f"Ошибка анализа сезонности: {e}")
    
    return result

def parse_file(file_content: bytes, filename: str):
    ext = filename.split('.')[-1].lower()
    df = None
    
    if ext == 'csv':
        encodings = ['cp1251', 'utf-8', 'windows-1251', 'latin1']
        for encoding in encodings:
            try:
                text = file_content.decode(encoding)
                from io import StringIO
                df = pd.read_csv(StringIO(text))
                print(f"✅ Прочитан CSV с кодировкой {encoding}")
                break
            except:
                continue
    
    elif ext in ['xlsx', 'xls', 'xlsm']:
        try:
            df = pd.read_excel(BytesIO(file_content), engine='openpyxl')
            print(f"✅ Прочитан Excel файл: {filename}")
            print(f"📊 Найдено {len(df)} строк")
        except Exception as e:
            raise Exception(f"Ошибка чтения Excel: {e}")
    
    elif ext == 'pdf':
        try:
            import pdfplumber
            
            all_tables = []
            pdf_file = io.BytesIO(file_content)
            
            with pdfplumber.open(pdf_file) as pdf:
                print(f"📄 PDF страниц: {len(pdf.pages)}")
                for page_num, page in enumerate(pdf.pages):
                    tables = page.extract_tables()
                    print(f"  Страница {page_num + 1}: найдено {len(tables)} таблиц")
                    for table in tables:
                        if table and len(table) > 1:
                            headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(table[0])]
                            rows = table[1:]
                            rows = [row for row in rows if any(cell for cell in row)]
                            if rows:
                                page_df = pd.DataFrame(rows, columns=headers)
                                all_tables.append(page_df)
            
            if all_tables:
                df = pd.concat(all_tables, ignore_index=True)
                print(f"✅ Прочитан PDF: {len(df)} строк, {len(all_tables)} таблиц")
            else:
                raise Exception("В PDF не найдено таблиц")
                
        except ImportError:
            raise Exception("Установите pdfplumber: pip install pdfplumber")
        except Exception as e:
            raise Exception(f"Ошибка чтения PDF: {e}")
    
    else:
        raise Exception(f"Неподдерживаемый формат: {ext}. Поддерживаются: CSV, Excel, PDF")
    
    if df is None or df.empty:
        raise Exception("Не удалось прочитать файл или файл пуст")
    
    df.columns = df.columns.str.lower().str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    
    rename_map = {
        'дата и время': 'date',
        'дата': 'date',
        'дата операции': 'date',
        'сумма операции': 'amount',
        'сумма в валюте': 'amount',
        'сумма': 'amount',
        'описание': 'description',
        'тип операции': 'type',
        'статус': 'status',
        'категория': 'category',
        'merchant': 'merchant'
    }
    
    for old_name, new_name in rename_map.items():
        if old_name in df.columns:
            df.rename(columns={old_name: new_name}, inplace=True)
            print(f"🔄 Переименована: {old_name} → {new_name}")
    
    df = df.loc[:, ~df.columns.duplicated()]
    
    return df

def detect_income_expense(row):
    for col in ['type', 'transactiontype', 'операция', 'тип']:
        if col in row and pd.notna(row[col]):
            type_value = str(row[col]).lower()
            
            expense_types = ['списание', 'оплата', 'покупка', 'перевод', 'комиссия', 
                           'снятие', 'оплата услуг', 'налог', 'аренда', 'реклама']
            for kw in expense_types:
                if kw in type_value:
                    for amount_col in ['amount', 'sum', 'total', 'сумма']:
                        if amount_col in row and pd.notna(row[amount_col]):
                            try:
                                return 'expense', abs(float(row[amount_col]))
                            except:
                                pass
                    return 'expense', 0
            
            income_types = ['пополнение', 'зачисление', 'поступление', 'возврат', 
                          'перевод от', 'поступление от', 'зачисление от']
            for kw in income_types:
                if kw in type_value:
                    for amount_col in ['amount', 'sum', 'total', 'сумма']:
                        if amount_col in row and pd.notna(row[amount_col]):
                            try:
                                return 'income', abs(float(row[amount_col]))
                            except:
                                pass
                    return 'income', 0
    
    if 'description' in row and pd.notna(row['description']):
        desc = str(row['description']).lower()
        
        income_keywords = ['пополнение', 'зачисление', 'поступление', 'возврат', 'перевод от']
        for kw in income_keywords:
            if kw in desc:
                for amount_col in ['amount', 'sum', 'total', 'сумма']:
                    if amount_col in row and pd.notna(row[amount_col]):
                        try:
                            return 'income', abs(float(row[amount_col]))
                        except:
                            pass
                return 'income', 0
        
        expense_keywords = ['списание', 'оплата', 'покупка', 'перевод', 'комиссия', 
                          'снятие', 'налог', 'аренда', 'реклама']
        for kw in expense_keywords:
            if kw in desc:
                for amount_col in ['amount', 'sum', 'total', 'сумма']:
                    if amount_col in row and pd.notna(row[amount_col]):
                        try:
                            return 'expense', abs(float(row[amount_col]))
                        except:
                            pass
                return 'expense', 0
    
    if 'amount' in row:
        try:
            val = float(row['amount'])
            if val < 0:
                return 'expense', abs(val)
            elif val > 0:
                return 'income', val
        except:
            pass
    
    return 'unknown', 0

def analyze_statement(file_content: bytes, filename: str):
    global last_analysis_result
    
    df = parse_file(file_content, filename)
    
    date_col = None
    for col in ['date', 'operationdate']:
        if col in df.columns:
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
        except:
            pass
    
    if 'status' in df.columns:
        def should_keep(row):
            type_val = str(row.get('type', '')).lower() if pd.notna(row.get('type')) else ''
            status_val = str(row.get('status', '')).lower() if pd.notna(row.get('status')) else ''
            if 'пополнение' in type_val:
                return status_val == '' or status_val == 'выполнен' or pd.isna(row.get('status'))
            if 'списание' in type_val:
                return status_val == 'выполнен'
            return status_val == 'выполнен'
        
        mask = df.apply(should_keep, axis=1)
        df = df[mask]
    
    incomes, expenses, expense_details = [], [], []
    
    for idx, row in df.iterrows():
        typ, amount = detect_income_expense(row)
        if typ == 'income' and amount > 0:
            incomes.append(amount)
        elif typ == 'expense' and amount > 0:
            expenses.append(amount)
            desc = str(row.get('merchant', row.get('description', row.get('comment', ''))))
            if desc and desc != 'nan' and len(desc) > 2:
                expense_details.append({'description': desc, 'amount': amount})
    
    total_income = sum(incomes)
    total_expense = sum(expenses)
    net_profit = total_income - total_expense
    
    categories = {}
    if expense_details:
        expense_df = pd.DataFrame(expense_details)
        expense_df = expense_df.head(20)
        expense_df['category'] = expense_df['description'].apply(ai_categorize)
        expenses_by_cat = expense_df.groupby('category')['amount'].sum()
        categories = {category_names.get(cat, cat): float(amount) for cat, amount in expenses_by_cat.items()}
    
    tips = ""
    if categories and total_expense > 0:
        top_expenses = sorted(expense_details, key=lambda x: x['amount'], reverse=True)[:5]
        top_with_desc = [(d['description'], d['amount']) for d in top_expenses]
        tips = get_savings_tips(categories, total_expense, top_with_desc)
    
    predicted_total = None
    predicted_change = None
    predicted_categories = {}
    if total_expense > 0 and days_count > 0:
        predicted_total, predicted_change, predicted_categories = predict_next_month(
            categories, total_expense, days_count
        )
    
    seasonality = {}
    if date_col:
        seasonality = analyze_seasonality(df, date_col)
    
    recommendations, insights = analyze_trends(
        categories, total_income, total_expense, net_profit, days_count
    )
    
    type_stats = {}
    if 'type' in df.columns:
        for t in df['type']:
            type_name = str(t).lower()
            if type_name and type_name != 'nan':
                type_stats[type_name] = type_stats.get(type_name, 0) + 1
    
    last_analysis_result = {
        'income': float(total_income),
        'expense': float(total_expense),
        'net_profit': float(net_profit),
        'categories': categories,
        'rows_count': len(df),
        'incomes_count': len(incomes),
        'expenses_count': len(expenses),
        'days_count': days_count,
        'predicted_total': float(predicted_total) if predicted_total else None,
        'predicted_change': float(predicted_change) if predicted_change else None
    }
    
    return {
        'income': float(total_income),
        'expense': float(total_expense),
        'net_profit': float(net_profit),
        'categories': categories,
        'tips': tips,
        'rows_count': len(df),
        'days_count': days_count,
        'predicted_total': float(predicted_total) if predicted_total else None,
        'predicted_change': float(predicted_change) if predicted_change else None,
        'predicted_categories': predicted_categories,
        'recommendations': recommendations,
        'insights': insights,
        'seasonality': seasonality,
        'columns': list(df.columns),
        'incomes_count': len(incomes),
        'expenses_count': len(expenses),
        'type_stats': type_stats
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
    
    if last_analysis_result.get('predicted_total'):
        context += f"\nПрогноз расходов на следующий месяц: {last_analysis_result['predicted_total']:.2f} ₽"
        if last_analysis_result.get('predicted_change'):
            context += f" (изменение: {last_analysis_result['predicted_change']:+.1f}%)"
    
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

# HTML страница с улучшениями
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
        
        @keyframes fadeInUp {
            from { opacity: 0; transform: translateY(30px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
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
        
        @keyframes confettiFall {
            0% { transform: translateY(0) rotate(0deg); opacity: 1; }
            100% { transform: translateY(100vh) rotate(720deg); opacity: 0; }
        }
        
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: var(--card-bg);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb {
            background: var(--accent);
            border-radius: 10px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: var(--primary-start);
        }
        
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
        .card:hover {
            transform: translateY(-4px);
            box-shadow: var(--hover-shadow);
        }
        
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
        
        .mobile-header { display: none; justify-content: space-between; align-items: center; margin-bottom: 1rem; background: rgba(0

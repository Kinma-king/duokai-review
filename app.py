from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import math

app = Flask(__name__, static_folder='app/static', template_folder='app/templates')
CORS(app)

# 加载完整A股列表
def load_all_stocks():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, 'app', 'data', 'all_stocks.json')
        with open(file_path, 'r', encoding='utf-8') as f:
            stocks = json.load(f)
            return {s['code']: s['name'] for s in stocks}
    except Exception as e:
        print(f"Error loading stock list: {e}")
        return {}

STOCK_MAP = load_all_stocks()
print(f"Loaded {len(STOCK_MAP)} stocks into memory")

# ========== 指标简介库 ==========
INDICATOR_INFO = {
    "MA": {
        "name": "移动平均线 Moving Average",
        "desc": "MA 是过去 N 个周期的收盘价平均值，是最基础的趋势跟踪指标。",
        "principle": "1. 短期均线上穿长期均线 → 金叉，看涨\n2. 短期均线下穿长期均线 → 死叉，看跌\n3. 价格在 MA 上方运行 → 多头趋势\n4. 价格在 MA 下方运行 → 空头趋势\n5. 均线斜率陡峭 → 趋势强劲\n6. 均线走平粘合 → 震荡整理",
        "periods": [5, 10, 20, 30, 60],
        "author": "统计学基本概念",
        "category": "趋势"
    },
    "MACD": {
        "name": "指数平滑异同移动平均线 MACD",
        "desc": "MACD 通过快慢均线的差值判断趋势强弱和转折点，由 Gerald Appel 发明。",
        "principle": "1. DIF 上穿 DEA → 金叉，买入信号\n2. DIF 下穿 DEA → 死叉，卖出信号\n3. MACD 柱由负转正 → 空转多\n4. MACD 柱由正转负 → 多转空\n5. 顶背离：股价新高、MACD 未新高 → 见顶\n6. 底背离：股价新低、MACD 未新低 → 见底",
        "author": "Gerald Appel, 1979",
        "category": "趋势"
    },
    "KDJ": {
        "name": "随机指标 KDJ",
        "desc": "KDJ 通过比较收盘价在近期价格区间的位置，判断超买超卖状态，由 George Lane 发明。",
        "principle": "1. K 值 > 80 → 超买区域，警惕回调\n2. K 值 < 20 → 超卖区域，关注反弹\n3. K 线上穿 D 线 → 金叉，买入\n4. K 线下穿 D 线 → 死叉，卖出\n5. J 值 > 100 → 严重超买\n6. J 值 < 0 → 严重超卖",
        "author": "George Lane, 1950s",
        "category": "摆动"
    },
    "RSI": {
        "name": "相对强弱指标 RSI",
        "desc": "RSI 衡量近期价格变动的速度和幅度，用于判断超买超卖和趋势强度，由 Welles Wilder 发明。",
        "principle": "1. RSI > 70 → 超买，可能回调\n2. RSI < 30 → 超卖，可能反弹\n3. RSI 在 50 上方 → 多头市场\n4. RSI 在 50 下方 → 空头市场\n5. RSI 顶背离 → 卖出信号\n6. RSI 底背离 → 买入信号",
        "author": "Welles Wilder, 1978",
        "category": "摆动"
    },
    "BOLL": {
        "name": "布林带 Bollinger Bands",
        "desc": "布林带由中轨(MA20)、上轨(+2σ)、下轨(-2σ)组成，反映价格波动区间，由 John Bollinger 发明。",
        "principle": "1. 价格触及上轨 → 超买，可能回调\n2. 价格触及下轨 → 超卖，可能反弹\n3. 带宽收窄 → 变盘在即，酝酿突破\n4. 带宽扩张 → 趋势加速\n5. 价格沿上轨运行 → 强势上涨\n6. 价格沿下轨运行 → 弱势下跌",
        "author": "John Bollinger, 1980s",
        "category": "通道"
    },
    "OBV": {
        "name": "能量潮 On-Balance Volume",
        "desc": "OBV 通过累计成交量（上涨日加、下跌日减）判断资金流向，由 Joseph Granville 发明。",
        "principle": "1. OBV 创新高、价格未创新高 → 量价背离，看跌\n2. OBV 创新低、价格未创新低 → 量价背离，看涨\n3. OBV 与价格同步上升 → 上升趋势确认\n4. OBV 与价格同步下降 → 下降趋势确认\n5. OBV 横盘时价格下跌 → 底部吸筹信号",
        "author": "Joseph Granville, 1963",
        "category": "量价"
    },
    "WR": {
        "name": "威廉指标 Williams %R",
        "desc": "WR 衡量收盘价在近期高低价区间的位置，与 KDJ 原理相似，由 Larry Williams 发明。",
        "principle": "1. WR < 20 → 超买，可能回调 (WR 数值越小越超买)\n2. WR > 80 → 超卖，可能反弹\n3. WR 多次触及 0~20 区域 → 顶部信号\n4. WR 多次触及 80~100 区域 → 底部信号\n5. 注意：WR 是反向指标，越低越看跌",
        "author": "Larry Williams, 1973",
        "category": "摆动"
    },
    "CCI": {
        "name": "商品通道指数 CCI",
        "desc": "CCI 衡量价格偏离统计平均的程度，对极端行情反应敏感，由 Donald Lambert 发明。",
        "principle": "1. CCI > 100 → 超买，强势上涨\n2. CCI < -100 → 超卖，弱势下跌\n3. CCI 从 +100 上方回落 → 卖出信号\n4. CCI 从 -100 下方回升 → 买入信号\n5. CCI 在 ±100 之间 → 盘整行情",
        "author": "Donald Lambert, 1980",
        "category": "摆动"
    },
    "ATR": {
        "name": "平均真实波幅 ATR",
        "desc": "ATR 衡量价格波动幅度（非方向），用于设定止损和仓位管理，由 Welles Wilder 发明。",
        "principle": "1. ATR 上升 → 波动加大，趋势可能启动\n2. ATR 下降 → 波动减小，可能盘整\n3. 止损可设为 2~3 倍 ATR\n4. ATR 极高 → 恐慌或狂热，可能反转\n5. 注意：ATR 不指示方向，只指示波动烈度",
        "author": "Welles Wilder, 1978",
        "category": "波动"
    },
    "BIAS": {
        "name": "乖离率 BIAS",
        "desc": "BIAS 衡量价格偏离移动平均线的程度，反映超买超卖和回归均值的概率。",
        "principle": "1. BIAS > 正阈值 → 严重超买，短期回调概率大\n2. BIAS < 负阈值 → 严重超卖，短期反弹概率大\n3. BIAS 由负转正 → 趋势转多\n4. BIAS 由正转负 → 趋势转空\n5. 不同股票有不同的乖离阈值，活跃股阈值高",
        "author": "技术分析通用概念",
        "category": "摆动"
    },
    "PSY": {
        "name": "心理线 PSY",
        "desc": "PSY 统计 N 日内上涨天数的占比，反映市场多空心理。",
        "principle": "1. PSY > 75 → 市场过热，警惕回调\n2. PSY < 25 → 市场过度悲观，关注反弹\n3. PSY 在 50 附近 → 多空均衡\n4. PSY 从高点回落 → 卖出信号\n5. PSY 从低点回升 → 买入信号",
        "author": "技术分析通用指标",
        "category": "心理"
    },
    "VR": {
        "name": "成交量比率 Volume Ratio",
        "desc": "VR 比较上涨日成交量与下跌日成交量的比率，判断资金流入流出。",
        "principle": "1. VR > 450 → 市场过热，顶部信号\n2. VR < 70 → 市场低迷，底部信号\n3. VR 在 150 左右 → 安全区域\n4. VR 上升、价格盘整 → 吸筹，后市看涨\n5. VR 下降、价格盘整 → 出货，后市看跌",
        "author": "技术分析通用指标",
        "category": "量价"
    },
    "VOL_MA": {
        "name": "成交量均线 Volume MA",
        "desc": "成交量移动平均线，用于判断量能变化趋势。",
        "principle": "1. 放量上涨 → 上涨趋势有力\n2. 缩量上涨 → 上攻乏力，警惕回落\n3. 放量下跌 → 恐慌抛售\n4. 缩量下跌 → 跌势减缓，可能见底\n5. 量比 > 1.5 → 当日活跃",
        "author": "技术分析通用指标",
        "category": "量价"
    }
}


# ========== 路由 ==========

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stock/<code>')
def stock_detail(code):
    return render_template('stock.html', code=code)

# 获取实时行情
@app.route('/api/quote/<code>')
def get_quote(code):
    try:
        url = f"https://qt.gtimg.cn/q={code}"
        if code.startswith('6'):
            url = f"https://qt.gtimg.cn/q=sh{code}"
        elif code.startswith('0') or code.startswith('3'):
            url = f"https://qt.gtimg.cn/q=sz{code}"
        
        response = requests.get(url, timeout=10)
        data = response.text
        
        # 字段: 3=最新价 4=昨收 5=今开 31=涨跌额 32=涨跌幅 33=最高 34=最低 36=成交量 37=成交额(万)
        import re
        try:
            data = data.encode('latin1').decode('gb2312')
        except:
            pass
        
        match = re.search(r'v_\w+="([^"]+)"', data)
        if match:
            fields = match.group(1).split('~')
            if len(fields) > 37:
                return jsonify({
                    'code': code,
                    'name': fields[1] if len(fields) > 1 else '',
                    'price': float(fields[3]) if len(fields) > 3 else 0,
                    'preClose': float(fields[4]) if len(fields) > 4 else 0,
                    'open': float(fields[5]) if len(fields) > 5 else 0,
                    'high': float(fields[33]) if len(fields) > 33 else 0,
                    'low': float(fields[34]) if len(fields) > 34 else 0,
                    'change': float(fields[31]) if len(fields) > 31 else 0,
                    'changePercent': float(fields[32]) if len(fields) > 32 else 0,
                    'volume': float(fields[36]) if len(fields) > 36 else 0,
                    'amount': float(fields[37]) if len(fields) > 37 else 0,
                })
        return jsonify({'error': 'No data'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 获取K线数据
@app.route('/api/kline/<code>')
def get_kline(code):
    try:
        period = request.args.get('period', 'day')
        count = request.args.get('count', 120, type=int)
        
        market = 'sh' if code.startswith('6') or code.startswith('689') else 'sz'
        
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        params = {'param': f"{market}{code},{period},,,{count},qfq"}
        
        response = requests.get(url, params=params, timeout=15)
        data = response.json()
        
        stock_key = f"{market}{code}"
        period_key = f"qfq{period}"
        
        if data.get('data') and data['data'].get(stock_key):
            stock_data = data['data'][stock_key]
            klines_raw = None
            if period_key in stock_data:
                klines_raw = stock_data[period_key]
            elif period in stock_data:
                klines_raw = stock_data[period]
            
            if klines_raw:
                klines = []
                for item in klines_raw:
                    klines.append({
                        'date': item[0],
                        'open': float(item[1]),
                        'close': float(item[2]),
                        'high': float(item[3]),
                        'low': float(item[4]),
                        'volume': float(item[5]),
                        'amount': 0, 'amplitude': 0,
                        'changePercent': 0, 'change': 0, 'turnover': 0,
                    })
                
                for i in range(len(klines)):
                    if i > 0:
                        prev = klines[i-1]['close']
                        cur = klines[i]['close']
                        klines[i]['change'] = cur - prev
                        klines[i]['changePercent'] = round((cur - prev) / prev * 100, 2)
                        klines[i]['amplitude'] = round((klines[i]['high'] - klines[i]['low']) / prev * 100, 2)
                
                klines = calculate_indicators(klines)
                return jsonify(klines)
        
        return jsonify([])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 计算全部技术指标
def calculate_indicators(data):
    df = pd.DataFrame(data)
    n = len(df)
    
    # --- MA ---
    for period in [5, 10, 20, 30, 60]:
        df[f'MA{period}'] = df['close'].rolling(window=period).mean()
    
    # --- MACD ---
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_HIST'] = df['MACD'] - df['MACD_SIGNAL']
    
    # --- KDJ ---
    low9 = df['low'].rolling(window=9, min_periods=9).min()
    high9 = df['high'].rolling(window=9, min_periods=9).max()
    rsv = (df['close'] - low9) / (high9 - low9) * 100
    df['K'] = rsv.ewm(com=2, adjust=False).mean()
    df['D'] = df['K'].ewm(com=2, adjust=False).mean()
    df['J'] = 3 * df['K'] - 2 * df['D']
    
    # --- RSI ---
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta.clip(upper=0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # --- BOLL ---
    df['BOLL_MID'] = df['close'].rolling(window=20).mean()
    df['BOLL_STD'] = df['close'].rolling(window=20).std()
    df['BOLL_UP'] = df['BOLL_MID'] + 2 * df['BOLL_STD']
    df['BOLL_DOWN'] = df['BOLL_MID'] - 2 * df['BOLL_STD']
    
    # --- OBV (能量潮) ---
    obv = [0]
    for i in range(1, n):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    
    # --- WR (威廉指标) ---
    high14 = df['high'].rolling(window=14).max()
    low14 = df['low'].rolling(window=14).min()
    df['WR'] = (high14 - df['close']) / (high14 - low14) * -100
    
    # --- CCI ---
    tp = (df['high'] + df['low'] + df['close']) / 3
    ma_tp = tp.rolling(window=20).mean()
    md = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df['CCI'] = (tp - ma_tp) / (0.015 * md)
    
    # --- ATR ---
    tr = pd.concat([
        df['high'] - df['low'],
        abs(df['high'] - df['close'].shift(1)),
        abs(df['low'] - df['close'].shift(1))
    ], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    
    # --- BIAS (乖离率) ---
    df['BIAS6'] = (df['close'] - df['close'].rolling(window=6).mean()) / df['close'].rolling(window=6).mean() * 100
    df['BIAS12'] = (df['close'] - df['close'].rolling(window=12).mean()) / df['close'].rolling(window=12).mean() * 100
    df['BIAS24'] = (df['close'] - df['close'].rolling(window=24).mean()) / df['close'].rolling(window=24).mean() * 100
    
    # --- PSY (心理线) ---
    up_days = (df['close'].diff() > 0).astype(int)
    df['PSY'] = up_days.rolling(window=12).sum() / 12 * 100
    
    # --- VR (成交量比率) ---
    vol_up = df['volume'].where(df['close'].diff() > 0, 0)
    vol_down = df['volume'].where(df['close'].diff() < 0, 0)
    vol_eq = df['volume'].where(df['close'].diff() == 0, 0)
    df['VR'] = (vol_up.rolling(26).sum() + vol_eq.rolling(26).sum() / 2) / \
               (vol_down.rolling(26).sum() + vol_eq.rolling(26).sum() / 2) * 100
    
    # --- 成交量均线 ---
    df['VOL_MA5'] = df['volume'].rolling(window=5).mean()
    df['VOL_MA10'] = df['volume'].rolling(window=10).mean()
    
    return clean_nan(df.to_dict('records'))

def clean_nan(records):
    for r in records:
        for k, v in r.items():
            if isinstance(v, float) and math.isnan(v):
                r[k] = None
    return records

# 指标简介接口
@app.route('/api/indicator-info/<name>')
def get_indicator_info(name):
    info = INDICATOR_INFO.get(name.upper())
    if info:
        return jsonify(info)
    return jsonify({'error': 'Unknown indicator'}), 404

# 获取所有指标简介列表
@app.route('/api/indicator-info')
def list_indicator_info():
    return jsonify([{'id': k, 'name': v['name'], 'category': v['category']} for k, v in INDICATOR_INFO.items()])

# ========== 点位分析（点击K线当天）==========
@app.route('/api/point-analysis/<code>/<date>')
def point_analysis(code, date):
    """返回每个指标独立的预测和准确性，以及汇总评分"""
    try:
        klines = get_kline_data(code, 200)
        if not klines:
            return jsonify({'error': 'No data'}), 404
        
        idx = None
        for i, k in enumerate(klines):
            if k['date'] == date:
                idx = i
                break
        if idx is None:
            return jsonify({'error': f'Date {date} not found'}), 404
        
        current = klines[idx]
        prev = klines[idx - 1] if idx > 0 else current
        
        # 后续数据用于验证准确性
        future_data = klines[idx+1:idx+6]
        start_price = current['close']
        actual_trend = None
        d1_ret = None
        if len(future_data) > 0:
            d1_ret = round((future_data[0]['close'] - start_price) / start_price * 100, 2)
            if d1_ret > 0.5:
                actual_trend = '上涨'
            elif d1_ret < -0.5:
                actual_trend = '下跌'
            else:
                actual_trend = '震荡'
        
        # 后续走势数据
        comparison = None
        if len(future_data) > 0:
            comparison = {
                'start_price': start_price,
                'subsequent': [{
                    'date': f['date'],
                    'close': f['close'],
                    'change_pct': round((f['close'] - start_price) / start_price * 100, 2)
                } for f in future_data[:5]],
                'actual_trend': actual_trend,
                'd1_return_pct': d1_ret,
            }
        
        # ---- 每个指标独立分析 ----
        indicator_results = []
        
        # MA
        if current.get('MA20') is not None:
            v = round(current['MA20'], 2)
            if current['close'] > current['MA20']:
                pred = '看涨'
            else:
                pred = '看跌'
            acc = judge_accuracy(pred, actual_trend)
            indicator_results.append({
                'id': 'MA', 'name': '移动平均线', 'value': f'{v}', 'unit': '¥',
                'signal': '多头' if current['close'] > v else '空头',
                'prediction': pred, 'actual': actual_trend, 'accuracy': acc
            })
        
        # MACD
        if current.get('MACD') is not None and current.get('MACD_SIGNAL') is not None:
            v = round(current['MACD'], 4)
            if current['MACD'] > current['MACD_SIGNAL']:
                pred = '看涨'
            else:
                pred = '看跌'
            acc = judge_accuracy(pred, actual_trend)
            indicator_results.append({
                'id': 'MACD', 'name': 'MACD', 'value': f'{v} (DIF)', 'unit': '',
                'signal': '金叉' if current['MACD'] > current['MACD_SIGNAL'] and prev.get('MACD', 0) <= prev.get('MACD_SIGNAL', 0) else '',
                'prediction': pred, 'actual': actual_trend, 'accuracy': acc
            })
        
        # KDJ
        if current.get('K') is not None and current.get('D') is not None:
            v = round(current['K'], 2)
            if current['K'] > current['D']:
                pred = '看涨'
            else:
                pred = '看跌'
            acc = judge_accuracy(pred, actual_trend)
            indicator_results.append({
                'id': 'KDJ', 'name': 'KDJ', 'value': f'K={v} D={round(current["D"],2)}', 'unit': '',
                'signal': '金叉' if current['K'] > current['D'] else '死叉',
                'prediction': pred, 'actual': actual_trend, 'accuracy': acc
            })
        
        # RSI
        if current.get('RSI') is not None:
            v = round(current['RSI'], 2)
            if v > 50:
                pred = '看涨'
            else:
                pred = '看跌'
            acc = judge_accuracy(pred, actual_trend)
            indicator_results.append({
                'id': 'RSI', 'name': 'RSI', 'value': f'{v}', 'unit': '',
                'signal': '超买' if v > 70 else '超卖' if v < 30 else '中性',
                'prediction': pred, 'actual': actual_trend, 'accuracy': acc
            })
        
        # BOLL
        if current.get('BOLL_MID') is not None:
            v = round(current['BOLL_MID'], 2)
            if current['close'] > v:
                pred = '看涨'
            else:
                pred = '看跌'
            acc = judge_accuracy(pred, actual_trend)
            indicator_results.append({
                'id': 'BOLL', 'name': '布林带', 'value': f'{v} (中轨)', 'unit': '¥',
                'signal': '中轨上方' if current['close'] > v else '中轨下方',
                'prediction': pred, 'actual': actual_trend, 'accuracy': acc
            })
        
        # OBV
        if current.get('OBV') is not None and prev.get('OBV') is not None:
            v = int(current['OBV'])
            if current['OBV'] > prev['OBV']:
                pred = '看涨'
            else:
                pred = '看跌'
            acc = judge_accuracy(pred, actual_trend)
            indicator_results.append({
                'id': 'OBV', 'name': '能量潮', 'value': f'{v}', 'unit': '',
                'signal': '资金流入' if current['OBV'] > prev['OBV'] else '资金流出',
                'prediction': pred, 'actual': actual_trend, 'accuracy': acc
            })
        
        # WR
        if current.get('WR') is not None:
            v = round(current['WR'], 2)
            if v > -50:
                pred = '看涨'
            else:
                pred = '看跌'
            acc = judge_accuracy(pred, actual_trend)
            indicator_results.append({
                'id': 'WR', 'name': '威廉指标', 'value': f'{v}', 'unit': '',
                'signal': '超买' if v > -20 else '超卖' if v < -80 else '中性',
                'prediction': pred, 'actual': actual_trend, 'accuracy': acc
            })
        
        # CCI
        if current.get('CCI') is not None:
            v = round(current['CCI'], 2)
            if v > 0:
                pred = '看涨'
            else:
                pred = '看跌'
            acc = judge_accuracy(pred, actual_trend)
            indicator_results.append({
                'id': 'CCI', 'name': '商品通道', 'value': f'{v}', 'unit': '',
                'signal': '强多' if v > 100 else '强空' if v < -100 else '中性',
                'prediction': pred, 'actual': actual_trend, 'accuracy': acc
            })
        
        # BIAS
        if current.get('BIAS6') is not None:
            v = round(current['BIAS6'], 2)
            if v > 0:
                pred = '看涨'
            else:
                pred = '看跌'
            acc = judge_accuracy(pred, actual_trend)
            indicator_results.append({
                'id': 'BIAS', 'name': '乖离率', 'value': f'{v}%', 'unit': '',
                'signal': '超买' if v > 5 else '超卖' if v < -5 else '正常',
                'prediction': pred, 'actual': actual_trend, 'accuracy': acc
            })
        
        # PSY
        if current.get('PSY') is not None:
            v = round(current['PSY'], 2)
            if v > 50:
                pred = '看涨'
            else:
                pred = '看跌'
            acc = judge_accuracy(pred, actual_trend)
            indicator_results.append({
                'id': 'PSY', 'name': '心理线', 'value': f'{v}', 'unit': '',
                'signal': '过热' if v > 75 else '悲观' if v < 25 else '中性',
                'prediction': pred, 'actual': actual_trend, 'accuracy': acc
            })
        
        # VR
        if current.get('VR') is not None:
            v = round(current['VR'], 2)
            if v > 150:
                pred = '看涨'
            else:
                pred = '看跌'
            acc = judge_accuracy(pred, actual_trend)
            indicator_results.append({
                'id': 'VR', 'name': '量比', 'value': f'{v}', 'unit': '',
                'signal': '活跃' if v > 450 else '低迷' if v < 70 else '正常',
                'prediction': pred, 'actual': actual_trend, 'accuracy': acc
            })
        
        # ATR
        if current.get('ATR') is not None:
            v = round(current['ATR'], 2)
            indicator_results.append({
                'id': 'ATR', 'name': '平均波幅', 'value': f'{v}', 'unit': '',
                'signal': '高波动' if v > current['close'] * 0.03 else '正常',
                'prediction': '--', 'actual': actual_trend, 'accuracy': '参考'
            })
        
        # ---- 汇总评分 ----
        buy_count = sum(1 for r in indicator_results if r['prediction'] == '看涨')
        sell_count = sum(1 for r in indicator_results if r['prediction'] == '看跌')
        total = buy_count + sell_count
        
        if total > 0:
            if buy_count > sell_count:
                agg_pred = '看涨'
                agg_conf = min(100, 50 + (buy_count - sell_count) * (50 // max(total, 1)))
            elif sell_count > buy_count:
                agg_pred = '看跌'
                agg_conf = min(100, 50 + (sell_count - buy_count) * (50 // max(total, 1)))
            else:
                agg_pred = '震荡'
                agg_conf = 50
        else:
            agg_pred = '--'
            agg_conf = 0
        
        agg_acc = judge_accuracy(agg_pred, actual_trend) if agg_pred != '--' else '--'
        
        return jsonify({
            'code': code,
            'date': date,
            'price': start_price,
            'indicator_results': indicator_results,
            'aggregate': {
                'buy_count': buy_count,
                'sell_count': sell_count,
                'neutral_count': len(indicator_results) - buy_count - sell_count,
                'prediction': agg_pred,
                'confidence': agg_conf,
                'accuracy': agg_acc,
            },
            'comparison': comparison,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def judge_accuracy(prediction, actual):
    """判断单个预测的准确性"""
    if not prediction or not actual or prediction == '--':
        return '--'
    mapping = {'看涨': '上涨', '看跌': '下跌', '震荡': '震荡'}
    expected = mapping.get(prediction, prediction)
    if expected == actual:
        return '正确 ✅'
    elif prediction == '震荡':
        return '中性'
    else:
        return '错误 ❌'


def analyze_signals_at_point(current, prev):
    """对单个时间点做信号分析，不依赖未来数据"""
    signals = []
    
    # MA
    ma_vals = [(current.get('MA5'), 'MA5'), (current.get('MA10'), 'MA10'),
               (current.get('MA20'), 'MA20'), (current.get('MA30'), 'MA30'),
               (current.get('MA60'), 'MA60')]
    valid_ma = [(v, n) for v, n in ma_vals if v is not None]
    if len(valid_ma) >= 2 and valid_ma[0][0] > valid_ma[1][0] > valid_ma[2][0] if len(valid_ma) >= 3 else valid_ma[0][0] > valid_ma[1][0]:
        signals.append({'indicator': 'MA', 'signal': '多头排列', 'trend': '上涨', 'strength': '强'})
    elif len(valid_ma) >= 2 and valid_ma[0][0] < valid_ma[1][0]:
        signals.append({'indicator': 'MA', 'signal': '空头排列', 'trend': '下跌', 'strength': '强'})
    
    # MACD
    if current.get('MACD') is not None and current.get('MACD_SIGNAL') is not None:
        cur_m = current['MACD']; cur_s = current['MACD_SIGNAL']
        prev_m = prev.get('MACD'); prev_s = prev.get('MACD_SIGNAL')
        if prev_m is not None and prev_s is not None:
            if cur_m > cur_s and prev_m <= prev_s:
                signals.append({'indicator': 'MACD', 'signal': '金叉', 'trend': '买入', 'strength': '中'})
            elif cur_m < cur_s and prev_m >= prev_s:
                signals.append({'indicator': 'MACD', 'signal': '死叉', 'trend': '卖出', 'strength': '中'})
    
    # KDJ
    if current.get('K') is not None and current.get('D') is not None:
        cur_k = current['K']; cur_d = current['D']
        prev_k = prev.get('K'); prev_d = prev.get('D')
        if prev_k is not None and prev_d is not None:
            if cur_k > cur_d and prev_k <= prev_d and cur_k < 80:
                signals.append({'indicator': 'KDJ', 'signal': '金叉', 'trend': '买入', 'strength': '中'})
            elif cur_k < cur_d and prev_k >= prev_d and cur_k > 20:
                signals.append({'indicator': 'KDJ', 'signal': '死叉', 'trend': '卖出', 'strength': '中'})
        if cur_k is not None:
            if cur_k > 80:
                signals.append({'indicator': 'KDJ', 'signal': '超买', 'trend': '回调', 'strength': '弱'})
            elif cur_k < 20:
                signals.append({'indicator': 'KDJ', 'signal': '超卖', 'trend': '反弹', 'strength': '弱'})
    
    # RSI
    if current.get('RSI') is not None:
        rsi = current['RSI']
        if rsi > 70:
            signals.append({'indicator': 'RSI', 'signal': '超买', 'trend': '回调', 'strength': '弱'})
        elif rsi < 30:
            signals.append({'indicator': 'RSI', 'signal': '超卖', 'trend': '反弹', 'strength': '弱'})
    
    # BOLL
    if current.get('close') is not None and current.get('BOLL_UP') is not None:
        if current['close'] > current['BOLL_UP']:
            signals.append({'indicator': 'BOLL', 'signal': '突破上轨', 'trend': '强势', 'strength': '强'})
        elif current['close'] < current['BOLL_DOWN']:
            signals.append({'indicator': 'BOLL', 'signal': '跌破下轨', 'trend': '弱势', 'strength': '强'})
    
    # WR
    if current.get('WR') is not None:
        wr = current['WR']
        if wr > -20:
            signals.append({'indicator': 'WR', 'signal': '超买', 'trend': '回调', 'strength': '弱'})
        elif wr < -80:
            signals.append({'indicator': 'WR', 'signal': '超卖', 'trend': '反弹', 'strength': '弱'})
    
    # CCI
    if current.get('CCI') is not None:
        cci = current['CCI']
        if cci > 100:
            signals.append({'indicator': 'CCI', 'signal': '强势超买', 'trend': '上涨', 'strength': '强'})
        elif cci < -100:
            signals.append({'indicator': 'CCI', 'signal': '弱势超卖', 'trend': '下跌', 'strength': '强'})
    
    # BIAS
    if current.get('BIAS6') is not None:
        b6 = current['BIAS6']
        if b6 > 5:
            signals.append({'indicator': 'BIAS', 'signal': '严重超买', 'trend': '回调', 'strength': '中'})
        elif b6 < -5:
            signals.append({'indicator': 'BIAS', 'signal': '严重超卖', 'trend': '反弹', 'strength': '中'})
    
    # PSY
    if current.get('PSY') is not None:
        psy = current['PSY']
        if psy > 75:
            signals.append({'indicator': 'PSY', 'signal': '市场过热', 'trend': '回调', 'strength': '中'})
        elif psy < 25:
            signals.append({'indicator': 'PSY', 'signal': '过度悲观', 'trend': '反弹', 'strength': '中'})
    
    # VR
    if current.get('VR') is not None:
        vr = current['VR']
        if vr is not None:
            if vr > 450:
                signals.append({'indicator': 'VR', 'signal': '顶部信号', 'trend': '下跌', 'strength': '强'})
            elif vr < 70:
                signals.append({'indicator': 'VR', 'signal': '底部信号', 'trend': '上涨', 'strength': '强'})
    
    # OBV 趋势
    if current.get('OBV') is not None and prev.get('OBV') is not None:
        if current['OBV'] > prev['OBV'] and prev.get('close') and current['close'] < prev['close']:
            signals.append({'indicator': 'OBV', 'signal': '量价背离', 'trend': '下跌', 'strength': '中'})
        elif current['OBV'] < prev['OBV'] and current['close'] > prev['close']:
            signals.append({'indicator': 'OBV', 'signal': '量价背离', 'trend': '上涨', 'strength': '中'})
    
    return signals

# 搜索股票
@app.route('/api/search')
def search_stock():
    query = request.args.get('q', '').upper().strip()
    results = []
    if len(query) < 1:
        return jsonify(results)
    for code, name in STOCK_MAP.items():
        if query in code or query in name.upper():
            results.append({'code': code, 'name': name})
        if len(results) >= 50:
            break
    return jsonify(results)

@app.route('/api/hot')
def get_hot_stocks():
    hot_codes = [
        "600519", "000858", "002594", "300750", "000333",
        "600036", "601318", "600276", "002415", "000725",
        "600887", "002230", "300059", "601012", "002475",
        "000002", "600030", "601888", "603288", "300122"
    ]
    hot = []
    for code in hot_codes:
        if code in STOCK_MAP:
            hot.append({'code': code, 'name': STOCK_MAP[code]})
    return jsonify(hot)

@app.route('/api/all_stocks')
def get_all_stocks():
    stocks = [{'code': code, 'name': name} for code, name in STOCK_MAP.items()]
    return jsonify(stocks)

# 技术分析报告
@app.route('/api/analysis/<code>')
def get_analysis(code):
    try:
        klines = get_kline_data(code, 60)
        if not klines:
            return jsonify({'error': 'No data'}), 404
        
        latest = klines[-1]
        prev = klines[-2] if len(klines) > 1 else latest
        
        analysis = {
            'code': code,
            'date': latest['date'],
            'price': latest['close'],
            'signals': []
        }
        
        analysis['signals'] = analyze_signals_at_point(latest, prev)
        
        buy_count = sum(1 for s in analysis['signals'] if s['trend'] in ['买入', '上涨', '反弹', '强势'])
        sell_count = sum(1 for s in analysis['signals'] if s['trend'] in ['卖出', '下跌', '回调', '弱势'])
        
        if buy_count > sell_count:
            analysis['overall'] = '看多'
            analysis['score'] = min(100, 50 + (buy_count - sell_count) * 12)
        elif sell_count > buy_count:
            analysis['overall'] = '看空'
            analysis['score'] = max(0, 50 - (sell_count - buy_count) * 12)
        else:
            analysis['overall'] = '震荡'
            analysis['score'] = 50
        
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_kline_data(code, count=60):
    market = 'sh' if code.startswith('6') or code.startswith('689') else 'sz'
    
    url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
    params = {'param': f"{market}{code},day,,,{count},qfq"}
    
    response = requests.get(url, params=params, timeout=15)
    data = response.json()
    
    stock_key = f"{market}{code}"
    period_key = "qfqday"
    
    klines = []
    if data.get('data') and data['data'].get(stock_key):
        stock_data = data['data'][stock_key]
        klines_raw = None
        if period_key in stock_data:
            klines_raw = stock_data[period_key]
        elif 'day' in stock_data:
            klines_raw = stock_data['day']
        
        if klines_raw:
            for item in klines_raw:
                klines.append({
                    'date': item[0],
                    'open': float(item[1]),
                    'close': float(item[2]),
                    'high': float(item[3]),
                    'low': float(item[4]),
                    'volume': float(item[5]),
                    'amount': 0, 'amplitude': 0,
                    'changePercent': 0, 'change': 0, 'turnover': 0,
                })
            for i in range(len(klines)):
                if i > 0:
                    prev_close = klines[i-1]['close']
                    curr_close = klines[i]['close']
                    klines[i]['change'] = curr_close - prev_close
                    klines[i]['changePercent'] = round((curr_close - prev_close) / prev_close * 100, 2)
                    klines[i]['amplitude'] = round((klines[i]['high'] - klines[i]['low']) / prev_close * 100, 2)
    
    return calculate_indicators(klines)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

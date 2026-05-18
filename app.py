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
        "name": "移动平均线 MA (Moving Average)",
        "desc": "MA 是过去 N 个交易日的收盘价算术平均值。原理：将每日收盘价连成一条光滑曲线，过滤掉单日杂音，反映价格运行方向。短期均线（MA5/MA10）反应灵敏但噪音多；长期均线（MA60）迟钝但方向稳定。",
        "principle": "📖 核心原理\n价格 > MA → 多方占优，均线构成支撑\n价格 < MA → 空方占优，均线构成压力\n\n📊 使用法则\n1. 短期均线上穿长期均线 → 金叉，看涨信号\n2. 短期均线下穿长期均线 → 死叉，看跌信号\n3. 多条均线向上发散（多头排列） → 强势上涨\n4. 多条均线向下发散（空头排列） → 弱势下跌\n5. 均线粘连后向上发散 → 启动信号\n6. 均线斜率由陡变平 → 趋势衰竭",
        "periods": [5, 10, 20, 30, 60],
        "author": "统计学基本概念",
        "category": "趋势"
    },
    "MACD": {
        "name": "指数平滑异同移动平均线 MACD (Moving Average Convergence Divergence)",
        "desc": "MACD 由三部分组成：① DIF（差离值）= 12日EMA - 26日EMA，代表快慢均线的距离；② DEA（讯号线）= DIF的9日EMA，是DIF的平滑信号线；③ MACD柱（柱状线）= DIF - DEA，反映多空动能强弱。由 Gerald Appel 于1979年发明。",
        "principle": "📖 核心原理\nDIF是「快船」，DEA是「慢船」。快船超过慢船 → 金叉（多头占优）；快船落后慢船 → 死叉（空头占优）。MACD柱的高度 = 两船的距离，柱越长动能越强。\n\n📊 使用法则\n1. DIF 上穿 DEA → 金叉（MACD柱由绿转红），买入信号\n2. DIF 下穿 DEA → 死叉（MACD柱由红转绿），卖出信号\n3. MACD柱持续放大 → 趋势加速\n4. MACD柱持续缩小 → 趋势减速，警惕反转\n5. 顶背离：股价创新高，但DIF未创新高 → 见顶信号\n6. 底背离：股价创新低，但DIF未创新低 → 见底信号",
        "author": "Gerald Appel, 1979",
        "category": "趋势"
    },
    "KDJ": {
        "name": "随机指标 KDJ (Stochastic Oscillator)",
        "desc": "KDJ 由三条线组成：① RSV（未成熟随机值）= (收盘价-9日最低)/(9日最高-9日最低)×100，衡量收盘价在近期价格区间的位置；② K值 = RSV的加权移动平均（快速确认线）；③ D值 = K值的加权移动平均（慢速主线）；④ J值 = 3K-2D（方向敏感线，反应最灵敏）。由 George Lane 于1950年代发明。",
        "principle": "📖 核心原理\n收盘价越接近9日最高价 → RSV越高（买方主导，收在高位）\n收盘价越接近9日最低价 → RSV越低（卖方打压，收在低位）\nJ值是K和D的放大版，反应最快但假信号也最多\n\n📊 使用法则\n1. K值 > 80 → 超买区，警惕回调\n2. K值 < 20 → 超卖区，关注反弹\n3. K线上穿D线且处于超卖区 → 强烈买入信号\n4. K线下穿D线且处于超买区 → 强烈卖出信号\n5. J值 > 100 → 严重超买，随时可能回落\n6. J值 < 0 → 严重超卖，随时可能反弹",
        "author": "George Lane, 1950s",
        "category": "摆动"
    },
    "RSI": {
        "name": "相对强弱指标 RSI (Relative Strength Index)",
        "desc": "RSI 的计算公式：RSI = 100 - 100/(1+RS)，其中 RS = 14日内平均涨幅 ÷ 14日内平均跌幅。取值范围0~100，数值越高表示近期涨势越猛。由 Welles Wilder 于1978年发明。",
        "principle": "📖 核心原理\nRSI的本质是「涨的力量 vs 跌的力量」。如果14天里每天都在涨 → RSI接近100；每天都在跌 → RSI接近0。50是多空分界线。\n\n📊 使用法则\n1. RSI > 70 → 超买区（涨太猛，短期回调概率大）\n2. RSI < 30 → 超卖区（跌太狠，短期反弹概率大）\n3. RSI在50上方运行 → 多头市场，逢低做多\n4. RSI在50下方运行 → 空头市场，逢高做空\n5. 顶背离：股价新高RSI未新高 → 卖出信号\n6. 底背离：股价新低RSI未新低 → 买入信号",
        "author": "Welles Wilder, 1978",
        "category": "摆动"
    },
    "BOLL": {
        "name": "布林带 BOLL (Bollinger Bands)",
        "desc": "BOLL 由三条轨道组成：① 中轨 = 20日均线(MA20)；② 上轨 = MA20 + 2倍标准差(σ)；③ 下轨 = MA20 - 2倍标准差。标准差衡量价格离散程度，价格约95%的时间在上下轨之间运行。由 John Bollinger 于1980年代发明。",
        "principle": "📖 核心原理\n标准差(σ)越大 → 布林带越宽 → 波动剧烈\n标准差越小 → 布林带越窄 → 波动平缓\n带宽极窄（布林收口）→ 暴风雨前的宁静，即将变盘\n\n📊 使用法则\n1. 价格触及上轨 → 短期超买，有回调压力\n2. 价格触及下轨 → 短期超卖，有反弹动力\n3. 价格突破上轨且带宽放大 → 强势上涨启动\n4. 价格跌破下轨且带宽放大 → 恐慌下跌\n5. 沿着上轨爬升 → 主升浪（不要轻易做空）\n6. 沿着下轨滑落 → 主跌浪（不要轻易抄底）",
        "author": "John Bollinger, 1980s",
        "category": "通道"
    },
    "OBV": {
        "name": "能量潮 OBV (On-Balance Volume)",
        "desc": "OBV 计算方式：当天收盘价 > 昨天收盘价 → 累加成交量；当天收盘价 < 昨天 → 减去成交量；不变则 OBV 不变。核心假设：「量在价先」，成交量变化领先于价格变化。由 Joseph Granville 于1963年发明。",
        "principle": "📖 核心原理\nOBV上升 = 资金在低位吸筹、高位追买，多方控盘\nOBV下降 = 资金在持续流出，空方主导\nOBV横盘 = 多空平衡，方向待定\n\n📊 使用法则\n1. OBV创新高，价格未创新高 → 量价顶背离，看跌\n2. OBV创新低，价格未创新低 → 量价底背离，看涨\n3. OBV与价格同步上升 → 上升趋势健康，持股\n4. OBV与价格同步下降 → 下降趋势确认，观望\n5. 价格下跌但OBV横盘不跌 → 底部吸筹，后市看涨",
        "author": "Joseph Granville, 1963",
        "category": "量价"
    },
    "WR": {
        "name": "威廉指标 WR (Williams %R)",
        "desc": "WR 计算公式：(14日最高价-收盘价)/(14日最高价-14日最低价)×(-100)。衡量收盘价在近期价格区间中的相对位置。注意WR是反向指标：数值越高(-0附近)表示价格在区间顶部即超买，数值越低(-100附近)表示在底部即超卖。由 Larry Williams 于1973年发明。",
        "principle": "📖 核心原理\nWR接近0 → 收盘价接近14日最高点 → 短期涨过头了\nWR接近-100 → 收盘价接近14日最低点 → 短期跌过头了\nWR本质是「当前价格在N日价格区间中的百分位」取反\n\n📊 使用法则\n1. WR在0~-20区域 → 超买，警惕回调\n2. WR在-80~-100区域 → 超卖，关注反弹\n3. WR多次探顶(0~-20)而不破 → 顶部确认\n4. WR多次探底(-80~-100)而不破 → 底部确认\n5. 注意：数值越小(越接近-100)越看涨，越大(越接近0)越看跌",
        "author": "Larry Williams, 1973",
        "category": "摆动"
    },
    "CCI": {
        "name": "商品通道指数 CCI (Commodity Channel Index)",
        "desc": "CCI 计算公式：(典型价格-20日均值)/(0.015×平均偏差)。典型价格=(最高+最低+收盘)/3。CCI不局限于±100，理论上可到正负无穷，对极端行情反应敏感。由 Donald Lambert 于1980年发明。",
        "principle": "📖 核心原理\nCCI衡量「当前价格偏离正常水平的程度」。±100是统计意义上的异常边界——超过100意味着价格偏离均值超过1.5倍标准差。\n\n📊 使用法则\n1. CCI > +100 → 价格异常强势，超买\n2. CCI < -100 → 价格异常弱势，超卖\n3. CCI从上向下穿越+100 → 卖出信号\n4. CCI从下向上穿越-100 → 买入信号\n5. CCI在±100之间 → 正常波动，无明确方向\n6. CCI与价格背离 → 趋势反转信号",
        "author": "Donald Lambert, 1980",
        "category": "摆动"
    },
    "ATR": {
        "name": "平均真实波幅 ATR (Average True Range)",
        "desc": "ATR 计算方式：① 先算每日真实波幅 TR = max(今高-今低, |今高-昨收|, |今低-昨收|)；② 取14日TR的平均值。注意：ATR只衡量波动幅度，不指示涨跌方向。由 Welles Wilder 于1978年发明。",
        "principle": "📖 核心原理\nATR反映的是「市场情绪的温度」。高ATR = 多空分歧大、成交活跃；低ATR = 市场平淡、方向不明。\n\n📊 使用法则\n1. ATR上升 → 波动加大，趋势可能启动或加速\n2. ATR下降 → 波动减小，可能进入盘整\n3. 止损位 = 入场价 ± 2~3倍ATR（灵活止损）\n4. ATR突然飙升 → 市场出现恐慌或狂热，可能反转\n5. 注意：ATR不指示方向，只指示烈度",
        "author": "Welles Wilder, 1978",
        "category": "波动"
    },
    "BIAS": {
        "name": "乖离率 BIAS (Bias Ratio)",
        "desc": "BIAS = (收盘价-N日均线)/N日均线 × 100%。衡量价格偏离移动平均线的百分比。核心逻辑：价格围绕均线波动，偏离过多会回归，即「均值回归」。",
        "principle": "📖 核心原理\n正BIAS = 价格高于均线，短线涨多了\n负BIAS = 价格低于均线，短线跌多了\nBIAS越大(正或负)，回拉力量越强\n\n📊 使用法则\n1. BIAS6 > +5% → 严重超买，高抛时机\n2. BIAS6 < -5% → 严重超卖，低吸时机\n3. BIAS由负转正 → 站上均线，趋势转多\n4. BIAS由正转负 → 跌破均线，趋势转空\n5. 不同股票乖离阈值不同，大盘股阈值小、活跃股阈值大",
        "author": "技术分析通用概念",
        "category": "摆动"
    },
    "PSY": {
        "name": "心理线 PSY (Psychological Line)",
        "desc": "PSY = N日内上涨天数/N × 100%（默认N=12）。本质是统计「多方赢了多少天」，反映市场乐观/悲观程度。极端值反映群体心理过激。",
        "principle": "📖 核心原理\nPSY > 75 → 过去12天有9天以上在涨，市场过于乐观\nPSY < 25 → 过去12天有9天以上在跌，市场过于悲观\n物极必反：「别人贪婪我恐惧，别人恐惧我贪婪」\n\n📊 使用法则\n1. PSY > 75 → 市场过热，群体乐观到极点，警惕回调\n2. PSY < 25 → 市场极度悲观，恐慌抛售尾声，关注反弹\n3. PSY在50附近 → 多空均衡，方向不明\n4. PSY从高位回落 → 乐观情绪消退，卖出信号\n5. PSY从低位回升 → 悲观情绪修复，买入信号",
        "author": "技术分析通用指标",
        "category": "心理"
    },
    "VR": {
        "name": "成交量比率 VR (Volume Ratio)",
        "desc": "VR = (26日内上涨日成交量之和+平盘日量/2)/(下跌日成交量之和+平盘日量/2) × 100%。比较「买方投入的资金」vs「卖方投入的资金」，判断多空力道对比。VR > 100 表示买方花钱更多。",
        "principle": "📖 核心原理\nVR衡量「谁在花更多钱」。买方花更多钱 → VR上升 → 多方积极；卖方花更多钱 → VR下降 → 空方积极。极端VR值意味着资金过热或过冷。\n\n📊 使用法则\n1. VR > 450 → 资金过度涌入，市场过热，顶部信号\n2. VR < 70 → 资金极度低迷，市场冰点，底部信号\n3. VR在150附近 → 正常区域，多空均衡\n4. VR上升但价格横盘 → 资金在低位吸筹，后市看涨\n5. VR下降但价格横盘 → 资金在悄悄出货，后市看跌",
        "author": "技术分析通用指标",
        "category": "量价"
    },
    "VOL_MA": {
        "name": "成交量均线 VOL_MA (Volume Moving Average)",
        "desc": "VOL_MA 是成交量的移动平均线（5日/10日），用于判断量能变化趋势。核心逻辑：量在价先——趋势的启动和持续都需要成交量配合。",
        "principle": "📖 核心原理\n量能 = 趋势的燃料。有量有价 → 趋势健康；无量上涨 → 虚涨，随时可能崩塌。成交量均线帮助判断量能是放大还是萎缩。\n\n📊 使用法则\n1. 放量上涨（量>MA5）→ 上涨有力，趋势健康\n2. 缩量上涨（量<MA5）→ 上攻乏力，警惕回落\n3. 放量下跌（量>MA5）→ 恐慌出逃，跌势加速\n4. 缩量下跌（量<MA5）→ 抛压衰竭，可能见底\n5. 量突然放大3倍以上 → 变盘信号，关注方向",
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
                    'amount': float(fields[37]) * 10000 if len(fields) > 37 else 0,
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
        klines = get_kline_data(code, 500)
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
        
        # 后续数据用于多窗口验证
        future_data = klines[idx+1:idx+11]
        start_price = current['close']
        
        # 多窗口实际走势
        actual_windows = {}
        comparison = {'start_price': start_price, 'windows': {}, 'subsequent': []}
        
        for days, label in [(1, 'd1'), (3, 'd3'), (5, 'd5')]:
            if idx + days < len(klines):
                ret = round((klines[idx+days]['close'] - start_price) / start_price * 100, 2)
                if ret > 1.0:
                    trend = '上涨'
                elif ret < -1.0:
                    trend = '下跌'
                else:
                    trend = '震荡'
                actual_windows[label] = {'return': ret, 'trend': trend}
        
        # 默认用 d3 验证
        actual_trend = actual_windows.get('d3', {}).get('trend') if actual_windows else None
        
        # 后续走势数据（用于前端展示）
        for f in future_data[:5]:
            comparison['subsequent'].append({
                'date': f['date'],
                'close': f['close'],
                'change_pct': round((f['close'] - start_price) / start_price * 100, 2)
            })
        comparison['actual_windows'] = actual_windows
        
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
        
        # ---- 汇总评分 (动态权重) ----
        # 在整个K线序列上回测，得到每个指标的准确率和权重
        weights = compute_indicator_weights(klines)
        
        # 提取各指标准确率（用于前端展示）
        indicator_weights = {}
        for ind_id, win_stats in weights.items():
            if 'best_window' in win_stats:
                indicator_weights[ind_id] = {
                    'accuracy': win_stats['best_accuracy'],
                    'weight': round(win_stats['weight'] * 100),
                    'samples': win_stats.get(win_stats['best_window'], {}).get('total', 0),
                    'correct': win_stats.get(win_stats['best_window'], {}).get('correct', 0),
                    'best_window': win_stats['best_window'],
                    'windows': {
                        w: {
                            'accuracy': win_stats[w].get('accuracy', 0),
                            'samples': win_stats[w].get('total', 0),
                            'correct': win_stats[w].get('correct', 0),
                        }
                        for w in ['d1', 'd3', 'd5'] if 'accuracy' in win_stats.get(w, {})
                    }
                }
        
        # 等权重汇总（简单多数）
        buy_count = sum(1 for r in indicator_results if r['prediction'] == '看涨')
        sell_count = sum(1 for r in indicator_results if r['prediction'] == '看跌')
        
        # 动态权重汇总
        pa = {'indicator_results': indicator_results}
        wagg = weighted_aggregate(pa, weights)
        
        agg_acc = judge_accuracy(wagg['prediction'], actual_trend) if wagg['prediction'] != '--' else '--'
        
        return jsonify({
            'code': code,
            'date': date,
            'price': start_price,
            'indicator_results': indicator_results,
            'aggregate': {
                'buy_count': buy_count,
                'sell_count': sell_count,
                'neutral_count': len(indicator_results) - buy_count - sell_count,
                'prediction': wagg['prediction'],
                'confidence': wagg['confidence'],
                'accuracy': agg_acc,
            },
            'weighted_aggregate': {
                'prediction': wagg['prediction'],
                'confidence': wagg['confidence'],
                'buy_weight': wagg['buy_weight'],
                'sell_weight': wagg['sell_weight'],
                'buy_pct': wagg.get('buy_pct', 0),
                'sell_pct': wagg.get('sell_pct', 0),
                'accuracy': agg_acc,
            },
            'indicator_weights': indicator_weights,
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


def get_indicator_predictions(current, prev):
    """对单根K线，返回所有可计算指标的预测字典 {id: prediction}"""
    preds = {}
    
    # MA — 收盘价 vs MA20
    if current.get('MA20') is not None:
        preds['MA'] = '看涨' if current['close'] > current['MA20'] else '看跌'
    
    # MACD — DIF vs DEA
    if current.get('MACD') is not None and current.get('MACD_SIGNAL') is not None:
        preds['MACD'] = '看涨' if current['MACD'] > current['MACD_SIGNAL'] else '看跌'
    
    # KDJ — K vs D
    if current.get('K') is not None and current.get('D') is not None:
        preds['KDJ'] = '看涨' if current['K'] > current['D'] else '看跌'
    
    # RSI — vs 50
    if current.get('RSI') is not None:
        preds['RSI'] = '看涨' if current['RSI'] > 50 else '看跌'
    
    # BOLL — 收盘价 vs 中轨
    if current.get('BOLL_MID') is not None:
        preds['BOLL'] = '看涨' if current['close'] > current['BOLL_MID'] else '看跌'
    
    # OBV — 变化方向
    if current.get('OBV') is not None and prev.get('OBV') is not None:
        preds['OBV'] = '看涨' if current['OBV'] > prev['OBV'] else '看跌'
    
    # WR — vs -50
    if current.get('WR') is not None:
        preds['WR'] = '看涨' if current['WR'] > -50 else '看跌'
    
    # CCI — vs 0
    if current.get('CCI') is not None:
        preds['CCI'] = '看涨' if current['CCI'] > 0 else '看跌'
    
    # BIAS — BIAS6 vs 0
    if current.get('BIAS6') is not None:
        preds['BIAS'] = '看涨' if current['BIAS6'] > 0 else '看跌'
    
    # PSY — vs 50
    if current.get('PSY') is not None:
        preds['PSY'] = '看涨' if current['PSY'] > 50 else '看跌'
    
    # VR — vs 150
    if current.get('VR') is not None:
        preds['VR'] = '看涨' if current['VR'] > 150 else '看跌'
    
    return preds


def compute_indicator_weights(klines, min_samples=30):
    """d1/d3/d5 三窗口回测，每个指标取最佳窗口的胜率作为权重。
    返回 {indicator_id: {best_window, accuracy, weight, windows: {d1:{}, d3:{}, d5:{}}}}"""
    from collections import defaultdict
    
    windows = {'d1': 1, 'd3': 3, 'd5': 5}
    
    # stats[indicator][window] = {'correct': N, 'total': N}
    stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    
    n = len(klines)
    start = 60  # MA60 预热
    end = n - max(windows.values())
    
    for i in range(start, end):
        current = klines[i]
        prev = klines[i - 1]
        preds = get_indicator_predictions(current, prev)
        start_price = current['close']
        
        for win_name, win_days in windows.items():
            future_idx = i + win_days
            if future_idx >= n:
                continue
            end_price = klines[future_idx]['close']
            ret = (end_price - start_price) / start_price * 100
            
            if ret > 1.0:
                actual = '上涨'
            elif ret < -1.0:
                actual = '下跌'
            else:
                actual = '震荡'
            
            mapping = {'看涨': '上涨', '看跌': '下跌', '震荡': '震荡'}
            for ind_id, pred in preds.items():
                stats[ind_id][win_name]['total'] += 1
                if mapping.get(pred) == actual:
                    stats[ind_id][win_name]['correct'] += 1
    
    # 计算每个窗口的准确率
    for ind_id, win_stats in stats.items():
        for win_name, s in win_stats.items():
            if s['total'] >= min_samples:
                s['accuracy'] = round(s['correct'] / s['total'] * 100, 1)
    
    # 每个指标取最佳窗口
    for ind_id, win_stats in stats.items():
        best_acc = 0
        best_win = None
        for win_name in windows:
            s = win_stats[win_name]
            if 'accuracy' in s and s['accuracy'] > best_acc:
                best_acc = s['accuracy']
                best_win = win_name
        
        if best_win:
            win_stats['best_window'] = best_win
            win_stats['best_accuracy'] = best_acc
    
    # 归一化：所有权重之和 = 1.0
    total_best = sum(s.get('best_accuracy', 0) for s in stats.values())
    for ind_id, win_stats in stats.items():
        if 'best_accuracy' in win_stats and total_best > 0:
            # raw_weight = best_accuracy / total_best，保底 10% 后再归一化
            raw = win_stats['best_accuracy'] / total_best
            floor = 0.10 / len(stats) if len(stats) > 0 else 0.01
            win_stats['weight'] = round(max(raw, floor), 4)
    
    # 二次归一化保证和严格等于 1
    total_w = sum(s.get('weight', 0) for s in stats.values())
    if total_w > 0:
        for s in stats.values():
            if 'weight' in s:
                s['weight'] = round(s['weight'] / total_w, 4)
    
    return stats


def weighted_aggregate(pa_result, weights):
    """用动态权重（各指标取最佳窗口）重新计算汇总评分"""
    buy_weight = 0.0
    sell_weight = 0.0
    total_weight = 0.0
    
    for r in pa_result.get('indicator_results', []):
        ind_id = r['id']
        w = 1.0  # 等权重兜底
        # 取该指标的最佳窗口权重
        if ind_id in weights and 'weight' in weights[ind_id]:
            w = weights[ind_id]['weight']
        
        if r['prediction'] == '看涨':
            buy_weight += w
        elif r['prediction'] == '看跌':
            sell_weight += w
        total_weight += w
    
    if total_weight > 0:
        buy_pct = round(buy_weight / total_weight * 100)
        sell_pct = round(sell_weight / total_weight * 100)
        
        if buy_pct > 55:
            agg_pred = '看涨'
            agg_conf = buy_pct
        elif sell_pct > 55:
            agg_pred = '看跌'
            agg_conf = sell_pct
        else:
            agg_pred = '震荡'
            agg_conf = max(buy_pct, sell_pct)
    else:
        agg_pred = '--'
        agg_conf = 0
        buy_pct = 0
        sell_pct = 0
    
    return {
        'prediction': agg_pred,
        'confidence': agg_conf,
        'buy_weight': round(buy_weight, 2),
        'sell_weight': round(sell_weight, 2),
        'buy_pct': buy_pct,
        'sell_pct': sell_pct,
    }


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

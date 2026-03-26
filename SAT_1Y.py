import numpy as np
import pandas as pd
# Numpy 兼容性补丁
if not hasattr(np, 'bool8'): np.bool8 = np.bool_

import bokeh
from bokeh.document import Document
# Bokeh 3.x 兼容性补丁：解决 'Document' object has no attribute 'js_on_event'
if not hasattr(Document, 'js_on_event'):
    Document.js_on_event = Document.on_event

import os
import datetime
import pandas_ta as ta
import yfinance as yf
from backtesting import Backtest, Strategy
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import efinance as ef

warnings.filterwarnings('ignore')

# ==========================================
# 策略定义
# ==========================================
class StrategySix(Strategy):
    st_period = 10; st_mult = 3.0; cci_period = 14; vol_threshold = 1.2
    def init(self):
        c, h, l, v = self.data.Close, self.data.High, self.data.Low, self.data.Volume
        st = ta.supertrend(pd.Series(h), pd.Series(l), pd.Series(c), length=int(self.st_period), multiplier=float(self.st_mult))
        self.st_dir = self.I(lambda: st.iloc[:, 1])
        self.cci = self.I(ta.cci, pd.Series(h), pd.Series(l), pd.Series(c), length=int(self.cci_period))
        self.vol_ma5 = self.I(ta.sma, pd.Series(v), length=5)
    def next(self):
        price, vol = self.data.Close[-1], self.data.Volume[-1]
        vol_ok = vol > self.vol_ma5[-1] * self.vol_threshold
        if not self.position and self.st_dir[-1] == 1 and self.cci[-1] < -100 and vol_ok: self.buy()
        elif self.st_dir[-1] == -1: self.position.close()

class StrategyNine(Strategy):
    bb_length = 20; bb_std = 2.0; ma_fast_len = 5; vol_threshold = 1.2
    def init(self):
        c, v = pd.Series(self.data.Close), pd.Series(self.data.Volume)
        bb = ta.bbands(c, length=self.bb_length, std=self.bb_std)
        self.bb_l = self.I(lambda: bb.iloc[:, 0])
        self.ma_fast = self.I(ta.sma, c, self.ma_fast_len)
        self.vol_ma5 = self.I(ta.sma, v, 5)
    def next(self):
        price, vol = self.data.Close[-1], self.data.Volume[-1]
        vol_ok = vol > self.vol_ma5[-1] * self.vol_threshold
        if not self.position and price < self.bb_l[-1] and price < self.ma_fast[-1] and vol_ok: self.buy()
        elif price > self.ma_fast[-1]: self.position.close()

class StrategyTen(Strategy):
    vol_threshold = 1.2
    def init(self):
        c, h, l, v = pd.Series(self.data.Close), pd.Series(self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Volume)
        ichi = ta.ichimoku(h, l, c)[0]
        self.span_a, self.span_b = self.I(lambda: ichi.iloc[:, 2]), self.I(lambda: ichi.iloc[:, 3])
        self.vol_ma5 = self.I(ta.sma, v, 5)
    def next(self):
        price, vol = self.data.Close[-1], self.data.Volume[-1]
        vol_ok = vol > self.vol_ma5[-1] * self.vol_threshold
        if not self.position and price > self.span_a[-1] and price > self.span_b[-1] and vol_ok: self.buy()
        elif price < self.span_a[-1]: self.position.close()

# ==========================================
# 工具函数
# ==========================================
def get_data_yf(symbol, start_dt, end_dt):
    yf_code = f"{symbol}.SS" if symbol.startswith(('6', '9')) else f"{symbol}.SZ"
    try:
        df = yf.download(yf_code, start=start_dt, end=end_dt, progress=False)
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    except: return None

def process_stock(stock_info, start_dt, end_dt):
    symbol, name = stock_info['stock_code'], stock_info['stock_name']
    df = get_data_yf(symbol, start_dt, end_dt)
    if df is None or len(df) < 50: return None
    try:
        bt6, bt9, bt10 = Backtest(df, StrategySix), Backtest(df, StrategyNine), Backtest(df, StrategyTen)
        s6, s9, s10 = bt6.run(), bt9.run(), bt10.run()
        # 信号判定逻辑
        def is_buy(s): return s['_strategy'].data.Close[-1] < s['_strategy'].data.Close[-2] # 简化示例
        score = sum([1 if s6['Return [%]'] > 0 else 0, 1 if s9['Return [%]'] > 0 else 0, 1 if s10['Return [%]'] > 0 else 0])
        return {'code': symbol, 'name': name, 'close': df['Close'].iloc[-1], 'score': score}
    except: return None

def run_scanner():
    print(">>> 启动全市场扫描 (yfinance 模式)...")
    try:
        # 尝试获取列表，失败则使用预设核心池
        try:
            all_stocks = ef.stock.get_realtime_quotes()
            valid_stocks = all_stocks[all_stocks['股票代码'].str.match(r'^(60|00|30|68)')].copy()
            valid_stocks.rename(columns={'股票代码': 'stock_code', '股票名称': 'stock_name'}, inplace=True)
            valid_stocks = valid_stocks.head(100) # GitHub 环境建议限制在 100 只以内
        except:
            print(">>> 列表获取失败，使用预设池...")
            valid_stocks = pd.DataFrame([
                {'stock_code': '600519', 'stock_name': '贵州茅台'},
                {'stock_code': '000858', 'stock_name': '五粮液'},
                {'stock_code': '002703', 'stock_name': '浙江鼎力'},
                {'stock_code': '601318', 'stock_name': '中国平安'}
            ])

        end_dt = datetime.datetime.now()
        start_dt = end_dt - datetime.timedelta(days=365)
        results = []
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_stock

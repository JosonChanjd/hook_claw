import numpy as np
import pandas as pd
if not hasattr(np, 'bool8'): np.bool8 = np.bool_

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

# === 策略定义保持不变 (StrategySix, StrategyNine, StrategyTen) ===
# ... [此处省略策略代码，请使用你原有的策略定义] ...

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
        size = int(self._broker._cash * 0.9 // price // 100 * 100)
        if not self.position:
            if self.st_dir[-1] == 1 and self.cci[-1] < -100 and vol_ok and size >= 100: self.buy(size=size)
        elif self.st_dir[-1] == -1: self.position.close()

class StrategyNine(Strategy):
    bb_length = 20; bb_std = 2.0; ma_fast_len = 5; vol_threshold = 1.2
    def init(self):
        c, v = pd.Series(self.data.Close), pd.Series(self.data.Volume)
        bb = ta.bbands(c, length=self.bb_length, std=self.bb_std)
        self.bb_l, self.bb_u = self.I(lambda: bb.iloc[:, 0]), self.I(lambda: bb.iloc[:, 2])
        self.ma_fast, self.vol_ma5 = self.I(ta.sma, c, self.ma_fast_len), self.I(ta.sma, v, 5)
    def next(self):
        price, vol = self.data.Close[-1], self.data.Volume[-1]
        vol_ok = vol > self.vol_ma5[-1] * self.vol_threshold
        size = int(self._broker._cash * 0.9 // price // 100 * 100)
        if not self.position:
            if price < self.bb_l[-1] and price < self.ma_fast[-1] and vol_ok and size >= 100: self.buy(size=size)
        elif price > self.bb_u[-1]: self.position.close()

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
        size = int(self._broker._cash * 0.9 // price // 100 * 100)
        if not self.position:
            if price > self.span_a[-1] and price > self.span_b[-1] and vol_ok and size >= 100: self.buy(size=size)
        elif price < self.span_a[-1] or price < self.span_b[-1]: self.position.close()

# === 适配 yfinance 的数据抓取函数 ===
def get_data_yf(symbol, start_dt, end_dt):
    yf_code = f"{symbol}.SS" if symbol.startswith('6') else f"{symbol}.SZ"
    try:
        # yfinance 获取数据
        df = yf.download(yf_code, start=start_dt, end=end_dt, progress=False)
        if df.empty or len(df) < 30: return None
        # yfinance 返回的数据可能是 MultiIndex，需要打平
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
    except:
        return None

def get_signal_status(strategy_instance):
    try:
        data = strategy_instance.data
        idx = -1
        price, vol = data.Close[idx], data.Volume[idx]
        if isinstance(strategy_instance, StrategySix):
            vol_ok = vol > strategy_instance.vol_ma5[idx] * strategy_instance.vol_threshold
            return (strategy_instance.st_dir[idx] == 1 and strategy_instance.cci[idx] < -100 and vol_ok), ""
        elif isinstance(strategy_instance, StrategyNine):
            vol_ok = vol > strategy_instance.vol_ma5[idx] * strategy_instance.vol_threshold
            return (price < strategy_instance.bb_l[idx] and price < strategy_instance.ma_fast[idx] and vol_ok), ""
        elif isinstance(strategy_instance, StrategyTen):
            vol_ok = vol > strategy_instance.vol_ma5[idx] * strategy_instance.vol_threshold
            sa, sb = strategy_instance.span_a[idx], strategy_instance.span_b[idx]
            return (price > sa and price > sb and vol_ok), ""
    except: return False, ""

def process_stock(stock_info, start_date, end_date):
    symbol, name = stock_info['stock_code'], stock_info['stock_name']
    df = get_data_yf(symbol, start_date, end_date)
    if df is None: return None
    try:
        bt6, bt9, bt10 = Backtest(df, StrategySix), Backtest(df, StrategyNine), Backtest(df, StrategyTen)
        s6, s9, s10 = bt6.run(), bt9.run(), bt10.run()
        score = sum([get_signal_status(s6['_strategy'])[0], get_signal_status(s9['_strategy'])[0], get_signal_status(s10['_strategy'])[0]])
        return {'code': symbol, 'name': name, 'close': df['Close'].iloc[-1], 'score': score, 's6_ret': s6['Return [%]']}
    except: return None

def run_scanner():
    try:
        print(">>> 正在尝试获取股票列表...")
        try:
            all_stocks = ef.stock.get_realtime_quotes()
            valid_stocks = all_stocks[all_stocks['股票代码'].str.match(r'^(60|00|30|68)')].copy()
            valid_stocks.rename(columns={'股票代码': 'stock_code', '股票名称': 'stock_name'}, inplace=True)
            valid_stocks = valid_stocks.head(200) # 示例：GitHub Actions 有运行时间限制，先跑200只
        except:
            print(">>> 实时列表获取失败，切换到备用核心池...")
            valid_stocks = pd.DataFrame([{'stock_code': '600519', 'stock_name': '贵州茅台'}, {'stock_code': '000858', 'stock_name': '五粮液'}, {'stock_code': '002703', 'stock_name': '浙江鼎力'}])

        end_dt = datetime.datetime.now()
        start_dt = end_dt - datetime.timedelta(days=365)
        
        results = []
        with ThreadPoolExecutor(max_workers=5) as executor: # yfinance限制并发，不宜过高
            future_to_stock = {executor.submit(process_stock, row, start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')): row['stock_code'] for _, row in valid_stocks.iterrows()}
            for future in tqdm(as_completed(future_to_stock), total=len(valid_stocks)):
                res = future.result()
                if res and res['score'] >= 1: results.append(res)
        
        if not results: return "✅ 扫描完成，今日无买入信号。"
        
        df_res = pd.DataFrame(results).sort_values(by=['score', 's6_ret'], ascending=False)
        msg = f"✅ yfinance 扫描完成！发现 {len(df_res[df_res['score']>=2])} 只优选股票\n"
        msg += df_res[['code', 'name', 'close', 'score']].head(10).to_string(index=False)
        return msg
    except Exception as e:
        return f"❌ 扫描崩溃: {str(e)[:100]}"

if __name__ == "__main__":
    report_msg = run_scanner()
    with open("feishu_msg_1.txt", "w", encoding="utf-8") as f: f.write(report_msg)

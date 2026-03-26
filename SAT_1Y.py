import numpy as np
import pandas as pd
if not hasattr(np, 'bool8'): np.bool8 = np.bool_

import os
import datetime
import pandas_ta as ta
import efinance as ef
from backtesting import Backtest, Strategy
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

warnings.filterwarnings('ignore')

# === 策略定义保持不变 ===
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

# === 辅助函数保持不变 ===
def get_signal_status(strategy_instance):
    try:
        data = strategy_instance.data
        if len(data) < 20: return False, "数据不足"
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
            if pd.isna(sa) or pd.isna(sb): return False, ""
            return (price > sa and price > sb and vol_ok), ""
    except Exception:
        return False, "Error"

def process_stock(stock_info, start_date, end_date):
    symbol, name = stock_info['stock_code'], stock_info['stock_name']
    try:
        df = ef.stock.get_quote_history(symbol, beg=start_date, end=end_date)
        if df is None or len(df) < 50: return None
        df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量']]
        df.columns =['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df[~df.index.duplicated()] 
        
        bt6 = Backtest(df, StrategySix, cash=100000, commission=0.0003)
        bt9 = Backtest(df, StrategyNine, cash=100000, commission=0.0003)
        bt10 = Backtest(df, StrategyTen, cash=100000, commission=0.0003)
        stats6, stats9, stats10 = bt6.run(), bt9.run(), bt10.run()
        
        sig6 = get_signal_status(stats6['_strategy'])
        sig9 = get_signal_status(stats9['_strategy'])
        sig10 = get_signal_status(stats10['_strategy'])
        
        score = sum([sig6[0], sig9[0], sig10[0]])
        return {'code': symbol, 'name': name, 'close': df['Close'].iloc[-1], 'score': score, 's6_ret': stats6['Return [%]']}
    except Exception:
        return None

def run_scanner(limit=None):
    # 新增：包裹在一个大 try...except 中，用于生成友好的飞书消息
    try:
        all_stocks = ef.stock.get_realtime_quotes()
        if all_stocks is None or len(all_stocks) == 0:
            return "❌ 获取股票列表失败，可能由于 GitHub 海外 IP 被东方财富拦截。"

        valid_stocks = all_stocks[all_stocks['股票代码'].str.match(r'^(60|00|30|68)')].copy()
        valid_stocks.rename(columns={'股票代码': 'stock_code', '股票名称': 'stock_name'}, inplace=True)
        if limit: valid_stocks = valid_stocks.head(limit)
        
        end_date = datetime.datetime.now().strftime("%Y%m%d")
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y%m%d")
        results =[]
        
        with ThreadPoolExecutor(max_workers=16) as executor:
            future_to_stock = {executor.submit(process_stock, row, start_date, end_date): row['stock_code'] for _, row in valid_stocks.iterrows()}
            for future in tqdm(as_completed(future_to_stock), total=len(valid_stocks)):
                res = future.result()
                if res and res['score'] >= 1: results.append(res)
        
        if not results:
            return "✅ 扫描完成，但今日没有符合任何买入条件的股票。"

        df_res = pd.DataFrame(results)
        df_res.sort_values(by=['score', 's6_ret'], ascending=[False, False], inplace=True)
        final_picks = df_res[df_res['score'] >= 2]
        
        # 将结果保存为 CSV 供下载
        df_res.to_csv("A_Share_Scan_Result.csv", index=False, encoding='utf-8-sig')

        # 构造人类可读的飞书消息
        msg = f"✅ 全市场扫描完成！共发现 {len(final_picks)} 只综合优选股票 (满足>=2个策略)\n\n"
        msg += "🏆 前 10 名股票列表 (按得分和收益率排序)：\n"
        msg += df_res[['code', 'name', 'close', 'score', 's6_ret']].head(10).to_string(index=False)
        return msg

    except Exception as e:
        error_info = str(e)
        if "Max retries exceeded" in error_info or "Connection" in error_info:
            return "❌ 扫描失败：网络连接被拒绝。原因通常是 GitHub 服务器的海外 IP 被东方财富反爬虫系统拦截。"
        return f"❌ 扫描崩溃，发生未知错误：\n{error_info[:200]}"

if __name__ == "__main__":
    # 运行逻辑并把人类可读的结果写入文本文件
    report_msg = run_scanner(limit=None)
    with open("feishu_msg_1.txt", "w", encoding="utf-8") as f:
        f.write(report_msg)
    print("SAT_1Y.py 运行完毕，消息已写入 feishu_msg_1.txt")

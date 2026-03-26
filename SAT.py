import numpy as np
import pandas as pd
# Numpy 兼容性补丁
if not hasattr(np, 'bool8'): np.bool8 = np.bool_

import os
import pandas_ta as ta
import yfinance as yf
from backtesting import Backtest, Strategy
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 定义三个核心策略类 (保持原有逻辑)
# ==========================================

# --- 策略六 (SuperTrend + CCI + Vol) ---
class StrategySix(Strategy):
    st_period = 10
    st_mult = 3.0
    cci_period = 14
    vol_threshold = 1.2

    def init(self):
        c, h, l, v = pd.Series(self.data.Close), pd.Series(self.data.High), \
                     pd.Series(self.data.Low), pd.Series(self.data.Volume)
        st = ta.supertrend(high=h, low=l, close=c, length=int(self.st_period), multiplier=float(self.st_mult))
        self.st_dir = self.I(lambda: st.iloc[:, 1]) 
        self.cci = self.I(ta.cci, h, l, c, length=int(self.cci_period))
        self.vol_ma5 = self.I(ta.sma, v, length=5)

    def next(self):
        price, vol = self.data.Close[-1], self.data.Volume[-1]
        vol_ok = vol > self.vol_ma5[-1] * self.vol_threshold
        size = int(self._broker._cash * 0.9 // price // 100 * 100)
        if not self.position:
            if self.st_dir[-1] == 1 and self.cci[-1] < -100 and vol_ok:
                if size >= 100: self.buy(size=size)
        else:
            if self.st_dir[-1] == -1: self.position.close()

# --- 策略九 (布林带反转 + Vol) ---
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
            if price < self.bb_l[-1] and price < self.ma_fast[-1] and vol_ok:
                if size >= 100: self.buy(size=size)
        else:
            if price > self.bb_u[-1]: self.position.close()

# --- 策略十 (一目均衡云 + Vol) ---
class StrategyTen(Strategy):
    vol_threshold = 1.2
    def init(self):
        c, h, l, v = pd.Series(self.data.Close), pd.Series(self.data.High), \
                     pd.Series(self.data.Low), pd.Series(self.data.Volume)
        ichi = ta.ichimoku(h, l, c)[0]
        self.span_a, self.span_b = self.I(lambda: ichi.iloc[:, 2]), self.I(lambda: ichi.iloc[:, 3])
        self.vol_ma5 = self.I(ta.sma, v, 5)

    def next(self):
        price, vol = self.data.Close[-1], self.data.Volume[-1]
        vol_ok = vol > self.vol_ma5[-1] * self.vol_threshold
        size = int(self._broker._cash * 0.9 // price // 100 * 100)
        if not self.position:
            if price > self.span_a[-1] and price > self.span_b[-1] and vol_ok:
                if size >= 100: self.buy(size=size)
        else:
            if price < self.span_a[-1] or price < self.span_b[-1]: self.position.close()

# ==========================================
# 2. 辅助工具函数
# ==========================================

def get_signal_status(strategy_instance):
    data, idx = strategy_instance.data, -1
    price, vol = data.Close[idx], data.Volume[idx]
    try:
        if isinstance(strategy_instance, StrategySix):
            vol_ok = vol > strategy_instance.vol_ma5[idx] * strategy_instance.vol_threshold
            cond = (strategy_instance.st_dir[idx] == 1 and strategy_instance.cci[idx] < -100 and vol_ok)
            return cond, f"ST={strategy_instance.st_dir[idx]}, CCI={strategy_instance.cci[idx]:.1f}"
        elif isinstance(strategy_instance, StrategyNine):
            vol_ok = vol > strategy_instance.vol_ma5[idx] * strategy_instance.vol_threshold
            cond = (price < strategy_instance.bb_l[idx] and price < strategy_instance.ma_fast[idx] and vol_ok)
            return cond, f"Price < LowerBB: {price < strategy_instance.bb_l[idx]}"
        elif isinstance(strategy_instance, StrategyTen):
            vol_ok = vol > strategy_instance.vol_ma5[idx] * strategy_instance.vol_threshold
            sa, sb = strategy_instance.span_a[idx], strategy_instance.span_b[idx]
            cond = (price > sa and price > sb and vol_ok)
            return cond, f"Above Cloud: {cond}"
    except: return False, "Error"
    return False, ""

def inject_combined_dashboard(filename, symbol, stats_map, signals):
    """注入 HTML 看板逻辑保持不变，但移除了自动打开浏览器"""
    rows_html = ""
    for name, stats in stats_map.items():
        is_buy, detail = signals[name]
        signal_text = "<b>★ 买入</b>" if is_buy else "观望"
        rows_html += f"<tr><td>{name}</td><td>{stats['Return [%]']:.2f}%</td><td>{signal_text}</td><td>{detail}</td></tr>"
    
    buy_count = sum([s[0] for s in signals.values()])
    dashboard = f"<div style='background:#f0f2f5; padding:20px;'><h2>{symbol} 综合评分: {buy_count}/3</h2><table border='1'>{rows_html}</table></div>"
    
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f: html = f.read()
        with open(filename, 'w', encoding='utf-8') as f: f.write(html.replace('</body>', dashboard + '</body>'))

# ==========================================
# 3. 主程序 (适配 yfinance)
# ==========================================

def run_combined_system(symbol):
    print(f"\n>>> 正在启动 yfinance 模式分析: {symbol}")
    
    # 1. 转换代码格式
    yf_code = f"{symbol}.SS" if symbol.startswith(('6', '9')) else f"{symbol}.SZ"
    
    try:
        # 2. 下载数据 (取近2年)
        df = yf.download(yf_code, period="2y", progress=False)
        if df.empty or len(df) < 50:
            return f"❌ {symbol} 数据获取失败。请检查代码是否正确或 yfinance 服务状态。"
        
        # 3. 清洗 yfinance 数据格式
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        
        # 4. 运行回测
        bt6 = Backtest(df, StrategySix, cash=1000000, commission=0.0003)
        stats6 = bt6.run()
        bt9 = Backtest(df, StrategyNine, cash=1000000, commission=0.0003)
        stats9 = bt9.run()
        bt10 = Backtest(df, StrategyTen, cash=1000000, commission=0.0003)
        stats10 = bt10.run()

        # 5. 提取信号
        sig6 = get_signal_status(stats6['_strategy'])
        sig9 = get_signal_status(stats9['_strategy'])
        sig10 = get_signal_status(stats10['_strategy'])

        stats_map = {"S6": stats6, "S9": stats9, "S10": stats10}
        signals_map = {"S6": sig6, "S9": sig9, "S10": sig10}

        # 6. 生成简报
        buy_count = sum([sig6[0], sig9[0], sig10[0]])
        last_price = df['Close'].iloc[-1]
        
        msg = f"📊 单股分析报告: {symbol} ({yf_code})\n"
        msg += f"最新收盘: {last_price:.2f} ({df.index[-1].date()})\n"
        msg += "------------------------\n"
        msg += f"S6 SuperTrend: {'🟢 买入' if sig6[0] else '⚪ 观望'} (年化收益: {stats6['Return [%]']:.1f}%)\n"
        msg += f"S9 Bollinger : {'🟢 买入' if sig9[0] else '⚪ 观望'} (年化收益: {stats9['Return [%]']:.1f}%)\n"
        msg += f"S10 Ichimoku : {'🟢 买入' if sig10[0] else '⚪ 观望'} (年化收益: {stats10['Return [%]']:.1f}%)\n"
        msg += "------------------------\n"
        msg += f"💡 综合决策: {'🔥 建议买入' if buy_count >= 2 else '⏳ 建议观望'} ({buy_count}/3)"

        # 7. 生成本地报告 (供 Artifacts 下载)
        report_dir = "combined_reports"
        if not os.path.exists(report_dir): os.makedirs(report_dir)
        report_path = os.path.join(report_dir, f"Combined_{symbol}.html")
        bt6.plot(filename=report_path, open_browser=False)
        inject_combined_dashboard(report_path, symbol, stats_map, signals_map)

        return msg

    except Exception as e:
        return f"❌ {symbol} 分析过程中发生错误:\n{str(e)[:200]}"

if __name__ == "__main__":
    # 执行分析
    final_report = run_combined_system("002703")
    
    # 写入文件供 GitHub Action 读取
    with open("feishu_msg_2.txt", "w", encoding="utf-8") as f:
        f.write(final_report)
    
    print("SAT.py 运行完毕。")

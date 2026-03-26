import numpy as np
import pandas as pd
if not hasattr(np, 'bool8'): np.bool8 = np.bool_

import os
import pandas_ta as ta
import efinance as ef
from backtesting import Backtest, Strategy
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. 定义三个核心策略类
# ==========================================

# --- 策略六 (SuperTrend + CCI + Vol) ---
class StrategySix(Strategy):
    st_period = 10
    st_mult = 3.0
    cci_period = 14
    vol_threshold = 1.2

    def init(self):
        c = pd.Series(self.data.Close)
        h = pd.Series(self.data.High)
        l = pd.Series(self.data.Low)
        v = pd.Series(self.data.Volume)

        # SuperTrend
        st = ta.supertrend(high=h, low=l, close=c, length=int(self.st_period), multiplier=float(self.st_mult))
        self.st_dir = self.I(lambda: st.iloc[:, 1]) 
        
        # CCI
        self.cci = self.I(ta.cci, h, l, c, length=int(self.cci_period))
        
        # Volume MA
        self.vol_ma5 = self.I(ta.sma, v, length=5)

    def next(self):
        price = self.data.Close[-1]
        vol = self.data.Volume[-1]
        vol_ok = vol > self.vol_ma5[-1] * self.vol_threshold
        size = int(self._broker._cash * 0.9 // price // 100 * 100)

        if not self.position:
            if self.st_dir[-1] == 1 and self.cci[-1] < -100 and vol_ok:
                if size >= 100: self.buy(size=size)
        else:
            if self.st_dir[-1] == -1:
                self.position.close()

# --- 策略九 (布林带反转 + Vol) ---
class StrategyNine(Strategy):
    bb_length = 20
    bb_std = 2.0
    ma_fast_len = 5
    vol_threshold = 1.2

    def init(self):
        c, v = pd.Series(self.data.Close), pd.Series(self.data.Volume)
        
        # 布林带
        bb = ta.bbands(c, length=self.bb_length, std=self.bb_std)
        self.bb_l = self.I(lambda: bb.iloc[:, 0]) # 下轨
        self.bb_u = self.I(lambda: bb.iloc[:, 2]) # 上轨
        
        # 5日均线
        self.ma_fast = self.I(ta.sma, c, self.ma_fast_len)
        self.vol_ma5 = self.I(ta.sma, v, 5)

    def next(self):
        price = self.data.Close[-1]
        vol = self.data.Volume[-1]
        vol_ok = vol > self.vol_ma5[-1] * self.vol_threshold
        size = int(self._broker._cash * 0.9 // price // 100 * 100)

        if not self.position:
            # 价格 < 下轨 且 价格 < MA5 且 放量
            if price < self.bb_l[-1] and price < self.ma_fast[-1] and vol_ok:
                if size >= 100: self.buy(size=size)
        else:
            if price > self.bb_u[-1]:
                self.position.close()

# --- 策略十 (一目均衡云 + Vol) ---
class StrategyTen(Strategy):
    vol_threshold = 1.2

    def init(self):
        c, h, l, v = pd.Series(self.data.Close), pd.Series(self.data.High), \
                     pd.Series(self.data.Low), pd.Series(self.data.Volume)
        
        # Ichimoku
        ichi = ta.ichimoku(h, l, c)[0]
        self.span_a = self.I(lambda: ichi.iloc[:, 2]) # 先行带A
        self.span_b = self.I(lambda: ichi.iloc[:, 3]) # 先行带B
        self.vol_ma5 = self.I(ta.sma, v, 5)

    def next(self):
        price = self.data.Close[-1]
        vol = self.data.Volume[-1]
        vol_ok = vol > self.vol_ma5[-1] * self.vol_threshold
        size = int(self._broker._cash * 0.9 // price // 100 * 100)

        if not self.position:
            # 价格在云层之上 且 放量
            if price > self.span_a[-1] and price > self.span_b[-1] and vol_ok:
                if size >= 100: self.buy(size=size)
        else:
            if price < self.span_a[-1] or price < self.span_b[-1]:
                self.position.close()

# ==========================================
# 2. 信号检测与HTML注入函数
# ==========================================
def get_signal_status(strategy_instance):
    """
    解析策略实例的最后一根K线状态，返回是否满足买入条件
    """
    data = strategy_instance.data
    idx = -1 # 最新一根K线
    price = data.Close[idx]
    vol = data.Volume[idx]
    
    is_buy = False
    desc = ""
    
    try:
        if isinstance(strategy_instance, StrategySix):
            vol_ok = vol > strategy_instance.vol_ma5[idx] * strategy_instance.vol_threshold
            st_cond = strategy_instance.st_dir[idx] == 1
            cci_cond = strategy_instance.cci[idx] < -100
            is_buy = st_cond and cci_cond and vol_ok
            desc = f"SuperTrend={'多' if st_cond else '空'}, CCI={strategy_instance.cci[idx]:.1f}, Vol={vol_ok}"
            
        elif isinstance(strategy_instance, StrategyNine):
            vol_ok = vol > strategy_instance.vol_ma5[idx] * strategy_instance.vol_threshold
            bb_cond = price < strategy_instance.bb_l[idx]
            ma_cond = price < strategy_instance.ma_fast[idx]
            is_buy = bb_cond and ma_cond and vol_ok
            desc = f"价<布林下轨={bb_cond}, 价<MA5={ma_cond}, Vol={vol_ok}"
            
        elif isinstance(strategy_instance, StrategyTen):
            vol_ok = vol > strategy_instance.vol_ma5[idx] * strategy_instance.vol_threshold
            # Span A/B may be NaN early on, handle carefully
            sa = strategy_instance.span_a[idx]
            sb = strategy_instance.span_b[idx]
            cloud_cond = price > sa and price > sb
            is_buy = cloud_cond and vol_ok
            desc = f"价>云层={cloud_cond}, Vol={vol_ok}"
            
    except Exception as e:
        desc = f"计算错误: {e}"
        
    return is_buy, desc

def inject_combined_dashboard(filename, symbol, stats_map, signals):
    """
    将三个策略的统计结果和综合信号注入到 HTML 中
    """
    # 1. 汇总统计表格行
    rows_html = ""
    for name, stats in stats_map.items():
        is_buy, detail = signals[name]
        signal_color = "#ff4d4f" if is_buy else "#d9d9d9"
        signal_text = "<b>★ 买入</b>" if is_buy else "观望"
        
        rows_html += f"""
        <tr>
            <td style="font-weight:bold;">{name}</td>
            <td>{stats['Return [%]']:.2f}%</td>
            <td>{stats['Win Rate [%]']:.2f}%</td>
            <td>{stats['Max. Drawdown [%]']:.2f}%</td>
            <td><span style="background:{signal_color}; color:white; padding:4px 8px; border-radius:4px;">{signal_text}</span></td>
            <td style="font-size:0.85em; color:#666;">{detail}</td>
        </tr>
        """

    # 2. 计算综合结果
    buy_count = sum([s[0] for s in signals.values()])
    final_decision = buy_count >= 2
    
    final_color = "#f5222d" if final_decision else "#52c41a"
    final_text = f"建议买入 (满足 {buy_count}/3)" if final_decision else f"建议观望 (满足 {buy_count}/3)"
    
    last_date = list(stats_map.values())[0]['_strategy'].data.index[-1]
    last_price = list(stats_map.values())[0]['_strategy'].data.Close[-1]

    # 3. 构造仪表盘 HTML
    dashboard = f"""
    <hr style="margin-top: 50px;">
    <div id="combined-dashboard" style="padding: 20px; font-family: 'Helvetica Neue', Arial, sans-serif; background-color: #f0f2f5;">
        <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);">
            <div style="display:flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #f0f0f0; padding-bottom: 15px; margin-bottom: 20px;">
                <div>
                    <h2 style="margin:0; color: #333;">全能策略综合看板 - {symbol}</h2>
                    <p style="margin:5px 0 0 0; color: #888;">最新数据时间: {last_date} | 收盘价: {last_price:.2f}</p>
                </div>
                <div style="background: {final_color}; color: white; padding: 15px 30px; border-radius: 8px; text-align: center;">
                    <h3 style="margin:0; font-size: 1.2em;">综合决策 (>=2)</h3>
                    <p style="margin:5px 0 0 0; font-size: 1.5em; font-weight: bold;">{final_text}</p>
                </div>
            </div>

            <h3 style="color: #555;">单策略表现与实时信号</h3>
            <table class="strategy-table">
                <thead>
                    <tr>
                        <th>策略名称</th>
                        <th>总收益率</th>
                        <th>胜率</th>
                        <th>最大回撤</th>
                        <th>当前信号</th>
                        <th>信号逻辑详情</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
            <p style="margin-top:15px; font-size:0.9em; color:#888;">注：回测统计基于 2010-01-01 至今数据；实时信号基于最新一根 K 线判定。</p>
        </div>
    </div>
    <style>
        .strategy-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        .strategy-table th {{ background-color: #fafafa; padding: 12px; border: 1px solid #e8e8e8; text-align: left; color: #555; }}
        .strategy-table td {{ padding: 12px; border: 1px solid #e8e8e8; }}
        .strategy-table tr:hover {{ background-color: #e6f7ff; }}
    </style>
    """

    with open(filename, 'r', encoding='utf-8') as f:
        html = f.read()
    
    new_html = html.replace('</body>', dashboard + '</body>')
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(new_html)

# ==========================================
# 3. 主程序逻辑
# ==========================================
def run_combined_system(symbol, start_date="20100101", end_date="20261231"):
    print(f"\n{'='*50}")
    print(f" >>> 正在启动综合策略系统: {symbol}")
    print(f"{'='*50}")

    # 1. 获取数据 (一次获取，多次使用)
    try:
        df = ef.stock.get_quote_history(symbol, beg=start_date, end=end_date)
        df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量']]
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.dropna()
        print(f"数据获取成功: {len(df)} 条记录 (最新日期: {df.index[-1].date()})")
    except Exception as e:
        print(f"数据获取失败: {e}")
        return

    # 2. 分别运行三个策略
    print("正在运行 S6 (SuperTrend)...")
    bt6 = Backtest(df, StrategySix, cash=1000000, commission=0.0003)
    stats6 = bt6.run()

    print("正在运行 S9 (Bollinger)...")
    bt9 = Backtest(df, StrategyNine, cash=1000000, commission=0.0003)
    stats9 = bt9.run()

    print("正在运行 S10 (Ichimoku)...")
    bt10 = Backtest(df, StrategyTen, cash=1000000, commission=0.0003)
    stats10 = bt10.run()

    # 3. 提取实时信号
    signal6 = get_signal_status(stats6['_strategy'])
    signal9 = get_signal_status(stats9['_strategy'])
    signal10 = get_signal_status(stats10['_strategy'])

    # 4. 控制台输出简报
    stats_map = {
        "S6 (SuperTrend)": stats6,
        "S9 (Bollinger)": stats9,
        "S10 (Ichimoku)": stats10
    }
    signals_map = {
        "S6 (SuperTrend)": signal6,
        "S9 (Bollinger)": signal9,
        "S10 (Ichimoku)": signal10
    }

    print("\n--- 策略结果汇总 ---")
    print(f"{'策略':<15} | {'收益率':<8} | {'胜率':<8} | {'信号':<6}")
    print("-" * 50)
    buy_count = 0
    for name, stats in stats_map.items():
        is_buy = signals_map[name][0]
        if is_buy: buy_count += 1
        sig_str = "【买入】" if is_buy else "观望"
        print(f"{name:<15} | {stats['Return [%]']:.2f}%   | {stats['Win Rate [%]']:.2f}%   | {sig_str}")

    final_decision = "★★ 综合买入 ★★" if buy_count >= 2 else "观望"
    print(f"\n>>> 最终决策: {final_decision} (满足条件数: {buy_count}/3)")

    # 5. 生成综合报表
    report_dir = "combined_reports"
    if not os.path.exists(report_dir): os.makedirs(report_dir)
    report_path = os.path.abspath(os.path.join(report_dir, f"Combined_{symbol}.html"))

    # 使用 S6 生成基础图表 (因为 S6 的指标在图表上最直观)
    # plot() 会生成一个包含 S6 交易记录的 HTML
    bt6.plot(filename=report_path, open_browser=False)

    # 注入综合看板
    inject_combined_dashboard(report_path, symbol, stats_map, signals_map)

    print(f"\n✅ 完整综合报表已生成: {report_path}")
#    os.startfile(report_path)
import os
if os.name == 'nt': # 只有在 Windows 电脑上才自动打开文件
    try:
        os.startfile(report_path)
    except:
        pass

if __name__ == "__main__":
    # 输入股票代码（例如 002195 二三四五）
    run_combined_system("002703")

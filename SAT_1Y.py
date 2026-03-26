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
from tqdm import tqdm  # 进度条库

warnings.filterwarnings('ignore')

# ==========================================
# 1. 核心策略定义 (保持原有逻辑不变)
# ==========================================

# --- S6: SuperTrend ---
class StrategySix(Strategy):
    st_period = 10
    st_mult = 3.0
    cci_period = 14
    vol_threshold = 1.2
    
    def init(self):
        c, h, l, v = self.data.Close, self.data.High, self.data.Low, self.data.Volume
        st = ta.supertrend(pd.Series(h), pd.Series(l), pd.Series(c), length=int(self.st_period), multiplier=float(self.st_mult))
        self.st_dir = self.I(lambda: st.iloc[:, 1])
        self.cci = self.I(ta.cci, pd.Series(h), pd.Series(l), pd.Series(c), length=int(self.cci_period))
        self.vol_ma5 = self.I(ta.sma, pd.Series(v), length=5)

    def next(self):
        # 回测逻辑保留，用于计算收益率
        price = self.data.Close[-1]
        vol = self.data.Volume[-1]
        vol_ok = vol > self.vol_ma5[-1] * self.vol_threshold
        size = int(self._broker._cash * 0.9 // price // 100 * 100)
        if not self.position:
            if self.st_dir[-1] == 1 and self.cci[-1] < -100 and vol_ok and size >= 100:
                self.buy(size=size)
        elif self.st_dir[-1] == -1:
            self.position.close()

# --- S9: Bollinger ---
class StrategyNine(Strategy):
    bb_length = 20
    bb_std = 2.0
    ma_fast_len = 5
    vol_threshold = 1.2

    def init(self):
        c, v = pd.Series(self.data.Close), pd.Series(self.data.Volume)
        bb = ta.bbands(c, length=self.bb_length, std=self.bb_std)
        self.bb_l = self.I(lambda: bb.iloc[:, 0])
        self.bb_u = self.I(lambda: bb.iloc[:, 2])
        self.ma_fast = self.I(ta.sma, c, self.ma_fast_len)
        self.vol_ma5 = self.I(ta.sma, v, 5)

    def next(self):
        price = self.data.Close[-1]
        vol = self.data.Volume[-1]
        vol_ok = vol > self.vol_ma5[-1] * self.vol_threshold
        size = int(self._broker._cash * 0.9 // price // 100 * 100)
        if not self.position:
            if price < self.bb_l[-1] and price < self.ma_fast[-1] and vol_ok and size >= 100:
                self.buy(size=size)
        elif price > self.bb_u[-1]:
            self.position.close()

# --- S10: Ichimoku ---
class StrategyTen(Strategy):
    vol_threshold = 1.2

    def init(self):
        c, h, l, v = pd.Series(self.data.Close), pd.Series(self.data.High), pd.Series(self.data.Low), pd.Series(self.data.Volume)
        ichi = ta.ichimoku(h, l, c)[0]
        self.span_a = self.I(lambda: ichi.iloc[:, 2])
        self.span_b = self.I(lambda: ichi.iloc[:, 3])
        self.vol_ma5 = self.I(ta.sma, v, 5)

    def next(self):
        price = self.data.Close[-1]
        vol = self.data.Volume[-1]
        vol_ok = vol > self.vol_ma5[-1] * self.vol_threshold
        size = int(self._broker._cash * 0.9 // price // 100 * 100)
        if not self.position:
            if price > self.span_a[-1] and price > self.span_b[-1] and vol_ok and size >= 100:
                self.buy(size=size)
        elif price < self.span_a[-1] or price < self.span_b[-1]:
            self.position.close()

# ==========================================
# 2. 辅助函数：提取信号与注入仪表盘
# ==========================================
def get_signal_status(strategy_instance):
    """提取最新一根K线的信号状态"""
    try:
        data = strategy_instance.data
        if len(data) < 20: return False, "数据不足" # 防止新股数据过少
        
        idx = -1
        price = data.Close[idx]
        vol = data.Volume[idx]
        is_buy = False
        desc = ""
        
        if isinstance(strategy_instance, StrategySix):
            vol_ok = vol > strategy_instance.vol_ma5[idx] * strategy_instance.vol_threshold
            st_cond = strategy_instance.st_dir[idx] == 1
            cci_cond = strategy_instance.cci[idx] < -100
            is_buy = st_cond and cci_cond and vol_ok
            desc = f"ST多={st_cond}, CCI={strategy_instance.cci[idx]:.0f}, Vol={vol_ok}"
            
        elif isinstance(strategy_instance, StrategyNine):
            vol_ok = vol > strategy_instance.vol_ma5[idx] * strategy_instance.vol_threshold
            bb_cond = price < strategy_instance.bb_l[idx]
            ma_cond = price < strategy_instance.ma_fast[idx]
            is_buy = bb_cond and ma_cond and vol_ok
            desc = f"破下轨={bb_cond}, 破MA5={ma_cond}, Vol={vol_ok}"
            
        elif isinstance(strategy_instance, StrategyTen):
            vol_ok = vol > strategy_instance.vol_ma5[idx] * strategy_instance.vol_threshold
            sa = strategy_instance.span_a[idx]
            sb = strategy_instance.span_b[idx]
            if pd.isna(sa) or pd.isna(sb): return False, "云层计算中"
            cloud_cond = price > sa and price > sb
            is_buy = cloud_cond and vol_ok
            desc = f"云上={cloud_cond}, Vol={vol_ok}"
            
    except Exception as e:
        is_buy = False
        desc = f"Error: {str(e)[:10]}"
        
    return is_buy, desc

def inject_dashboard(filename, symbol, stock_name, stats_map, signals):
    """注入HTML仪表盘 (仅针对筛选出的股票)"""
    rows_html = ""
    for name, stats in stats_map.items():
        is_buy, detail = signals[name]
        signal_color = "#ff4d4f" if is_buy else "#d9d9d9"
        signal_text = "买入" if is_buy else "观望"
        rows_html += f"<tr><td>{name}</td><td>{stats['Return [%]']:.1f}%</td><td>{signal_text}</td><td style='font-size:0.8em'>{detail}</td></tr>"

    buy_count = sum([s[0] for s in signals.values()])
    
    dashboard = f"""
    <div style="padding:20px; background:#f0f2f5; font-family:Arial;">
        <h2 style="margin-top:0;">{stock_name} ({symbol}) - 综合买入评分: {buy_count}/3</h2>
        <table border="1" style="border-collapse:collapse; width:100%; background:white;">
            <tr style="background:#fafafa;"><th>策略</th><th>近1年收益</th><th>当前信号</th><th>详情</th></tr>
            {rows_html}
        </table>
    </div>
    """
    with open(filename, 'r', encoding='utf-8') as f: html = f.read()
    with open(filename, 'w', encoding='utf-8') as f: f.write(html.replace('</body>', dashboard + '</body>'))

# ==========================================
# 3. 单只股票处理逻辑
# ==========================================
def process_stock(stock_info, start_date, end_date):
    """
    处理单个股票：获取数据 -> 运行3个策略 -> 返回结果
    """
    symbol = stock_info['stock_code']
    name = stock_info['stock_name']
    
    try:
        # 1. 获取数据 (仅获取近1-2年，提高速度)
        df = ef.stock.get_quote_history(symbol, beg=start_date, end=end_date)
        if df is None or len(df) < 50: return None
        
        df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量']]
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        # 去重，防止efinance返回重复数据
        df = df[~df.index.duplicated()] 
        
        # 2. 运行三个策略
        # 必须重新实例化 Backtest
        bt6 = Backtest(df, StrategySix, cash=100000, commission=0.0003)
        stats6 = bt6.run()
        
        bt9 = Backtest(df, StrategyNine, cash=100000, commission=0.0003)
        stats9 = bt9.run()
        
        bt10 = Backtest(df, StrategyTen, cash=100000, commission=0.0003)
        stats10 = bt10.run()
        
        # 3. 获取信号
        sig6 = get_signal_status(stats6['_strategy'])
        sig9 = get_signal_status(stats9['_strategy'])
        sig10 = get_signal_status(stats10['_strategy'])
        
        buy_count = sum([sig6[0], sig9[0], sig10[0]])
        
        # 4. 如果满足条件(>=2)，保存用于生成的对象，否则只返回统计数据
        result = {
            'code': symbol,
            'name': name,
            'close': df['Close'].iloc[-1],
            'score': buy_count,
            's6_signal': sig6[0],
            's9_signal': sig9[0],
            's10_signal': sig10[0],
            's6_ret': stats6['Return [%]'],
            's6_obj': bt6,      # 保存回测对象用于绘图
            'stats_map': {"S6": stats6, "S9": stats9, "S10": stats10},
            'signals_map': {"S6": sig6, "S9": sig9, "S10": sig10}
        }
        return result

    except Exception as e:
        # print(f"Error {symbol}: {e}")
        return None

# ==========================================
# 4. 全市场扫描主程序
# ==========================================
def run_scanner(limit=None):
    print(">>> 正在获取A股全市场股票列表...")
    all_stocks = ef.stock.get_realtime_quotes()
    # 过滤掉北交所(可选)和指数，仅保留沪深主板/创业/科创
    # 简单的筛选逻辑：代码为60, 00, 30, 68开头
    valid_stocks = all_stocks[all_stocks['股票代码'].str.match(r'^(60|00|30|68)')].copy()
    valid_stocks.rename(columns={'股票代码': 'stock_code', '股票名称': 'stock_name'}, inplace=True)
    
    # 测试模式：限制数量
    if limit:
        valid_stocks = valid_stocks.head(limit)
    
    print(f">>> 开始扫描 {len(valid_stocks)} 只股票 (回测周期: 近1年)")
    print(">>> 注意：扫描全市场可能需要 5-20 分钟，请耐心等待...")
    
    # 设置时间范围：过去365天到今天
    end_date = datetime.datetime.now().strftime("%Y%m%d")
    start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime("%Y%m%d")
    
    results = []
    
    # 使用线程池并发处理
    # max_workers 建议根据CPU核心数调整，I/O密集型任务可以设大一点，如 10-20
    with ThreadPoolExecutor(max_workers=16) as executor:
        # 提交任务
        future_to_stock = {
            executor.submit(process_stock, row, start_date, end_date): row['stock_code'] 
            for index, row in valid_stocks.iterrows()
        }
        
        # 使用tqdm显示进度条
        for future in tqdm(as_completed(future_to_stock), total=len(valid_stocks), unit="stock"):
            res = future.result()
            if res and res['score'] >= 1: # 只收集至少满足1个条件的，减少内存占用
                results.append(res)
    
    # 结果处理
    if not results:
        print("没有找到符合条件的股票。")
        return

    df_res = pd.DataFrame(results)
    
    # 按满足条件数(Score)降序排序，其次按S6收益率排序
    df_res.sort_values(by=['score', 's6_ret'], ascending=[False, False], inplace=True)
    
    # 筛选最终名单 (score >= 2)
    final_picks = df_res[df_res['score'] >= 2]
    
    print(f"\n{'='*60}")
    print(f"扫描完成！共发现 {len(final_picks)} 只综合优选股票 (Score >= 2)")
    print(f"{'='*60}")
    
    # 打印前 10 名
    display_cols = ['code', 'name', 'close', 'score', 's6_signal', 's9_signal', 's10_signal', 's6_ret']
    print(df_res[display_cols].head(10).to_string(index=False))
    
    # 1. 保存CSV
    csv_path = "A_Share_Scan_Result.csv"
    df_res[display_cols].to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n[文件] 完整扫描结果已保存至: {csv_path}")
    
    # 2. 生成 HTML 报告 (仅针对 Top 5 或所有 Score=3 的股票，防止生成太多)
    report_dir = "Scan_Reports"
    if not os.path.exists(report_dir): os.makedirs(report_dir)
    
    print("\n>>> 正在为优选股票生成详细图表...")
    count = 0
    for idx, row in final_picks.iterrows():
        # 限制生成数量，防止卡死浏览器，例如只生成前20个
        if count >= 200: break 
        
        try:
            filename = os.path.join(report_dir, f"{row['score']}分_{row['code']}_{row['name']}.html")
            # 使用 S6 对象绘图
            row['s6_obj'].plot(filename=filename, open_browser=False)
            # 注入看板
            inject_dashboard(filename, row['code'], row['name'], row['stats_map'], row['signals_map'])
            count += 1
            print(f"  已生成: {filename}")
        except Exception as e:
            print(f"  生成失败 {row['code']}: {e}")

    print(f"\n>>> 全部完成！请查看 {report_dir} 文件夹中的图表。")

if __name__ == "__main__":
    # limit=None 表示跑全市场。如果只想测试，可以填 limit=50
    run_scanner(limit=10)

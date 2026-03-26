import numpy as np
import pandas as pd
if not hasattr(np, 'bool8'): np.bool8 = np.bool_

import os
import pandas_ta as ta
import efinance as ef
from backtesting import Backtest, Strategy
import warnings

warnings.filterwarnings('ignore')

# === 策略定义保持不变，此处为节省篇幅折叠，请保留你原来的策略逻辑 ===
# (请将你原来的 StrategySix, StrategyNine, StrategyTen, get_signal_status, inject_combined_dashboard 完整粘贴在这里)
# ... （省略的策略代码，和原版一致）...

# 为了完整性，这里补全精简版，你需要用你原文件的这部分覆盖：
class StrategySix(Strategy): ...
class StrategyNine(Strategy): ...
class StrategyTen(Strategy): ...
def get_signal_status(strategy_instance): ...
def inject_combined_dashboard(filename, symbol, stats_map, signals): ...

# === 修改主程序 ===
def run_combined_system(symbol, start_date="20100101", end_date="20261231"):
    try:
        df = ef.stock.get_quote_history(symbol, beg=start_date, end=end_date)
        if df is None or len(df) < 50:
            return f"❌ {symbol} 数据获取失败 (可能触发了防爬虫限制或数据不足)。"

        df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量']]
        df.columns =['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df = df.dropna()

        # 运行回测
        bt6 = Backtest(df, StrategySix, cash=1000000, commission=0.0003)
        stats6 = bt6.run()
        bt9 = Backtest(df, StrategyNine, cash=1000000, commission=0.0003)
        stats9 = bt9.run()
        bt10 = Backtest(df, StrategyTen, cash=1000000, commission=0.0003)
        stats10 = bt10.run()

        # 提取信号
        signal6 = get_signal_status(stats6['_strategy'])
        signal9 = get_signal_status(stats9['_strategy'])
        signal10 = get_signal_status(stats10['_strategy'])

        stats_map = {"S6 (SuperTrend)": stats6, "S9 (Bollinger)": stats9, "S10 (Ichimoku)": stats10}
        signals_map = {"S6 (SuperTrend)": signal6, "S9 (Bollinger)": signal9, "S10 (Ichimoku)": signal10}

        buy_count = sum([signal6[0], signal9[0], signal10[0]])
        final_decision = "★★ 综合买入 ★★" if buy_count >= 2 else "观望"

        # 生成人类可读的简报
        msg = f"📊 单股分析报告: {symbol}\n"
        msg += f"最新收盘价: {df['Close'].iloc[-1]:.2f} ({df.index[-1].date()})\n"
        msg += "------------------------\n"
        for name, stats in stats_map.items():
            is_buy = signals_map[name][0]
            sig_str = "🟢 【买入】" if is_buy else "⚪ 观望"
            msg += f"{name}: {sig_str} (总收益: {stats['Return [%]']:.1f}%)\n"
        msg += "------------------------\n"
        msg += f"💡 最终决策: {final_decision} (满足条件数: {buy_count}/3)"

        return msg

    except Exception as e:
        error_info = str(e)
        if "Max retries exceeded" in error_info or "Connection" in error_info:
             return f"❌ {symbol} 单股分析失败：网络连接被拒绝 (GitHub 节点 IP 被拉黑)。"
        return f"❌ {symbol} 单股分析发生未知错误：\n{error_info[:200]}"

if __name__ == "__main__":
    report_msg = run_combined_system("002703")
    with open("feishu_msg_2.txt", "w", encoding="utf-8") as f:
        f.write(report_msg)
    print("SAT.py 运行完毕，消息已写入 feishu_msg_2.txt")

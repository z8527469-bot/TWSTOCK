"""全市場台股波段量化選股掃描器 (Full-Market TW Stock Screener)

- 標的來源: 自動動態抓取 TWSE (上市) 與 TPEx (上櫃) 全部 4 碼普通股 (約 1,800+ 檔)
- 併發架構: 分塊批量下載 (Batch Chunking)，兼顧下載速度與防 429 速率限制
- 整合策略: VCP 壓縮突破、強勢股回後買、底部扭轉底底高
- 輸出: 自動生成 docs/data.json，直供 GitHub Pages 儀表板呈現
"""

from datetime import datetime, timezone, timedeltaimport json
import io
import os
import re
import time
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ==================== 參數配置 ====================
OUTPUT_DIR = "docs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "data.json")

# 篩選門檻
MIN_PRICE = 15.0  # 過濾雞蛋水餃股
MIN_VOLUME_MA20 = 500  # 20日均量門檻 (張)
CHUNK_SIZE = 60  # 每批次下載股票數量
REQUEST_SLEEP = 1.0  # 批次間隔秒數，保護連線不被封鎖

# 若想測試可設為數字 (例如 120)，正式運行請設為 None (全市場掃描)
SCAN_LIMIT: Optional[int] = None


# ==================== 1. 動態抓取上市櫃全市場股票清單 ====================
def fetch_tw_stock_universe() -> Dict[str, str]:
  """自 TWSE ISIN 爬取所有上市 (strMode=2) 與上櫃 (strMode=4) 的普通股名單

  Returns:
      dict: { '2330.TW': '台積電', '8069.TWO': '元太', ... }
  """
  headers = {
      "User-Agent": (
          "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
      )
  }
  market_configs = [
      ("https://isin.twse.com.tw/isin/C_public.jsp?strMode=2", ".TW"),  # 上市
      ("https://isin.twse.com.tw/isin/C_public.jsp?strMode=4", ".TWO"),  # 上櫃
  ]

  stock_dict = {}
  pattern = re.compile(r"^([1-9]\d{3})\s+(.+)$")  # 僅匹配標準 4 碼普通股

  for url, suffix in market_configs:
    try:
      resp = requests.get(url, headers=headers, timeout=15)
      resp.encoding = "big5-hkscs"
      dfs = pd.read_html(io.StringIO(resp.text))
      if not dfs:
        continue

      table = dfs[0]
      # 第 0 欄通常為「有價證券代號及名稱」
      for raw_val in table.iloc[:, 0].dropna():
        raw_val = str(raw_val).strip()
        match = pattern.match(raw_val)
        if match:
          code, name = match.group(1), match.group(2).strip()
          # 排除特別股（名稱含「特」）或存託憑證（91開頭）
          if not name.endswith("特") and not code.startswith("91"):
            stock_dict[f"{code}{suffix}"] = name
    except Exception as err:
      print(f"⚠️ 抓取清單時發生例外 ({url}): {err}")

  print(
      f"✅ 成功獲取台股普通股全市場清單：共 {len(stock_dict)} 檔標的 (上市櫃)"
  )
  return stock_dict


# ==================== 2. 三大核心波段策略邏輯 ====================
def screen_vcp_setup(df: pd.DataFrame) -> Optional[dict]:
  """策略一：VCP 壓縮突破動能策略"""
  if len(df) < 100:
    return None

  today = df.iloc[-1]
  yesterday = df.iloc[-2]

  # 趨勢過濾：多頭排列且未跌破季線
  cond_trend = (
      today["close"] > today["ma20"]
      and today["ma20"] > today["ma60"]
      and today["close"] >= (today["high_60"] * 0.82)
  )

  # 收縮過濾：前 5 日振幅 <= 8% 且相較 20 日振幅明顯收斂
  cond_vcp_price = (yesterday["range_5d"] <= 0.08) and (
      yesterday["range_5d"] < yesterday["range_20d"] * 0.65
  )

  # 籌碼沉澱：整理期量縮
  cond_vcp_volume = yesterday["vol_ma5"] <= (yesterday["vol_ma20"] * 0.80)

  # 動能突破：帶量長紅過樞紐點
  cond_breakout = (
      today["close"] > today["pivot_point"]
      and today["close"] > today["open"]
      and today["volume"] >= (today["vol_ma20"] * 1.5)
  )

  if cond_trend and cond_vcp_price and cond_vcp_volume and cond_breakout:
    return {
        "pivot_point": round(float(today["pivot_point"]), 2),
        "breakout_ratio": round(
            float(
                (today["close"] - today["pivot_point"])
                / today["pivot_point"]
                * 100
            ),
            2,
        ),
        "volume_surge_ratio": round(
            float(today["volume"] / max(today["vol_ma20"], 1)), 2
        ),
        "stop_loss": round(float(today["pivot_point"] * 0.96), 2),
    }
  return None


def screen_pullback_setup(df: pd.DataFrame) -> Optional[dict]:
  """策略二：強勢股回後買上漲策略"""
  if len(df) < 60:
    return None

  today = df.iloc[-1]
  yesterday = df.iloc[-2]

  # 前期強勢度：近 30 日曾站上波段新高且具備動能
  cond_prior_strength = (
      today["high_30"] >= df["close"].iloc[-40:-10].min() * 1.15
  )

  # 回測支撐：貼近 20MA（回測月線在 ±2.5% 內），且月線本身斜率向上
  ma20_slope_positive = today["ma20"] > df["ma20"].iloc[-5]
  cond_touch_ma20 = (
      abs(today["low"] - today["ma20"]) / today["ma20"] <= 0.025
  ) or (today["low"] <= today["ma20"] <= today["close"])

  # 窒息量清洗：前 3 日均量縮至均量以下
  cond_dryup = df["volume"].iloc[-4:-1].mean() <= (today["vol_ma20"] * 0.70)

  # 轉強確認：當日收紅 K 且站上 5MA
  cond_reversal = (
      today["close"] > today["open"]
      and today["close"] >= today["ma5"]
      and today["close"] > yesterday["close"]
  )

  if (
      cond_prior_strength
      and ma20_slope_positive
      and cond_touch_ma20
      and cond_dryup
      and cond_reversal
  ):
    recent_swing_low = df["low"].iloc[-5:].min()
    return {
        "support_ma20": round(float(today["ma20"]), 2),
        "pullback_ratio": round(
            float((today["high_30"] - today["close"]) / today["high_30"] * 100),
            2,
        ),
        "stop_loss": round(float(recent_swing_low * 0.985), 2),
    }
  return None


def screen_bottom_reversal_setup(df: pd.DataFrame) -> Optional[dict]:
  """策略三：底部底底高扭轉策略"""
  if len(df) < 100:
    return None

  today = df.iloc[-1]

  # 均線糾結：20MA 與 60MA 差距在 4% 以內
  ma_tangled = abs(today["ma20"] - today["ma60"]) / today["ma60"] <= 0.04

  # 底部形態：底底高 (近 15 天波谷低點 > 前 16~40 天波谷低點)
  trough_recent = df["low"].iloc[-15:].min()
  trough_previous = df["low"].iloc[-40:-15].min()
  higher_low = trough_recent > trough_previous

  # 扭轉起漲：突破季線或箱型頸線，成交量放大
  cond_breakout = (
      today["close"] > today["ma60"]
      and today["close"] > today["open"]
      and today["volume"] >= (today["vol_ma20"] * 1.3)
      and today["ma20"] >= df["ma20"].iloc[-5]
  )

  if ma_tangled and higher_low and cond_breakout:
    return {
        "ma60_level": round(float(today["ma60"]), 2),
        "recent_trough": round(float(trough_recent), 2),
        "stop_loss": round(float(trough_recent * 0.98), 2),
    }
  return None


# ==================== 3. 指標計算與特徵提取 ====================
def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
  """計算技術指標與收縮特徵"""
  data = df.copy()
  c = data["close"]
  h = data["high"]
  l = data["low"]
  v = data["volume"]

  data["ma5"] = c.rolling(5).mean()
  data["ma20"] = c.rolling(20).mean()
  data["ma60"] = c.rolling(60).mean()
  data["vol_ma5"] = v.rolling(5).mean()
  data["vol_ma20"] = v.rolling(20).mean()

  # 區間極值 (不含當日)
  data["high_30"] = h.shift(1).rolling(30).max()
  data["high_60"] = h.shift(1).rolling(60).max()

  # 振幅計算
  h20 = h.shift(1).rolling(20).max()
  l20 = l.shift(1).rolling(20).min()
  data["range_20d"] = (h20 - l20) / h20

  h5 = h.shift(1).rolling(5).max()
  l5 = l.shift(1).rolling(5).min()
  data["range_5d"] = (h5 - l5) / h5
  data["pivot_point"] = h.shift(1).rolling(10).max()

  return data


# ==================== 4. 分塊批次全市場掃描引擎 ====================
def run_full_market_screener() -> List[dict]:
  universe = fetch_tw_stock_universe()
  ticker_list = list(universe.keys())

  if SCAN_LIMIT:
    ticker_list = ticker_list[:SCAN_LIMIT]
    print(f"🔧 測試模式：限制掃描前 {SCAN_LIMIT} 檔標的")

  total_stocks = len(ticker_list)
  matched_results = []
  scanned_count = 0

  print(
      f"🚀 開始全市場掃描：共 {total_stocks} 檔標的，以 {CHUNK_SIZE}"
      " 檔為一組批次處理..."
  )

  for i in range(0, total_stocks, CHUNK_SIZE):
    chunk = ticker_list[i : i + CHUNK_SIZE]
    batch_str = " ".join(chunk)

    try:
      # 批量抓取近 7 個月日 K (約 140 個交易日，兼顧均線計算與頻寬效率)
      batch_data = yf.download(
          tickers=batch_str,
          period="7mo",
          interval="1d",
          group_by="ticker",
          threads=True,
          progress=False,
      )
    except Exception as err:
      print(f"⚠️ 下載批次時發生錯誤 (區間 {i}~{i+CHUNK_SIZE}): {err}")
      continue

    for ticker in chunk:
      scanned_count += 1
      try:
        # 從 MultiIndex 提取單一股票資料
        if (
            isinstance(batch_data.columns, pd.MultiIndex)
            and ticker in batch_data.columns.levels[0]
        ):
          df = batch_data[ticker].dropna().copy()
        elif not isinstance(batch_data.columns, pd.MultiIndex):
          df = batch_data.dropna().copy()
        else:
          continue

        if len(df) < 70:
          continue

        # 欄位正規化為小寫
        df.columns = [col.lower() for col in df.columns]

        # 台股單位換算 (yfinance 取得的台股 Volume 為股數，轉為張數 /1000)
        df["volume"] = df["volume"] / 1000.0

        # 流動性與股價基礎過濾 (大幅節省後續複雜特徵計算時間)
        latest_close = float(df["close"].iloc[-1])
        vol_ma20 = float(df["volume"].rolling(20).mean().iloc[-1])

        if latest_close < MIN_PRICE or vol_ma20 < MIN_VOLUME_MA20:
          continue

        # 計算指標
        feat_df = compute_technical_indicators(df)

        # 檢驗三大策略
        vcp_signal = screen_vcp_setup(feat_df)
        pullback_signal = screen_pullback_setup(feat_df)
        reversal_signal = screen_bottom_reversal_setup(feat_df)

        triggered_strategies = []
        if vcp_signal:
          triggered_strategies.append("VCP_BREAKOUT")
        if pullback_signal:
          triggered_strategies.append("PULLBACK_BUY")
        if reversal_signal:
          triggered_strategies.append("BOTTOM_REVERSAL")

        if triggered_strategies:
          # 取出最後 15 日收盤價供前端繪製微走勢圖 (Sparkline)
          chart_history = [
              round(float(p), 2) for p in df["close"].iloc[-15:].tolist()
          ]

          matched_results.append({
              "ticker": ticker,
              "name": universe.get(ticker, ticker),
              "close": round(latest_close, 2),
              "volume": int(df["volume"].iloc[-1]),
              "strategies": triggered_strategies,
              "vcp": vcp_signal,
              "pullback": pullback_signal,
              "reversal": reversal_signal,
              "chart_history": chart_history,
          })

      except Exception:
        continue

    # 印出當前進度與暫歇避免 Rate limit
    print(
        f"⏳ 進度：[{min(scanned_count, total_stocks)}/{total_stocks}]"
        f" ({scanned_count/total_stocks*100:.1f}%) | 目前入選:"
        f" {len(matched_results)} 檔"
    )
    time.sleep(REQUEST_SLEEP)

  return matched_results


# ==================== 5. 輸出 JSON 與自動維護歷史存檔 ====================
def save_output(results: List[dict]):
  # 設定台灣時間 (UTC+8)
  tz_tw = timezone(timedelta(hours=8))
  now = datetime.now(tz_tw)
  date_str = now.strftime("%Y-%m-%d")
  updated_str = now.strftime("%Y-%m-%d %H:%M:%S (CST)")

  # 建立目錄架構
  os.makedirs(OUTPUT_DIR, exist_ok=True)
  history_dir = os.path.join(OUTPUT_DIR, "history")
  os.makedirs(history_dir, exist_ok=True)

  payload = {
      "updated_at": updated_str,
      "total_scanned": len(results),
      "results": results,
  }

  # 1. 寫入最新覆寫檔 (供首頁預設讀取)
  with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

  # 2. 寫入每日歷史存檔 (例如: docs/history/2026-10-28.json)
  history_file = os.path.join(history_dir, f"{date_str}.json")
  with open(history_file, "w", encoding="utf-8") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

  # 3. 更新歷史目錄清單 (供網頁下拉選單讀取)
  index_file = os.path.join(history_dir, "index.json")
  available_dates = []
  
  if os.path.exists(index_file):
    try:
      with open(index_file, "r", encoding="utf-8") as f:
        available_dates = json.load(f)
    except Exception:
      pass

  # 將新日期加入並由新到舊排序
  if date_str not in available_dates:
    available_dates.append(date_str)
    
  available_dates.sort(reverse=True)

  with open(index_file, "w", encoding="utf-8") as f:
    json.dump(available_dates, f, ensure_ascii=False, indent=2)

  print(f"\n🎉 全市場掃描完成！符合條件標的共 {len(results)} 檔。")
  print(f"📁 最新結果已儲存至: {OUTPUT_FILE}")
  print(f"🗄️ 歷史紀錄已歸檔至: {history_file}")

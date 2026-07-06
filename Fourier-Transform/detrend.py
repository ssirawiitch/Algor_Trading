import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.signal import detrend

# ตั้งค่าการแสดงผลกราฟ
plt.style.use('seaborn-v0_8-whitegrid')

# 1. Data Acquisition & Preparation
ticker = 'SPY'
df = yf.download(ticker, start="2020-01-01", end="2025-01-01", progress=False)

# Robust column selection (เลือกคอลัมน์ Close)
close_cols = [col for col in df.columns if 'Close' in (col[0] if isinstance(col, tuple) else col)]
prices_raw = df[close_cols[0]].dropna().values.flatten()
N = len(prices_raw) # จำนวนข้อมูล (วัน)

# 2. Processing Pipeline A: RAW DATA
# ทำ FFT กับข้อมูลดิบๆ เลย
yf_raw = fft(prices_raw)
xf_freqs = fftfreq(N, d=1) # d=1 คือระยะห่าง 1 วัน

# กรองเอาเฉพาะความถี่ที่เป็นบวก
mask = xf_freqs > 0
freqs_raw = xf_freqs[mask]
amps_raw = np.abs(yf_raw[mask])
periods_raw = 1 / freqs_raw

# 3. Processing Pipeline B: DETRENDED DATA
# ทำ Detrend ก่อน (หัวใจสำคัญ)
prices_detrended = detrend(prices_raw)

# ทำ FFT กับข้อมูลที่ Detrend แล้ว
yf_detrended = fft(prices_detrended)
# (ใช้ความถี่ชุดเดิม xf_freqs และ mask เดิม)
amps_detrended = np.abs(yf_detrended[mask])
periods_detrended = 1 / freqs_raw # Period คำนวณเหมือนเดิม

# =========================================
# 4. Visualization (เปรียบเทียบ)
# =========================================
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f'Fourier Transform Comparison: Raw vs. Detrended ({ticker})', fontsize=16, y=1.02)

# --- ROW 1: Time Domain (โดเมนเวลา) ---
# Plot 1A: Raw Prices
axes[0, 0].plot(prices_raw, color='forestgreen', alpha=0.8)
axes[0, 0].set_title('A1. Time Domain: Raw Prices (มี Trend)', fontweight='bold')
axes[0, 0].set_ylabel('Price ($)')
axes[0, 0].grid(True)

# Plot 1B: Detrended Prices
axes[0, 1].plot(prices_detrended, color='royalblue', alpha=0.8)
axes[0, 1].set_title('B1. Time Domain: Detrended Prices (ตัด Trend ออก)', fontweight='bold')
axes[0, 1].set_ylabel('Deviation from Trend')
axes[0, 1].axhline(0, color='black', lw=1, ls='--')
axes[0, 1].grid(True)

# --- ROW 2: Frequency Domain (โดเมนความถี่) ---
# Plot 2A: FFT of Raw Prices
# หมายเหตุ: ต้องเริ่ม plot ที่ index 5 เพราะ index 0-4 ของ Raw data มีค่าพลังงานสูงมากจนทับกราฟอื่นมิด
start_idx = 5
axes[1, 0].plot(periods_raw[start_idx:], amps_raw[start_idx:], color='forestgreen')
axes[1, 0].set_title('A2. Freq Domain: FFT on RAW Data (ดูยาก)', fontweight='bold')
axes[1, 0].set_xlabel('Cycle Period (Days)')
axes[1, 0].set_ylabel('Amplitude (Strength)')
axes[1, 0].set_xlim(0, 200) # ซูมดูช่วง 0-200 วัน
# สร้างพื้นที่สีแดงแสดงโซนที่ Trend บดบัง
axes[1, 0].axvspan(0, periods_raw[start_idx], color='red', alpha=0.2, label='Zone โดน Trend บัง')
axes[1, 0].legend()

# Plot 2B: FFT of Detrended Prices
axes[1, 1].plot(periods_detrended, amps_detrended, color='royalblue')
axes[1, 1].set_title('B2. Freq Domain: FFT on DETRENDED Data (ชัดเจน)', fontweight='bold')
axes[1, 1].set_xlabel('Cycle Period (Days)')
axes[1, 1].set_xlim(0, 200) # ซูมดูช่วง 0-200 วัน

# Annotate Peak ที่น่าสนใจใน Detrended Plot
peak_idx = np.argmax(amps_detrended)
axes[1, 1].annotate(f'Clear Peak: ~{periods_detrended[peak_idx]:.1f} Days',
                    xy=(periods_detrended[peak_idx], amps_detrended[peak_idx]),
                    xytext=(periods_detrended[peak_idx]+20, amps_detrended[peak_idx]*0.8),
                    arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)


plt.tight_layout()
plt.show()
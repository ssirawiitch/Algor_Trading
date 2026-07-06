"""
สคริปต์นี้จะต่อ cells ทั้ง 7 Steps เข้า linear_algebra.ipynb
รัน: python add_steps_to_notebook.py
"""
import nbformat as nbf
import json, os

NB_PATH = os.path.join(os.path.dirname(__file__), "linear_algebra.ipynb")

def md(source):
    return nbf.v4.new_markdown_cell(source)

def code(source):
    return nbf.v4.new_code_cell(source)

# ===================== NEW CELLS =====================

new_cells = []

# ── PREP ──────────────────────────────────────────────
new_cells.append(md("""---
# 🔧 เตรียมข้อมูลจาก SET สำหรับทุก Step

ก่อนเริ่ม Step 1–7 เราจะเตรียมข้อมูลจาก `close_df` ที่ได้จาก ezyquant:
- **คัดหุ้นที่มีข้อมูลครบ** อย่างน้อย 80% ของช่วงเวลา
- **คำนวณ log return รายวัน** สำหรับใช้ใน Markowitz, PCA, OLS, VaR
- **เลือก 20 หุ้นสภาพคล่องสูงสุด** (มีข้อมูลครบที่สุด) เพื่อให้ covariance matrix invertible ได้
- Import library เพิ่มเติมที่จำเป็น"""))

new_cells.append(code("""\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.optimize import minimize
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.covariance import LedoitWolf

plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 11
np.random.seed(42)

# --- คัดหุ้นที่มีข้อมูลครบ >= 80% ---
thresh = 0.80
valid_cols = close_df.columns[close_df.notna().mean() >= thresh]
price_clean = close_df[valid_cols].ffill().dropna()

# --- เลือก 20 หุ้นที่มีข้อมูลครบที่สุด ---
completeness = close_df[valid_cols].notna().mean().sort_values(ascending=False)
TOP_N = 20
top_tickers = completeness.head(TOP_N).index.tolist()
prices = price_clean[top_tickers].dropna()

# --- Log Returns รายวัน ---
returns = np.log(prices / prices.shift(1)).dropna()

TRADING_DAYS = 245  # วันซื้อขายต่อปีในตลาด SET

print(f'✅ จำนวนหุ้นที่ใช้        : {len(top_tickers)}')
print(f'   ช่วงเวลา              : {returns.index[0].date()} – {returns.index[-1].date()}')
print(f'   จำนวนวันซื้อขาย (T)   : {len(returns)}')
print(f'   อัตราส่วน T/n         : {len(returns)/len(top_tickers):.1f}')
print(f'\\n🏆 20 หุ้นที่คัดเลือก:')
print('  ', top_tickers)
returns.head(3)"""))

# ── STEP 1 ────────────────────────────────────────────
new_cells.append(md("""---
# 📌 Step 1: Portfolio Variance & Markowitz Optimization

## หลักการทางคณิตศาสตร์
- **ผลตอบแทนพอร์ต**: $R_p = w^\\top r$ → $E[R_p] = w^\\top \\mu$
- **ความแปรปรวนพอร์ต (Quadratic Form)**: $\\text{Var}(R_p) = w^\\top \\Sigma w$
- Off-diagonal ของ $\\Sigma$ คือ covariance ระหว่างคู่หุ้น — นี่คือเหตุผลที่ diversification ได้ผล

## Minimum Variance Portfolio (Closed-form)
Minimize $w^\\top \\Sigma w$ s.t. $w^\\top \\mathbf{1} = 1$
$$w^* = \\frac{\\Sigma^{-1} \\mathbf{1}}{\\mathbf{1}^\\top \\Sigma^{-1} \\mathbf{1}}$$

## Max Sharpe Portfolio + Efficient Frontier
เพิ่ม constraint $w \\ge 0$ (ห้าม short) แล้วแก้ด้วย Quadratic Programming (scipy.optimize)"""))

new_cells.append(code("""\
# ============================================================
# STEP 1: Markowitz Portfolio Optimization บนหุ้น SET
# ============================================================

mu_ann    = returns.mean().values * TRADING_DAYS          # expected return รายปี
Sigma_ann = returns.cov().values  * TRADING_DAYS          # covariance รายปี
n         = len(top_tickers)
ones      = np.ones(n)

# --- 1a. Closed-form Minimum Variance Portfolio ---
try:
    Sigma_inv = np.linalg.inv(Sigma_ann)
    w_mvp     = Sigma_inv @ ones / (ones @ Sigma_inv @ ones)
except np.linalg.LinAlgError:
    print('⚠️ Sigma singular — ใช้ Ledoit-Wolf')
    lw0 = LedoitWolf().fit(returns.values)
    Sigma_ann = lw0.covariance_ * TRADING_DAYS
    Sigma_inv = np.linalg.inv(Sigma_ann)
    w_mvp     = Sigma_inv @ ones / (ones @ Sigma_inv @ ones)

ret_mvp  = mu_ann @ w_mvp
risk_mvp = np.sqrt(w_mvp @ Sigma_ann @ w_mvp)
print('✅ Minimum Variance Portfolio (Closed-form):')
for t, w in zip(top_tickers, w_mvp):
    print(f'  {t:10s}: {w*100:+7.2f}%')
print(f'  Return={ret_mvp*100:.2f}%/yr  σ={risk_mvp*100:.2f}%/yr  Sharpe={ret_mvp/risk_mvp:.3f}')

# --- 1b. Max Sharpe (QP, no short-selling) ---
rf = 0.015  # อัตราดอกเบี้ยไทย ~1.5%/yr

def neg_sharpe(w, mu, Sigma, rf=rf):
    r   = mu @ w
    vol = np.sqrt(w @ Sigma @ w)
    return -(r - rf) / vol if vol > 1e-10 else 0.0

res_sr = minimize(
    neg_sharpe, x0=np.ones(n)/n,
    args=(mu_ann, Sigma_ann),
    method='SLSQP',
    bounds=[(0, 1)] * n,
    constraints=[{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}],
    options={'ftol': 1e-12, 'maxiter': 2000}
)
w_sr    = res_sr.x
ret_sr  = mu_ann @ w_sr
risk_sr = np.sqrt(w_sr @ Sigma_ann @ w_sr)
print(f'\\nMax Sharpe: Return={ret_sr*100:.2f}%/yr  σ={risk_sr*100:.2f}%/yr  Sharpe={ret_sr/risk_sr:.2f}')

# --- 1c. Efficient Frontier ---
target_rets = np.linspace(mu_ann.min(), mu_ann.max(), 60)
ef_risk, ef_ret = [], []
for target in target_rets:
    res = minimize(
        lambda w: w @ Sigma_ann @ w, x0=np.ones(n)/n,
        method='SLSQP',
        bounds=[(0,1)]*n,
        constraints=[
            {'type': 'eq', 'fun': lambda w: np.sum(w)-1},
            {'type': 'eq', 'fun': lambda w, t=target: mu_ann @ w - t}
        ],
        options={'ftol': 1e-10, 'maxiter': 1000}
    )
    if res.success:
        ef_risk.append(np.sqrt(res.fun));  ef_ret.append(target)

# --- Plot ---
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
ax = axes[0]
ax.plot(np.array(ef_risk)*100, np.array(ef_ret)*100,
        'royalblue', lw=2.5, label='Efficient Frontier')
ax.scatter(risk_mvp*100, ret_mvp*100, s=250, color='green',
           zorder=5, marker='*', label='Min Variance')
ax.scatter(risk_sr*100,  ret_sr*100,  s=200, color='red',
           zorder=5, marker='D', label='Max Sharpe')
for i, t in enumerate(top_tickers):
    ax.scatter(np.sqrt(Sigma_ann[i,i])*100, mu_ann[i]*100, s=60, alpha=0.7)
    ax.annotate(t, (np.sqrt(Sigma_ann[i,i])*100, mu_ann[i]*100),
                textcoords='offset points', xytext=(4,3), fontsize=7.5)
ax.set_xlabel('Volatility (σ) %/ปี');  ax.set_ylabel('Expected Return %/ปี')
ax.set_title('Efficient Frontier — หุ้น SET (No Short-Selling)')
ax.legend()

ax2 = axes[1]
colors = plt.cm.tab20(np.linspace(0,1,n))
bars = ax2.bar(top_tickers, w_sr*100, color=colors)
ax2.set_xticklabels(top_tickers, rotation=45, ha='right')
ax2.set_ylabel('น้ำหนัก (%)')
ax2.set_title(f'Max Sharpe Weights  |  Sharpe={ret_sr/risk_sr:.2f}')
for bar, val in zip(bars, w_sr):
    if val > 0.01:
        ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                 f'{val*100:.1f}%', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('set_step1_markowitz.png', dpi=120, bbox_inches='tight')
plt.show()
print('\\n✅ Step 1 เสร็จสิ้น')"""))

# ── STEP 2 ────────────────────────────────────────────
new_cells.append(md("""---
# 📌 Step 2: PCA บน Covariance Matrix ของหุ้น SET

## หลักการ
Covariance matrix $\\Sigma$ เป็น symmetric positive semi-definite → Spectral Theorem:
$$\\Sigma = V \\Lambda V^\\top$$
- $V$ = matrix ของ eigenvector (orthonormal)
- **PC1 มักเป็น Market Factor** เพราะหุ้น SET ส่วนใหญ่มี correlation เป็นบวกกับกัน (Perron-Frobenius)
- **% variance explained** = $\\lambda_i / \\sum_j \\lambda_j$

## ประยุกต์ใช้: แยก Market Risk ออกจาก Idiosyncratic Risk
Residual = risk ที่ market factor อธิบายไม่ได้ → ใช้สร้างสัญญาณเทรด mean-reversion"""))

new_cells.append(code("""\
# ============================================================
# STEP 2: PCA บน Covariance Matrix ของหุ้น SET
# ============================================================

R   = returns.values
R_dm = R - R.mean(axis=0)

Sigma_s  = np.cov(R_dm.T)
evals, evecs = np.linalg.eigh(Sigma_s)
idx = np.argsort(evals)[::-1]
evals, evecs = evals[idx], evecs[:, idx]

exp_var = evals / evals.sum() * 100
cum_var = np.cumsum(exp_var)

print('Eigenvalue Analysis (Top 10):')
print(f'{"PC":>4} | {"Eigenvalue":>12} | {"Explained%":>11} | {"Cumulative%":>12}')
print('-'*50)
for i in range(min(10, n)):
    print(f'  {i+1:2d} | {evals[i]:12.6f} | {exp_var[i]:10.2f}% | {cum_var[i]:11.2f}%')

k = 3
V_k = evecs[:, :k]
F   = R_dm @ V_k                   # T x k  (factor scores)
reconstructed = F @ V_k.T          # T x n
residuals = R_dm - reconstructed   # T x n  (idiosyncratic)
print(f'\\nPC1–3 อธิบาย {cum_var[k-1]:.1f}% ของ variance ทั้งหมด')

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

ax = axes[0, 0]
ax.bar(range(1, n+1), exp_var, color='steelblue', alpha=0.7)
ax2t = ax.twinx()
ax2t.plot(range(1, n+1), cum_var, 'r-o', ms=5)
ax2t.set_ylabel('Cumulative %', color='red')
ax.axvline(k+0.5, color='orange', ls='--', lw=2, label=f'Top {k} PCs')
ax.set_xlabel('Principal Component');  ax.set_ylabel('Variance Explained (%)')
ax.set_title('Scree Plot — หุ้น SET');  ax.legend()

ax = axes[0, 1]
im = ax.imshow(V_k.T, aspect='auto', cmap='RdBu_r', vmin=-0.6, vmax=0.6)
ax.set_xticks(range(n));  ax.set_xticklabels(top_tickers, rotation=90, fontsize=7)
ax.set_yticks(range(k));  ax.set_yticklabels([f'PC{i+1}' for i in range(k)])
ax.set_title('Factor Loadings Heatmap');  plt.colorbar(im, ax=ax)

ax = axes[1, 0]
ax.plot(returns.index, F[:, 0], lw=0.8, color='navy', alpha=0.9)
ax.set_title('PC1 — Market Factor (SET)');  ax.set_ylabel('Factor Score')

ax = axes[1, 1]
resid_0 = pd.Series(residuals[:, 0], index=returns.index)
ax.plot(returns.index, resid_0, lw=0.7, color='darkgreen')
mu_r, sg_r = resid_0.mean(), resid_0.std()
ax.axhline(mu_r, color='red', ls='--', lw=1.5, label='Mean')
ax.fill_between(returns.index, mu_r+2*sg_r, mu_r-2*sg_r,
                alpha=0.15, color='orange', label='±2σ')
ax.set_title(f'Idiosyncratic Residual — {top_tickers[0]}');  ax.legend()

plt.tight_layout()
plt.savefig('set_step2_pca.png', dpi=120, bbox_inches='tight')
plt.show()
print('✅ Step 2 เสร็จสิ้น')"""))

# ── STEP 3 ────────────────────────────────────────────
new_cells.append(md("""---
# 📌 Step 3: Factor Models & OLS บนหุ้น SET

## OLS Normal Equation
Minimize $S(\\beta) = (r - X\\beta)^\\top(r - X\\beta)$:
$$\\hat{\\beta} = (X^\\top X)^{-1} X^\\top r$$

ในทางปฏิบัติใช้ **QR decomposition**: $X = QR \\Rightarrow \\hat{\\beta} = R^{-1}Q^\\top r$  
→ เร็วกว่าและเสถียรกว่าการ invert $(X^\\top X)$ ตรงๆ

## SET-Style 3-Factor Model
ใช้ PC1–3 เป็น synthetic factors:
$$r_i = \\alpha_i + \\beta_1 F_1 + \\beta_2 F_2 + \\beta_3 F_3 + \\varepsilon_i$$
- $\\beta_1$ ≈ market beta; $\\alpha_i$ = Jensen's alpha"""))

new_cells.append(code("""\
# ============================================================
# STEP 3: Factor Models & OLS บนหุ้น SET
# ============================================================

T_obs = F.shape[0]
X_ols = np.hstack([np.ones((T_obs, 1)), F])   # T x (k+1)

# Method 1: Direct Normal Equation
B_direct = np.linalg.inv(X_ols.T @ X_ols) @ X_ols.T @ R_dm

# Method 2: QR Decomposition (numerically stable)
Q_qr, R_qr = np.linalg.qr(X_ols)
B_qr       = np.linalg.solve(R_qr, Q_qr.T @ R_dm)

max_diff = np.max(np.abs(B_direct - B_qr))
print(f'✅ Max |Direct − QR| = {max_diff:.2e}  (ต้องใกล้ 0)')

# statsmodels สำหรับ t-stat, R²
print(f'\\n{"Ticker":>10} | {"α(%/yr)":>8} | {"β_PC1":>7} | {"β_PC2":>7} | {"β_PC3":>7} | {"R²":>6}')
print('-'*60)
results_ols = {}
for i, ticker in enumerate(top_tickers):
    model = sm.OLS(R_dm[:, i], X_ols).fit()
    results_ols[ticker] = model
    alpha_ann = model.params[0] * TRADING_DAYS * 100
    betas     = model.params[1:]
    print(f'{ticker:>10} | {alpha_ann:>8.3f} | {betas[0]:>7.3f} | {betas[1]:>7.3f} | {betas[2]:>7.3f} | {model.rsquared:>6.3f}')

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax = axes[0]
B_plot = B_direct[1:, :]
im = ax.imshow(B_plot, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
ax.set_yticks(range(k));  ax.set_yticklabels(['β_PC1 (Market)','β_PC2','β_PC3'])
ax.set_xticks(range(n));  ax.set_xticklabels(top_tickers, rotation=90, fontsize=8)
ax.set_title('Factor Beta Heatmap — หุ้น SET');  plt.colorbar(im, ax=ax)
for row in range(k):
    for col in range(n):
        ax.text(col, row, f'{B_plot[row,col]:.2f}', ha='center', va='center', fontsize=7)

ax2 = axes[1]
r2_vals = [results_ols[t].rsquared for t in top_tickers]
colors_r2 = plt.cm.RdYlGn(np.array(r2_vals))
bars = ax2.bar(top_tickers, [v*100 for v in r2_vals], color=colors_r2)
ax2.set_xticklabels(top_tickers, rotation=90)
ax2.set_ylabel('R² (%)');  ax2.set_ylim(0, 100)
ax2.set_title('R² ของ 3-Factor Model — หุ้น SET')
ax2.axhline(np.mean(r2_vals)*100, color='navy', ls='--', lw=1.5,
            label=f'Mean R²={np.mean(r2_vals)*100:.0f}%')
ax2.legend()

plt.tight_layout()
plt.savefig('set_step3_ols.png', dpi=120, bbox_inches='tight')
plt.show()
print('\\n✅ Step 3 เสร็จสิ้น')"""))

# ── STEP 4 ────────────────────────────────────────────
new_cells.append(md("""---
# 📌 Step 4: Cointegration, Kalman Filter & Pairs Trading (หุ้น SET)

## Correlation ≠ Cointegration
- ราคาหุ้นเป็น **non-stationary (random walk)** — correlation สูงอาจ diverge ระยะยาว
- **Cointegration**: หา linear combination ของราคา 2 ตัวที่ **stationary** → เทรดปลอดภัย

## Kalman Filter — Dynamic Hedge Ratio
แทนที่ hedge ratio คงที่ (OLS), Kalman Filter ให้ hedge ratio ปรับตามเวลา:
- **State**: $\\beta_t = \\beta_{t-1} + w_t$  
- **Observation**: $y_{1t} = \\beta_t y_{2t} + v_t$  
- **Kalman Gain**: $K_t = P_{t|t-1}H(HP_{t|t-1}H^\\top + R)^{-1}$ — ทุก step คือ matrix operation"""))

new_cells.append(code("""\
# ============================================================
# STEP 4: Cointegration + Kalman Filter Pairs Trading (SET)
# ============================================================

print('🔍 ทดสอบ Cointegration ทุกคู่...')
pairs_res = []
for i in range(len(top_tickers)):
    for j in range(i+1, len(top_tickers)):
        t1, t2 = top_tickers[i], top_tickers[j]
        p1, p2 = prices[t1].values, prices[t2].values
        try:
            _, pval, _ = coint(p1, p2)
            pairs_res.append((t1, t2, pval))
        except Exception:
            pass

pairs_df = pd.DataFrame(pairs_res, columns=['Ticker1','Ticker2','p-value'])
pairs_df = pairs_df.sort_values('p-value').reset_index(drop=True)
print('Top 5 Cointegrated Pairs:')
print(pairs_df.head(5).to_string(index=False))

best = pairs_df.iloc[0]
tk1, tk2 = best['Ticker1'], best['Ticker2']
print(f'\\n✅ Best pair: {tk1} – {tk2}  (p={best["p-value"]:.4f})')

y1 = prices[tk1].values.astype(float)
y2 = prices[tk2].values.astype(float)

def kalman_hedge(y1, y2, delta=1e-4, vt=1e-2):
    T = len(y1)
    beta = np.zeros(T);  P = np.zeros(T);  spread = np.zeros(T)
    beta[0] = y1[0]/y2[0];  P[0] = 1.0
    Q = delta/(1-delta);  R_obs = vt
    for t in range(1, T):
        beta_p = beta[t-1];  P_p = P[t-1] + Q
        H = y2[t];  S = H*P_p*H + R_obs;  K = P_p*H/S
        beta[t]   = beta_p + K*(y1[t] - H*beta_p)
        P[t]      = (1 - K*H)*P_p
        spread[t] = y1[t] - beta[t]*y2[t]
    return beta, spread

beta_kf, spread_kf = kalman_hedge(y1, y2)
beta_ols_s = sm.OLS(y1, sm.add_constant(y2)).fit().params[1]

sp_ser = pd.Series(spread_kf, index=prices.index)
win = 60
zscore = (sp_ser - sp_ser.rolling(win).mean()) / sp_ser.rolling(win).std()

adf_stat, adf_p, *_ = adfuller(spread_kf[win:])
print(f'ADF Test: stat={adf_stat:.4f}, p={adf_p:.4f}  →  {"Stationary ✓" if adf_p < 0.05 else "Not stationary"}')

fig, axes = plt.subplots(3, 1, figsize=(14, 11))

ax = axes[0]
ax.plot(prices.index, y1/y1[0]*100, label=tk1, lw=1.2)
ax.plot(prices.index, y2/y2[0]*100, label=tk2, lw=1.2)
ax.set_title(f'Normalized Price: {tk1} vs {tk2}');  ax.legend()

ax = axes[1]
ax.plot(prices.index, beta_kf, label='Kalman (dynamic)', color='navy', lw=1)
ax.axhline(beta_ols_s, color='red', ls='--', lw=1.5, label=f'OLS β={beta_ols_s:.3f}')
ax.set_title('Dynamic Hedge Ratio');  ax.legend()

ax = axes[2]
ax.plot(prices.index, zscore, lw=0.9, color='darkgreen')
ax.axhline(2,  color='red',   ls='--', lw=1)
ax.axhline(-2, color='red',   ls='--', lw=1)
ax.axhline(0,  color='black', ls='-',  lw=0.8)
ax.fill_between(prices.index, 2,  zscore.values, where=zscore.values> 2,
                alpha=0.25, color='red',   label='Short')
ax.fill_between(prices.index, -2, zscore.values, where=zscore.values<-2,
                alpha=0.25, color='green', label='Long')
ax.set_title('Z-Score ของ Spread (สัญญาณเทรด)');  ax.legend()

plt.tight_layout()
plt.savefig('set_step4_kalman.png', dpi=120, bbox_inches='tight')
plt.show()
print('\\n✅ Step 4 เสร็จสิ้น')"""))

# ── STEP 5 ────────────────────────────────────────────
new_cells.append(md("""---
# 📌 Step 5: Random Matrix Theory & Covariance Denoising (SET)

## ปัญหา: Noisy Sample Covariance
เมื่อ $n/T$ ไม่เล็กมาก eigenvalue เล็กๆ ถูกประเมินต่ำ และ eigenvalue ใหญ่ถูกประเมินสูง

## Marchenko-Pastur Distribution
สำหรับ $q = T/n$, ขอบเขต noise eigenvalue:
$$\\lambda_{max} = \\sigma^2\\left(1+\\sqrt{\\frac{1}{q}}\\right)^2$$
Eigenvalue ที่เกิน $\\lambda_{max}$ เท่านั้นที่เป็น signal จริง

## วิธี Denoising
1. **Eigenvalue Clipping** — แทน noise eigenvalue ด้วยค่าเฉลี่ย
2. **Ledoit-Wolf Shrinkage** — $\\Sigma_{shrunk} = \\delta F + (1-\\delta)\\Sigma_s$ (closed-form optimal $\\delta$)"""))

new_cells.append(code("""\
# ============================================================
# STEP 5: Random Matrix Theory & Covariance Denoising (SET)
# ============================================================

T_obs2, n_assets = R.shape
q = T_obs2 / n_assets

Sig_s = np.cov(R.T)
ev_s, evec_s = np.linalg.eigh(Sig_s)
idx_s = np.argsort(ev_s)[::-1]
ev_s, evec_s = ev_s[idx_s], evec_s[:, idx_s]

var_scale  = np.mean(np.diag(Sig_s))
lam_max_mp = var_scale * (1 + np.sqrt(1/q))**2
lam_min_mp = var_scale * (1 - np.sqrt(1/q))**2

signal_mask = ev_s > lam_max_mp
noise_mask  = ~signal_mask
n_signal    = signal_mask.sum()

print(f'q = T/n = {T_obs2}/{n_assets} = {q:.2f}')
print(f'Marchenko-Pastur λ_max = {lam_max_mp:.6f}')
print(f'Signal eigenvalues: {n_signal}  |  Noise: {noise_mask.sum()}')

# Clipping
ev_clip = ev_s.copy()
ev_clip[noise_mask] = ev_s[noise_mask].mean() if noise_mask.any() else 0
Sig_clip = evec_s @ np.diag(ev_clip) @ evec_s.T

# Ledoit-Wolf
lw_model     = LedoitWolf().fit(R)
Sig_lw       = lw_model.covariance_
delta_lw     = lw_model.shrinkage_
Sigma_lw_ann = Sig_lw * TRADING_DAYS  # เก็บไว้ใช้ Step 6-7

print(f'Ledoit-Wolf δ = {delta_lw:.4f}')
print(f'Condition: Sample={np.linalg.cond(Sig_s):.0f}  Clip={np.linalg.cond(Sig_clip):.0f}  LW={np.linalg.cond(Sig_lw):.0f}')

def mp_pdf(x, q, s2=1.0):
    lmax = s2*(1+np.sqrt(1/q))**2;  lmin = s2*(1-np.sqrt(1/q))**2
    mask = (x>=lmin)&(x<=lmax);     pdf = np.zeros_like(x)
    pdf[mask] = q/(2*np.pi*s2*x[mask])*np.sqrt((lmax-x[mask])*(x[mask]-lmin))
    return pdf

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.hist(ev_s, bins=12, density=True, alpha=0.6, color='steelblue', label='Sample eigenvalues')
x_mp = np.linspace(0, np.percentile(ev_s, 90), 300)
ax.plot(x_mp, mp_pdf(x_mp, q, var_scale), 'r-', lw=2, label='Marchenko-Pastur')
ax.axvline(lam_max_mp, color='orange', ls='--', lw=2, label=f'λ_max={lam_max_mp:.4f}')
ax.set_title('Eigenvalue vs Marchenko-Pastur (SET)');  ax.legend(fontsize=8)

ax = axes[1]
x_ = np.arange(n_assets);  w_ = 0.25
ax.bar(x_-w_, np.sqrt(np.diag(Sig_s)),   w_, label='Sample',     color='steelblue',  alpha=0.8)
ax.bar(x_,    np.sqrt(np.diag(Sig_clip)), w_, label='Clipped',    color='darkorange', alpha=0.8)
ax.bar(x_+w_, np.sqrt(np.diag(Sig_lw)),  w_, label='Ledoit-Wolf', color='green',      alpha=0.8)
ax.set_xticks(x_);  ax.set_xticklabels(top_tickers, rotation=90, fontsize=7)
ax.set_ylabel('Daily σ');  ax.set_title('Diagonal σ Comparison');  ax.legend()

ax = axes[2]
D_inv = np.diag(1.0/np.sqrt(np.diag(Sig_lw)))
corr_lw = D_inv @ Sig_lw @ D_inv
im = ax.imshow(corr_lw, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(n_assets));  ax.set_xticklabels(top_tickers, rotation=90, fontsize=7)
ax.set_yticks(range(n_assets));  ax.set_yticklabels(top_tickers, fontsize=7)
ax.set_title('Ledoit-Wolf Correlation Matrix (SET)');  plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('set_step5_rmt.png', dpi=120, bbox_inches='tight')
plt.show()
print('\\n✅ Step 5 เสร็จสิ้น')"""))

# ── STEP 6 ────────────────────────────────────────────
new_cells.append(md("""---
# 📌 Step 6: VaR & Cholesky Decomposition บนพอร์ต SET

## Variance-Covariance VaR (Analytical)
$$\\text{VaR}_{\\alpha} = z_\\alpha \\cdot \\sqrt{w^\\top \\Sigma w} \\cdot V_{portfolio}$$

## Monte Carlo VaR ด้วย Cholesky
เพื่อ simulate returns ที่มี correlation:
1. $\\Sigma = LL^\\top$ (Cholesky — L คือ lower triangular matrix)
2. สุ่ม $z \\sim \\mathcal{N}(0, I)$
3. $r_{sim} = \\mu + Lz$

**พิสูจน์**: $\\text{Cov}(Lz) = LIL^\\top = LL^\\top = \\Sigma$ ✓  
Cholesky เร็วกว่า full eigen-decomp: $O(n^3/3)$ vs $O(n^3)$"""))

new_cells.append(code("""\
# ============================================================
# STEP 6: VaR & Cholesky Decomposition บนพอร์ต SET
# ============================================================

PORT_VALUE = 10_000_000  # 10 ล้านบาท
CONF       = 0.99

w          = w_sr          # Max Sharpe weights จาก Step 1
Sig_daily  = Sig_lw.copy() # Ledoit-Wolf รายวัน
mu_daily   = mu_ann / TRADING_DAYS

# --- Method 1: Variance-Covariance ---
port_vol_d = np.sqrt(w @ Sig_daily @ w)
z_score    = stats.norm.ppf(CONF)
VaR_vc     = z_score * port_vol_d * PORT_VALUE
CVaR_vc    = (stats.norm.pdf(z_score)/(1-CONF)) * port_vol_d * PORT_VALUE

# --- Method 2: Monte Carlo + Cholesky ---
Sig_pd  = Sig_daily + 1e-8 * np.eye(n_assets)
L_chol  = np.linalg.cholesky(Sig_pd)
N_SIM   = 100_000
Z       = np.random.standard_normal((n_assets, N_SIM))
sim_ret = (mu_daily[:,None] + L_chol @ Z).T
port_pnl = sim_ret @ w * PORT_VALUE
VaR_mc   = -np.percentile(port_pnl, (1-CONF)*100)
CVaR_mc  = -port_pnl[port_pnl < -VaR_mc].mean()

# --- Method 3: Historical ---
hist_pnl  = (R @ w) * PORT_VALUE
VaR_hist  = -np.percentile(hist_pnl, (1-CONF)*100)
CVaR_hist = -hist_pnl[hist_pnl < -VaR_hist].mean()

recon_err = np.max(np.abs(L_chol @ L_chol.T - Sig_pd))

print(f'Portfolio daily σ  = {port_vol_d*100:.4f}%')
print(f'{"="*55}')
print(f'VaR-Covariance  VaR@99% : {VaR_vc:>12,.0f} บาท  CVaR: {CVaR_vc:>12,.0f}')
print(f'Monte Carlo     VaR@99% : {VaR_mc:>12,.0f} บาท  CVaR: {CVaR_mc:>12,.0f}')
print(f'Historical      VaR@99% : {VaR_hist:>12,.0f} บาท  CVaR: {CVaR_hist:>12,.0f}')
print(f'✅ Max |LLᵀ − Σ| = {recon_err:.2e}')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

ax = axes[0]
ax.hist(port_pnl/1e6, bins=200, density=True, alpha=0.7, color='steelblue')
ax.axvline(-VaR_mc/1e6,  color='red',    lw=2, ls='--', label=f'VaR={VaR_mc/1e6:.2f}M')
ax.axvline(-CVaR_mc/1e6, color='darkred', lw=2, ls=':',  label=f'CVaR={CVaR_mc/1e6:.2f}M')
ax.set_xlabel('Daily P&L (ล้านบาท)');  ax.set_ylabel('Density')
ax.set_title('P&L Distribution — Monte Carlo (SET)');  ax.legend(fontsize=9)

ax = axes[1]
im = ax.imshow(L_chol, cmap='RdBu_r')
ax.set_xticks(range(n_assets));  ax.set_xticklabels(top_tickers, rotation=90, fontsize=7)
ax.set_yticks(range(n_assets));  ax.set_yticklabels(top_tickers, fontsize=7)
ax.set_title('Cholesky Factor L');  plt.colorbar(im, ax=ax)

ax = axes[2]
x_ = np.arange(3)
ax.bar(x_-0.2, [VaR_vc/1e6, VaR_mc/1e6, VaR_hist/1e6],  0.35,
       label='VaR@99%',  color='steelblue',  alpha=0.85)
ax.bar(x_+0.2, [CVaR_vc/1e6, CVaR_mc/1e6, CVaR_hist/1e6], 0.35,
       label='CVaR@99%', color='darkorange', alpha=0.85)
ax.set_xticks(x_);  ax.set_xticklabels(['Var-Cov','Monte Carlo','Historical'])
ax.set_ylabel('ล้านบาท');  ax.set_title('VaR & CVaR: 3 วิธี');  ax.legend()

plt.tight_layout()
plt.savefig('set_step6_var.png', dpi=120, bbox_inches='tight')
plt.show()
print('\\n✅ Step 6 เสร็จสิ้น')"""))

# ── STEP 7 ────────────────────────────────────────────
new_cells.append(md("""---
# 📌 Step 7: Black-Litterman Model บนหุ้น SET

## ปัญหาของ Markowitz ดั้งเดิม
- อ่อนไหวต่อ $\\mu$ มากเกินไป (error maximization) — weight เปลี่ยนสุดขั้วเมื่อ $\\mu$ เปลี่ยนเล็กน้อย

## Black-Litterman Bayesian Framework
1. **Equilibrium Return**: $\\Pi = \\delta \\Sigma w_{market}$ (reverse optimization)
2. **Views**: $P$, $Q$, $\\Omega$ (ความไม่แน่นอน)
3. **Posterior** (Bayesian update):
$$\\mu_{BL} = \\left[(\\tau\\Sigma)^{-1} + P^\\top\\Omega^{-1}P\\right]^{-1}\\left[(\\tau\\Sigma)^{-1}\\Pi + P^\\top\\Omega^{-1}Q\\right]$$

ผลลัพธ์: weight ที่ **ไม่สุดขั้ว** และ **เสถียรกว่า** เมื่อ return เปลี่ยน"""))

new_cells.append(code("""\
# ============================================================
# STEP 7: Black-Litterman Model บนหุ้น SET
# ============================================================

delta_bl = 2.5;  tau = 0.05;  rf_bl = 0.015
w_mkt    = np.ones(n) / n
Sig_bl   = Sigma_lw_ann.copy()

# 7a. Equilibrium Return
Pi = delta_bl * Sig_bl @ w_mkt
print('Equilibrium Returns Π:')
for t, p in zip(top_tickers, Pi):
    print(f'  {t:10s}: {p*100:+.3f}%')

# 7b. Views
n_views = 3
P_v = np.zeros((n_views, n))
P_v[0,0]=1.0;  P_v[0,1]=-1.0
P_v[1,2]=1.0;  P_v[1,3]=-0.5;  P_v[1,4]=-0.5
P_v[2,5]=1.0
Q_v = np.array([0.03, 0.05, 0.04])
Omega_v = np.diag(np.diag(tau * P_v @ Sig_bl @ P_v.T))

print(f'\\nViews:')
print(f'  {top_tickers[0]} – {top_tickers[1]} = +{Q_v[0]*100:.1f}%/yr')
print(f'  {top_tickers[2]} – avg({top_tickers[3]},{top_tickers[4]}) = +{Q_v[1]*100:.1f}%/yr')
print(f'  {top_tickers[5]} = +{Q_v[2]*100:.1f}%/yr')

# 7c. Bayesian Posterior
tauSig_inv = np.linalg.inv(tau * Sig_bl)
Omega_inv  = np.linalg.inv(Omega_v)
M_post = np.linalg.inv(tauSig_inv + P_v.T @ Omega_inv @ P_v)
mu_BL  = M_post @ (tauSig_inv @ Pi + P_v.T @ Omega_inv @ Q_v)

print(f'\\n{"Ticker":>10} | {"Prior%":>8} | {"BL μ%":>8} | {"Δ%":>8}')
print('-'*45)
for t, p, m in zip(top_tickers, Pi, mu_BL):
    print(f'{t:>10} | {p*100:>8.3f} | {m*100:>8.3f} | {(m-p)*100:>+8.3f}')

# 7d. Optimize
def max_sharpe_port(mu_in, Sig_in):
    res = minimize(
        lambda w: -(mu_in@w - rf_bl)/max(np.sqrt(w@Sig_in@w),1e-10),
        x0=np.ones(n)/n, method='SLSQP',
        bounds=[(0,1)]*n,
        constraints=[{'type':'eq','fun':lambda w: np.sum(w)-1}],
        options={'ftol':1e-12,'maxiter':2000}
    )
    return res.x

w_naive = max_sharpe_port(mu_ann, Sig_bl)
w_bl_p  = max_sharpe_port(mu_BL,  Sig_bl)

# Stability test
n_perturb = 50;  shock_std = 0.005
std_naive_arr = np.array([max_sharpe_port(mu_ann+np.random.normal(0,shock_std,n), Sig_bl)
                           for _ in range(n_perturb)])
std_bl_arr    = np.array([max_sharpe_port(mu_BL +np.random.normal(0,shock_std,n), Sig_bl)
                           for _ in range(n_perturb)])
std_naive_w = std_naive_arr.std(axis=0)
std_bl_w    = std_bl_arr.std(axis=0)
improvement = (std_naive_w.mean()-std_bl_w.mean())/std_naive_w.mean()*100

print(f'\\nWeight std: Naive={std_naive_w.mean()*100:.3f}%  BL={std_bl_w.mean()*100:.3f}%')
print(f'BL เสถียรกว่า {improvement:.1f}%')

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
x_ = np.arange(n)

ax = axes[0]
ax.bar(x_-0.2, Pi*100,    0.35, label='Prior Π',     color='steelblue',  alpha=0.85)
ax.bar(x_+0.2, mu_BL*100, 0.35, label='BL Posterior', color='darkorange', alpha=0.85)
ax.set_xticks(x_);  ax.set_xticklabels(top_tickers, rotation=90, fontsize=8)
ax.set_ylabel('Expected Return (%/ปี)');  ax.set_title('Prior vs BL Return (SET)');  ax.legend()

ax = axes[1]
ax.bar(x_-0.2, w_naive*100, 0.35, label='Naive',          color='steelblue',  alpha=0.85)
ax.bar(x_+0.2, w_bl_p*100,  0.35, label='Black-Litterman', color='darkorange', alpha=0.85)
ax.set_xticks(x_);  ax.set_xticklabels(top_tickers, rotation=90, fontsize=8)
ax.set_ylabel('Weight (%)');  ax.set_title('Optimal Weights: Naive vs BL');  ax.legend()

ax = axes[2]
ax.bar(x_-0.2, std_naive_w*100, 0.35, label='Naive',          color='steelblue',  alpha=0.85)
ax.bar(x_+0.2, std_bl_w*100,    0.35, label='Black-Litterman', color='darkorange', alpha=0.85)
ax.set_xticks(x_);  ax.set_xticklabels(top_tickers, rotation=90, fontsize=8)
ax.set_ylabel('Weight Std (%)');  ax.set_title('Weight Stability');  ax.legend()

plt.tight_layout()
plt.savefig('set_step7_bl.png', dpi=120, bbox_inches='tight')
plt.show()
print('\\n✅ Step 7 เสร็จสิ้น')"""))

# ── FINAL SUMMARY ─────────────────────────────────────
new_cells.append(md("""---
# 🏁 สรุปผล — ทุก 7 Steps บนข้อมูลหุ้น SET จริง

| Step | หัวข้อ | เทคนิค | ผล |
|:----:|:-------|:-------|:---|
| 1 | Markowitz | Matrix inversion, QP | Efficient Frontier + Max Sharpe |
| 2 | PCA | Eigen-decomposition | Market Factor + Idiosyncratic |
| 3 | OLS | Normal equation, QR | β loadings, R² |
| 4 | Cointegration + Kalman | State-space model | Dynamic hedge ratio, z-score |
| 5 | Random Matrix Theory | Marchenko-Pastur, LW | Denoised Σ |
| 6 | VaR + Cholesky | $\\Sigma=LL^\\top$, MC | VaR/CVaR 3 วิธี |
| 7 | Black-Litterman | Bayesian update | Weight เสถียรกว่า |

> ข้อมูล: ezyquant (SET 2015–2025), 20 หุ้นที่มีข้อมูลครบที่สุด"""))

new_cells.append(code("""\
# ============================================================
# FINAL SUMMARY TABLE
# ============================================================
import pandas as pd
summary = pd.DataFrame({
    'Step': ['1','2','3','4','5','6','7'],
    'หัวข้อ': [
        'Markowitz Optimization',
        'PCA on Covariance',
        'Factor Models & OLS',
        'Cointegration + Kalman',
        'Random Matrix Theory',
        'VaR & Cholesky',
        'Black-Litterman'
    ],
    'Key Result': [
        f'Max Sharpe: Ret={ret_sr*100:.1f}%/yr, σ={risk_sr*100:.1f}%/yr, Sharpe={ret_sr/risk_sr:.2f}',
        f'PC1–3 อธิบาย {cum_var[k-1]:.1f}% variance | Signal EVs={n_signal}',
        f'Mean R² = {round(float(pd.Series([results_ols[t].rsquared for t in top_tickers]).mean())*100)}%',
        f'Best pair: {tk1}–{tk2} (p={best["p-value"]:.4f})',
        f'Signal EVs: {n_signal}/{n_assets} | LW δ={delta_lw:.3f}',
        f'MC VaR@99%={VaR_mc/1e6:.2f}M฿ | CVaR={CVaR_mc/1e6:.2f}M฿',
        f'Naive std={std_naive_w.mean()*100:.3f}% → BL={std_bl_w.mean()*100:.3f}% ({improvement:.1f}% stable)'
    ]
}).set_index('Step')

pd.set_option('display.max_colwidth', 80)
print('='*85)
print('📊 สรุปผลลัพธ์ทุก Step — Portfolio Optimization บนหุ้น SET')
print('='*85)
print(summary.to_string())
print('\\n✅ เสร็จสิ้นทุก 7 Steps!')"""))


# ===================== INJECT INTO NOTEBOOK =====================

with open(NB_PATH, "r", encoding="utf-8") as f:
    nb = nbf.read(f, as_version=4)

nb.cells.extend(new_cells)

with open(NB_PATH, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"✅ เพิ่ม {len(new_cells)} cells เข้า {NB_PATH} เรียบร้อยแล้ว!")
print("   เปิด Jupyter แล้วรัน notebook ได้เลย")

1. Portfolio Variance & Markowitz Optimization (แบบละเอียด)
Setup
สมมติมี asset n ตัว, return แต่ละตัวเป็น random variable ที่มี expected return μ = [μ₁, μ₂, ..., μₙ]ᵀ และ covariance matrix Σ (n×n, symmetric positive semi-definite)
ผลตอบแทนพอร์ต: Rₚ = wᵀr → E[Rₚ] = wᵀμ
ความแปรปรวนพอร์ต:
Var(Rₚ) = Var(wᵀr) = wᵀ Σ w = Σᵢ Σⱼ wᵢwⱼσᵢⱼ
ตรงนี้คือจุดที่ linear algebra เข้ามา เพราะ variance ของ linear combination ของ random variables เขียนเป็น quadratic form ได้พอดี ไม่ใช่แค่ sum ของ variance เฉยๆ แต่ต้องนับ covariance ระหว่างคู่สินทรัพย์ทุกคู่ (ตรง off-diagonal ของ Σ) — นี่คือเหตุผลที่ diversification ได้ผล ถ้า σᵢⱼ ติดลบหรือต่ำ พอร์ตรวมความเสี่ยงลดลงกว่าผลรวมความเสี่ยงแต่ละตัว
การหา Optimal Weight (Minimum Variance Portfolio)
โจทย์คือ minimize wᵀΣw subject to wᵀ1 = 1 (น้ำหนักรวมเท่ากับ 1)
ใช้ Lagrangian:
L = wᵀΣw - λ(wᵀ1 - 1)
หา first-order condition โดย derivative เทียบ w:
∂L/∂w = 2Σw - λ1 = 0
→ w = (λ/2) Σ⁻¹1
แล้วใช้ constraint wᵀ1=1 หาค่า λ ได้:
w* = Σ⁻¹1 / (1ᵀΣ⁻¹1)
สังเกตว่าคำตอบสุดท้ายต้องพึ่ง matrix inversion ของ Σ ทั้งหมด — นี่คือจุดที่ปัญหาจริงเกิดขึ้น เพราะถ้า Σ ใกล้ singular (สินทรัพย์มี correlation สูงมาก หรือจำนวนสินทรัพย์เยอะกว่าจำนวนวันที่มีข้อมูล) การ invert จะไม่เสถียร (ill-conditioned) ทำให้ weight ที่ได้สุดโต่งเกินจริง (บางตัว weight เป็น +500%, -400% เป็นต้น) ซึ่งเป็นปัญหาคลาสสิกของ Markowitz ในทางปฏิบัติ
ถ้าเพิ่ม target return constraint (wᵀμ = μ_target) ด้วย จะกลายเป็นระบบ Lagrangian สองตัวคูณ และคำตอบจะเป็น linear combination ของสอง vector คือ Σ⁻¹1 และ Σ⁻¹μ — สร้างเป็น efficient frontier ทั้งเส้นได้จาก 2 portfolio นี้เท่านั้น (Two-Fund Separation Theorem)
เมื่อมี constraint เพิ่ม (no short-selling, w≥0)
โจทย์กลายเป็น Quadratic Programming ที่ไม่มี closed-form ต้องแก้ด้วย numerical method เช่น Active Set method หรือ Interior Point method — ตรงนี้แหละที่ทักษะ computer engineering มีประโยชน์มาก เพราะการ implement solver ที่เร็วและเสถียรสำหรับ QP ขนาดใหญ่ (สินทรัพย์เป็นพันตัว) เป็นงานที่ quant infra team ทำจริง

2. PCA บน Covariance Matrix (แบบละเอียด)
ทำไม PCA ถึงใช้ eigen-decomposition
Σ เป็น symmetric matrix เสมอ (จาก spectral theorem) ดังนั้น diagonalize ได้เป็น:
Σ = V Λ Vᵀ
โดย V คือ matrix ของ eigenvector (orthonormal columns) และ Λ คือ diagonal matrix ของ eigenvalue เรียงจากมากไปน้อย λ₁ ≥ λ₂ ≥ ... ≥ λₙ
แต่ละ eigenvector vᵢ คือ "ทิศทาง" ใน asset-return space ที่ variance ถูก maximize ตามลำดับ และ eigenvalue λᵢ บอกว่า variance ของตลาดทั้งหมดถูกอธิบายโดย factor นั้นกี่เปอร์เซ็นต์:
% variance explained by factor i = λᵢ / Σλⱼ
ทำไม eigenvector ตัวแรกมักเป็น "market factor"
ในทางปฏิบัติ หุ้นเกือบทุกตัวใน correlation matrix มี correlation เป็นบวกกับกันหมด (ตลาดขึ้นพร้อมกัน ลงพร้อมกัน) ทำให้ eigenvector ตัวแรกมัก loading เป็นบวกเกือบทุกตัวใกล้เคียงกัน — พฤติกรรมนี้อธิบายด้วย Perron-Frobenius theorem (matrix ที่มีค่าเป็นบวกทั้งหมดจะมี eigenvector หลักที่ทุก component เป็นบวก) โดยทั่วไป λ₁/Σλ มักอธิบาย variance ของตลาดหุ้นได้ 20-40% เลยทีเดียว
การใช้งานจริง: Statistical Arbitrage

ทำ PCA บน return matrix ของหุ้นในดัชนี (เช่น S&P 500)
ใช้ top-k eigenvector สร้าง "synthetic market" หรือ risk factors
Regress return ของหุ้นแต่ละตัวกับ factor เหล่านี้ ส่วน residual (ที่ factor อธิบายไม่ได้) คือ idiosyncratic component
สร้างสัญญาณเทรดจาก residual ที่ mean-revert (เช่นใช้ Ornstein-Uhlenbeck process fit กับ residual แล้วเทรดเมื่อ residual เบี่ยงเบนจาก mean มากผิดปกติ)

นี่คือหลักการเบื้องหลังกลยุทธ์ statistical arbitrage ของกองทุนใหญ่หลายแห่ง (เช่นแนวทางที่ตีพิมพ์โดย Avellaneda & Lee, 2010)
ข้อควรระวัง
Eigenvector ที่ได้จากข้อมูลจริงมี noise มาก โดยเฉพาะเมื่อ n (จำนวนสินทรัพย์) ใกล้เคียง T (จำนวนวันสังเกต) — เรื่องนี้โยงกับหัวข้อ Random Matrix Theory ด้านล่าง

3. Factor Models & OLS (แบบละเอียด)
การพิสูจน์ที่มาของ OLS formula
โจทย์คือ minimize sum of squared residuals:
S(β) = (r - Xβ)ᵀ(r - Xβ)
ขยายออก:
S(β) = rᵀr - 2βᵀXᵀr + βᵀXᵀXβ
Derivative เทียบ β แล้วให้เท่ากับศูนย์:
∂S/∂β = -2Xᵀr + 2XᵀXβ = 0
→ XᵀXβ = Xᵀr
→ β = (XᵀX)⁻¹Xᵀr
นี่คือ Normal Equation — ในทางปฏิบัติ ไม่มีใครใช้การ invert XᵀX ตรงๆ (คำนวณช้าและไม่เสถียรเชิงตัวเลข) แต่ใช้ QR decomposition หรือ SVD ของ X แทน:
X = QR → β = R⁻¹Qᵀr
เพราะ R เป็น upper triangular ทำให้แก้สมการด้วย back-substitution ได้เร็วกว่าและแม่นยำกว่า invert matrix เต็มรูป
การประยุกต์ใช้: Fama-French 3/5-Factor Model
rᵢ - rf = αᵢ + β₁(Rm-Rf) + β₂·SMB + β₃·HML + εᵢ
แต่ละ β คือ sensitivity ของหุ้นต่อ factor นั้น การประมาณค่าทั้งหมดคือการแก้ระบบ linear equations พร้อมกันสำหรับหุ้นทุกตัว (multivariate regression) ซึ่งถ้าเขียนในรูป matrix จะเป็น B = (XᵀX)⁻¹XᵀR โดย R เป็น matrix ของ return หลายหุ้นพร้อมกัน — ทำทีเดียวได้ทั้งตลาดด้วยการคูณ matrix ครั้งเดียว
การประยุกต์ใช้ในการจัดพอร์ต
เมื่อรู้ factor exposure ของแต่ละหุ้นแล้ว สามารถควบคุมความเสี่ยงของพอร์ตให้ neutral ต่อ factor บางตัวได้ (เช่น market-neutral strategy ที่ β ต่อตลาดรวม ≈ 0) โดยตั้ง constraint เพิ่มในการ optimize weight

4. Cointegration, Kalman Filter & Pairs Trading (แบบละเอียด)
ทำไมใช้ Cointegration แทน Correlation
Correlation สูงระหว่างราคาหุ้นสองตัวไม่ได้แปลว่าเทรดคู่กันได้อย่างปลอดภัย เพราะราคาที่เป็น non-stationary (random walk) อาจ correlate กันเองในระยะสั้นแต่ diverge ออกไปเรื่อยๆ ในระยะยาวได้ (spurious correlation) สิ่งที่ต้องการจริงๆ คือ cointegration: หา linear combination ของราคาสองตัวที่ตัว combination นั้น stationary (mean-reverting)
Johansen Test (ภาพรวม)
ใช้ Vector Error Correction Model (VECM):
Δyₜ = Πy_{t-1} + Σ Γᵢ Δy_{t-i} + εₜ
โจทย์คือหา rank ของ matrix Π ผ่านการทำ eigenvalue decomposition ของ matrix ที่สร้างจาก residual covariance ของสองระบบ regression — eigenvector ที่ได้คือ cointegrating vector ซึ่งบอกว่าต้องผสมราคาสองตัวด้วยสัดส่วนเท่าไหร่ถึงจะได้ series ที่ stationary (นี่คือ "hedge ratio" นั่นเอง)
Kalman Filter สำหรับ Dynamic Hedge Ratio
ปัญหาของ hedge ratio แบบ static (คำนวณครั้งเดียวจาก regression ทั้ง history) คือมันไม่ปรับตามสภาพตลาดที่เปลี่ยน Kalman Filter แก้ปัญหานี้โดยโมเดล hedge ratio เป็น state ที่เปลี่ยนแปลงตามเวลา (state-space model):
State equation: βₜ = βₜ₋₁ + wₜ (ให้ β เปลี่ยนแบบ random walk ช้าๆ)
Observation equation: y₁ₜ = βₜ y₂ₜ + vₜ
ขั้นตอนของ Kalman Filter ทั้งหมดเป็น linear algebra:

Predict: x̂ₜ|ₜ₋₁ = Fx̂ₜ₋₁, Pₜ|ₜ₋₁ = FPₜ₋₁Fᵀ + Q
Update: Kₜ = Pₜ|ₜ₋₁Hᵀ(HPₜ|ₜ₋₁Hᵀ + R)⁻¹ (Kalman Gain)
x̂ₜ = x̂ₜ|ₜ₋₁ + Kₜ(yₜ - Hx̂ₜ|ₜ₋₁)

ทุกขั้นตอนเป็น matrix multiplication และ inversion (แม้ในกรณี 1-D จะลดรูปเหลือ scalar operation แต่ในโมเดลที่ซับซ้อนขึ้น เช่น multi-asset state, matrix เหล่านี้ขยายเป็นหลายมิติจริง) — implement ให้เร็วและเสถียรเชิงตัวเลข เป็นงานที่ quant developer ทำเป็นประจำ

5. Random Matrix Theory & Covariance Denoising (แบบละเอียด)
ปัญหา: Curse of Dimensionality ใน Covariance Estimation
ถ้ามี n สินทรัพย์และสังเกตข้อมูลเพียง T วัน และ n/T ไม่เล็กมาก (เช่น n=500, T=750) sample covariance matrix ที่ประมาณได้จะมี noise มหาศาล โดยเฉพาะ eigenvalue เล็กๆ ที่มักถูกประเมินต่ำเกินจริง และ eigenvalue ใหญ่ที่ถูกประเมินสูงเกินจริง
Marchenko-Pastur Distribution
ถ้าข้อมูล return เป็น pure noise (ไม่มี signal จริง) eigenvalue ของ sample covariance matrix (ที่ normalize แล้ว) จะกระจายตัวตาม Marchenko-Pastur distribution ซึ่งมีขอบเขตบนล่างที่คำนวณได้จาก q = T/n:
λ_max = (1 + √(1/q))²
λ_min = (1 - √(1/q))²
Eigenvalue ใดๆ ที่อยู่ เกิน λ_max ถือว่าเป็น "signal" จริง ส่วนที่อยู่ในช่วง [λ_min, λ_max] ถือว่าเป็น noise
Eigenvalue Clipping
เทคนิคคือ:

Diagonalize sample covariance: Σ = VΛVᵀ
Eigenvalue ที่อยู่ในช่วง noise → แทนที่ด้วยค่าเฉลี่ยของ eigenvalue เหล่านั้น (เพื่อรักษา trace ของ matrix ไว้)
Reconstruct: Σ_clean = VΛ_cleanVᵀ

Ledoit-Wolf Shrinkage (อีกวิธีที่นิยมกว่าในทางปฏิบัติ)
แทนที่จะ clip eigenvalue ตรงๆ ใช้วิธี shrink sample covariance เข้าหา structured target (เช่น diagonal matrix หรือ constant-correlation matrix):
Σ_shrunk = δF + (1-δ)Σ_sample
โดย δ (shrinkage intensity) หาได้จากการ minimize expected Frobenius norm ระหว่าง Σ_shrunk กับ true covariance matrix — มี closed-form solution ที่ Ledoit & Wolf (2004) พิสูจน์ไว้
ผลลัพธ์ของทั้งสองวิธีคือ matrix ที่ "denoise" แล้ว เอาไปใช้ต่อในขั้นตอน Markowitz optimization จะได้ weight ที่เสถียรกว่าการใช้ sample covariance ตรงๆ มาก — นี่คือเทคนิคที่แยกระหว่าง textbook Markowitz กับของจริงที่ hedge fund ใช้

6. VaR & Cholesky Decomposition (แบบละเอียด)
Variance-Covariance VaR
ถ้าสมมติ portfolio return เป็น normal distribution, VaR ที่ confidence level (1-α) คือ:
VaR = zα × √(wᵀΣw) × Portfolio Value
ตรงนี้ต้องใช้ quadratic form เดียวกับที่เจอใน Markowitz
Monte Carlo VaR ต้องใช้ Cholesky
ถ้า asset returns มี correlation กัน การ simulate scenario แบบสุ่มต้องสร้าง correlated random variables ไม่ใช่สุ่มอิสระ วิธีทำคือ:

Decompose Σ = LLᵀ (L เป็น lower triangular matrix จาก Cholesky decomposition)
สุ่ม vector z ของ independent standard normal
Correlated returns: r = μ + Lz

พิสูจน์ว่าวิธีนี้ให้ covariance ถูกต้อง:
Cov(Lz) = L Cov(z) Lᵀ = L I Lᵀ = LLᵀ = Σ ✓
Cholesky decomposition คำนวณเร็วกว่า eigen-decomposition มาก (O(n³/3) เทียบกับ O(n³) ของ eigen-decomposition แบบเต็ม) จึงเป็นที่นิยมสำหรับ Monte Carlo simulation ที่ต้อง simulate เป็นล้าน scenario

7. Black-Litterman Model (แบบละเอียด)
ปัญหาของ Markowitz ที่ Black-Litterman แก้
Markowitz อ่อนไหวต่อ expected return (μ) มาก แค่เปลี่ยน μ นิดเดียว weight ที่ optimal เปลี่ยนไปสุดขั้ว (extreme) เพราะ optimizer จะ "เชื่อ" estimate ที่ noisy เกินไป
แนวทางแก้: Bayesian Updating
Black-Litterman เริ่มจาก equilibrium return (Π) ที่คำนวณย้อนกลับจาก market cap weight (reverse optimization):
Π = δΣw_market
โดย δ คือ risk aversion coefficient
จากนั้นผสมกับ "views" ของนักลงทุน ที่เขียนในรูป matrix:

P: matrix บอกว่า view แต่ละอันเกี่ยวกับ asset ไหนบ้าง
Q: vector ของ expected return ตาม view
Ω: covariance matrix ของความไม่แน่นอนใน view

สูตร Bayesian updating (posterior expected return):
E[r] = [(τΣ)⁻¹ + PᵀΩ⁻¹P]⁻¹ [(τΣ)⁻¹Π + PᵀΩ⁻¹Q]
สมการนี้คือการรวม prior (τΣ, Π) กับ likelihood จาก views (P, Ω, Q) ในกรอบ Bayesian ที่แม่นยำ — ทุกส่วนเป็น matrix operations (inversion หลายชั้น) ผลลัพธ์ posterior return ที่ได้จะ "นุ่มนวล" กว่าการใช้ μ ดิบๆ ทำให้ weight ที่ optimize ออกมาไม่สุดขั้วเหมือน naive Markowitz

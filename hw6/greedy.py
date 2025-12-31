import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.metrics import r2_score

# 設定中文字型，避免繪圖時顯示亂碼 (依據你的作業系統調整)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # Windows 常用
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS'] # macOS 常用
plt.rcParams['axes.unicode_minus'] = False

# =========================================
# 1. 產生模擬資料 (Ground Truth)
# =========================================
# 我們製造一個高維度資料，但只有少數特徵是有用的 (Sparse Signal)
n_samples = 100    # 樣本數
n_features = 50    # 總特徵數 (維度)
n_informative = 5  # 真正有用的特徵數 (只有5個特徵真正影響結果)
noise_level = 5.0  # 雜訊大小

# coef=True 會回傳真實的係數向量，讓我們知道答案是什麼
X, y, true_coef = make_regression(n_samples=n_samples, 
                                  n_features=n_features,
                                  n_informative=n_informative, 
                                  noise=noise_level,
                                  coef=True, 
                                  random_state=42)

print(f"資料集維度: {X.shape}")
print(f"真實有用的特徵數量: {np.sum(true_coef != 0)}")
print("-" * 30)

# =========================================
# 2. 建立並訓練貪婪模型 (OMP)
# =========================================
# 這裡我們明確告訴貪婪演算法：請只挑選出「5個」最重要的特徵
# 這就是貪婪法的核心：逐步挑選對殘差解釋力最強的特徵，直到達到指定數量
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_informative)

print("開始執行貪婪法 (OMP)...")
omp.fit(X, y)

# 取得預測結果與估計的係數
y_pred = omp.predict(X)
estimated_coef = omp.coef_

# 計算模型表現分數
score = r2_score(y, y_pred)
print(f"貪婪法完成。")
print(f"模型解釋力 (R^2 Score): {score:.4f} (越接近1越好)")
print(f"貪婪法挑選出的非零係數數量: {np.sum(estimated_coef != 0)}")

# =========================================
# 3. 繪圖展示結果
# =========================================
plt.figure(figsize=(14, 6))

# --- 子圖 1: 真實值 vs. 預測值 散佈圖 ---
plt.subplot(1, 2, 1)
plt.scatter(y, y_pred, color='royalblue', alpha=0.7, edgecolors='k', s=60, label='資料點')
# 畫出一條完美的對角線作為參考
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='完美預測線')
plt.title('貪婪法回歸效果：真實值 vs. 預測值', fontsize=14, fontweight='bold')
plt.xlabel('真實數值 (Actual y)', fontsize=12)
plt.ylabel('預測數值 (Predicted y)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# --- 子圖 2: 係數比較圖 (Stem Plot) ---
# 這個圖展示貪婪法是否成功「抓到」了真正重要的特徵
plt.subplot(1, 2, 2)

# 繪製真實的係數 (Ground Truth) - 紅色圓點
# 只繪製非零的部分以保持圖面整潔
true_idx = np.where(true_coef != 0)[0]
plt.stem(true_idx, true_coef[true_idx], 
         linefmt='r-', markerfmt='ro', basefmt=' ', label='真實係數 (Ground Truth)')

# 繪製貪婪法估計的係數 (Estimated) - 藍色叉叉
# 為了視覺上不要完全重疊，將索引稍微偏移 +0.2
est_idx = np.where(estimated_coef != 0)[0]
plt.stem(est_idx + 0.2, estimated_coef[est_idx], 
         linefmt='b--', markerfmt='bx', basefmt=' ', label='貪婪法估計 (OMP)')

plt.title('特徵選擇能力：真實係數 vs. 貪婪法估計', fontsize=14, fontweight='bold')
plt.xlabel('特徵索引 (Feature Index)', fontsize=12)
plt.ylabel('係數值 (Coefficient Value)', fontsize=12)

# 為了讓圖表更清晰，只顯示有意義的特徵索引刻度
combined_idx = np.union1d(true_idx, est_idx)
plt.xticks(combined_idx, labels=[str(i) for i in combined_idx])

plt.legend()
plt.grid(True, linestyle='--', axis='y', alpha=0.6)

plt.tight_layout()
plt.show()
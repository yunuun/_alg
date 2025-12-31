import random
import matplotlib.pyplot as plt
import numpy as np

# 為了讓圖表正常顯示中文 (如果你的環境顯示亂碼，請註解掉這兩行或自行設定字體)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 用於 Windows
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 第一部分：資料準備與核心函數
# ==========================================

# 1. 準備數據 (Ground Truth)
# 假設真實世界有一條線 y = 2x + 5，加上一些隨機雜訊
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# 這裡為了讓每次執行結果一致，我把雜訊固定下來，實際應用時雜訊是隨機的
noise = np.array([-0.8, 0.1, 0.5, -0.2, 1.1, 0.3, -1.0, 0.2, 1.5, -0.2])
Y = 2 * X + 5 + noise
# 實際的 Y 值大約是: [7.2, 9.1, 11.5, 12.8, 15.1, 17.3, 19.0, 21.2, 23.5, 24.8]

# 2. 定義損失函數 (Loss Function) - 使用 MSE (均方誤差)
def compute_loss(w, b, x_data, y_data):
    total_error = 0
    N = len(x_data)
    for i in range(N):
        # 用目前的 w, b 預測出來的 y
        y_pred = w * x_data[i] + b
        # 累加 (預測值 - 真實值) 的平方
        total_error += (y_pred - y_data[i]) ** 2
    # 取平均
    return total_error / N

# ==========================================
# 第二部分：爬山演算法主迴圈
# ==========================================

# 參數設定
w = random.uniform(-5, 5)    # 隨機起點 w
b = random.uniform(0, 10)    # 隨機起點 b
step_size = 0.2              # 步伐大小 (每次嘗試微調的幅度)
iterations = 2000            # 迭代次數 (嘗試幾次)

current_loss = compute_loss(w, b, X, Y)
print(f"【初始狀態】 w={w:.2f}, b={b:.2f}, Loss={current_loss:.2f}")

print("爬山中...")
# 開始迭代
for i in range(iterations):
    # 1. 試探：隨機產生一個鄰居 (稍微改變 w 和 b)
    # 這裡我們同時對 w 和 b 進行一個小幅度的隨機增減
    w_try = w + random.uniform(-step_size, step_size)
    b_try = b + random.uniform(-step_size, step_size)
    
    # 2. 評估：計算鄰居的 Loss
    loss_try = compute_loss(w_try, b_try, X, Y)
    
    # 3. 貪婪選擇：如果鄰居的 Loss 比較小 (比較好)，就移動過去
    if loss_try < current_loss:
        w = w_try
        b = b_try
        current_loss = loss_try
        # (選擇性) 偶爾印出進度
        if (i+1) % 200 == 0:
             print(f"Iteration {i+1}: Loss 降至 {current_loss:.4f}")

print("-" * 30)
print(f"【最終結果】")
print(f"找到的最佳參數: w = {w:.3f}, b = {b:.3f}")
print(f"最終最低 Loss : {current_loss:.4f}")
print(f"真實目標線約為: w = 2.000, b = 5.000")
print("-" * 30)

# ==========================================
# 第三部分：繪製結果圖
# ==========================================

plt.figure(figsize=(8, 6)) # 設定圖片大小

# 1. 繪製原始數據點 (散點圖)
plt.scatter(X, Y, color='blue', label='真實數據點 (Real Data)', s=50, zorder=2)

# 2. 繪製我們找到的迴歸直線
# 利用最終找到的 w 和 b，算出對應 X 的預測值 Y_pred
Y_pred = w * X + b
# 畫出紅色的直線
plt.plot(X, Y_pred, color='red', linewidth=3, label=f'爬山法預測線 (y={w:.2f}x+{b:.2f})', zorder=1)

# 3. 加入圖表標示
plt.title('線性回歸：使用爬山演算法 (Linear Regression via Hill Climbing)', fontsize=14)
plt.xlabel('X 軸數值', fontsize=12)
plt.ylabel('Y 軸數值', fontsize=12)
plt.legend(fontsize=11) # 顯示圖例
plt.grid(True, linestyle='--', alpha=0.6) # 加入格線

# 4. 顯示圖表
plt.show()
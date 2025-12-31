import numpy as np
import matplotlib.pyplot as plt
import random
import math

# ==========================================
# 1. 準備數據 (Data Generation)
# ==========================================
# 設定隨機種子以確保結果可重現
np.random.seed(42)

# 產生 X 數據 (0 到 10 之間共 50 個點)
X = np.linspace(0, 10, 50)
# 設定真實的斜率和截距 (這是我們希望演算法找到的答案)
true_w = 2.5
true_b = 5.0
# 產生真實的 Y 值，並加入一些高斯雜訊
noise = np.random.randn(50) * 1.5
Y = true_w * X + true_b + noise

# ==========================================
# 2. 定義能量函數 (Cost Function)
# ==========================================
# 這裡使用均方誤差 (MSE) 作為能量
def calculate_mse(w, b, X_data, Y_data):
    # 預測值
    Y_pred = w * X_data + b
    # 計算誤差平方的平均
    mse = np.mean((Y_data - Y_pred)**2)
    return mse

# ==========================================
# 3. 模擬退火演算法 (Simulated Annealing)
# ==========================================

# --- SA 參數設定 ---
initial_temp = 100.0    # 初始溫度 (越高越容易接受差的解)
final_temp = 0.001      # 終止溫度
cooling_rate = 0.95     # 冷卻速率 (每次迭代溫度乘以這個數)
iterations_per_temp = 50 # 每個溫度下的迭代次數 (增加穩定性)

# --- 初始解 ---
# 隨機猜測一個初始的斜率和截距
current_w = np.random.uniform(-5, 5)
current_b = np.random.uniform(-5, 5)
current_energy = calculate_mse(current_w, current_b, X, Y)

# 紀錄目前找到最好的解
best_w = current_w
best_b = current_b
best_energy = current_energy

# 紀錄能量變化過程 (用於分析)
energy_history = [current_energy]

print(f"初始猜測: w={current_w:.2f}, b={current_b:.2f}, MSE={current_energy:.2f}")

# --- 主迴圈 ---
temp = initial_temp
while temp > final_temp:
    
    for _ in range(iterations_per_temp):
        # a. 產生鄰居解 (微調)
        # 在目前的 w 和 b 上加上一個小的隨機變動
        # step_size 控制微調的幅度，隨著溫度降低，步幅也可以稍微減小
        step_size = temp * 0.1 if temp < 1 else 0.1
        
        next_w = current_w + np.random.uniform(-1, 1) * step_size
        next_b = current_b + np.random.uniform(-1, 1) * step_size
        
        # b. 計算新解的能量
        next_energy = calculate_mse(next_w, next_b, X, Y)
        
        # c. 計算能量差 (Delta E)
        delta_energy = next_energy - current_energy
        
        # d. 決定是否接受新解
        # 情況 1: 新解比較好 (能量變低)，無條件接受
        if delta_energy < 0:
            current_w = next_w
            current_b = next_b
            current_energy = next_energy
            # 更新全域最佳解
            if current_energy < best_energy:
                best_energy = current_energy
                best_w = current_w
                best_b = current_b

        # 情況 2: 新解比較差 (能量變高)，以特定機率接受
        # 機率公式 P = exp(-DeltaE / T)
        else:
            acceptance_probability = math.exp(-delta_energy / temp)
            # 生成一個 0 到 1 的隨機數來決定
            if random.random() < acceptance_probability:
                current_w = next_w
                current_b = next_b
                current_energy = next_energy
                # 注意：這裡雖然接受了差的解作為下一步的起點，但不更新 best_energy

    # 紀錄每個溫度結束後的能量
    energy_history.append(current_energy)
            
    # e. 降溫 (Cooling)
    temp *= cooling_rate
    # (選擇性) 打印進度
    # print(f"Temp: {temp:.4f}, Best MSE: {best_energy:.4f}")

print("-" * 30)
print(f"真實參數: w={true_w}, b={true_b}")
print(f"SA 找到的最佳參數: w={best_w:.2f}, b={best_b:.2f}")
print(f"最終 MSE: {best_energy:.4f}")

# ==========================================
# 4. 視覺化結果 (Visualization)
# ==========================================
plt.figure(figsize=(10, 5))

# --- 子圖 1: 線性回歸結果 ---
plt.subplot(1, 2, 1)
# 畫出原始數據點
plt.scatter(X, Y, color='blue', label='Data Points (with noise)', alpha=0.6)
# 畫出真實的線 (參考用)
plt.plot(X, true_w * X + true_b, color='green', linestyle='--', label='True Line', linewidth=2)
# 畫出 SA 找到的最佳擬合線
X_fit = np.array([min(X), max(X)])
Y_fit = best_w * X_fit + best_b
plt.plot(X_fit, Y_fit, color='red', label=f'SA Fitted Line (w={best_w:.1f}, b={best_b:.1f})', linewidth=2)

plt.title('Linear Regression using Simulated Annealing')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)

# --- 子圖 2: 能量收斂過程 ---
plt.subplot(1, 2, 2)
plt.plot(energy_history, color='purple')
plt.title('MSE Loss History over Temperature Steps')
plt.xlabel('Temperature Steps (Cooling Iterations)')
plt.ylabel('Mean Squared Error (MSE)')
plt.yscale('log') # 使用對數座標軸看清收斂過程
plt.grid(True)


plt.show()
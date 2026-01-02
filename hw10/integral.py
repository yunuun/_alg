import numpy as np
import itertools
import time

class N_Dimensional_Integrator:
    def __init__(self, func, bounds):
        """
        :param func: 被積函數 f(x)，x 為一個 list 或 array
        :param bounds: 積分範圍，格式為 [[min1, max1], [min2, max2], ...]
        """
        self.func = func
        self.bounds = np.array(bounds)
        self.dim = len(bounds) # N 維
        
        # 計算總體積 (Volume of the integration region)
        # 這是所有維度長度的乘積： (max1-min1) * (max2-min2) * ...
        self.volume = np.prod(self.bounds[:, 1] - self.bounds[:, 0])

    # ---------------------------------------------------------
    # 方法 1: 黎曼積分 (Riemann Sum / Grid Method)
    # ---------------------------------------------------------
    def riemann_integration(self, grid_points_per_dim=10):
        """
        使用中點法則 (Midpoint Rule) 的黎曼和。
        注意：總計算次數是 (grid_points_per_dim) 的 N 次方，維度高時會極慢。
        """
        print(f"--- 開始黎曼積分 (維度 N={self.dim}, 每維切 {grid_points_per_dim} 份) ---")
        
        # 1. 準備每個維度的切分點 (取每個小格子的中點以提高精度)
        # 生成每個維度的座標軸點
        axis_points = []
        deltas = []
        
        for i in range(self.dim):
            low, high = self.bounds[i]
            # linspace 產生邊界，我們需要取區間中點
            edges = np.linspace(low, high, grid_points_per_dim + 1)
            midpoints = (edges[:-1] + edges[1:]) / 2
            axis_points.append(midpoints)
            
            # 計算該維度的 dx (寬度)
            deltas.append((high - low) / grid_points_per_dim)
            
        # 2. 計算單個小超立方體的體積 dV = dx1 * dx2 * ... * dxn
        dV = np.prod(deltas)
        
        # 3. 使用 itertools.product 生成 N 維網格的所有組合
        # 這解決了 "不知道 N 是多少，無法寫 N 個 for 迴圈" 的問題
        total_sum = 0.0
        
        # 這裡會迭代 grid_points^N 次
        for point in itertools.product(*axis_points):
            total_sum += self.func(point)
            
        return total_sum * dV

    # ---------------------------------------------------------
    # 方法 2: 蒙地卡羅積分 (Monte Carlo Integration)
    # ---------------------------------------------------------
    def monte_carlo_integration(self, num_samples=100000):
        """
        隨機撒點法。
        計算公式： 總體積 * (採樣點的函數平均值)
        """
        print(f"--- 開始蒙地卡羅積分 (維度 N={self.dim}, 樣本數 {num_samples}) ---")
        
        # 1. 在 N 維空間內生成均勻分佈的隨機點
        # random_points 形狀為 (num_samples, dim)
        lows = self.bounds[:, 0]
        highs = self.bounds[:, 1]
        
        # 利用 NumPy 的廣播機制一次生成所有點
        random_points = np.random.uniform(lows, highs, size=(num_samples, self.dim))
        
        # 2. 計算所有點的函數值
        # apply_along_axis 會將每一列 (一個 N 維點) 傳入 func
        f_values = np.apply_along_axis(self.func, 1, random_points)
        
        # 3. 計算平均高度
        mean_height = np.mean(f_values)
        
        # 4. 積分值 = 區域體積 * 平均高度
        return self.volume * mean_height

# ==========================================
# 測試案例
# ==========================================

if __name__ == "__main__":
    
    # 定義一個 N 維函數，例如：計算 N 維球體的 "質量" (假設密度分佈)
    # 這裡簡單用 f(x) = sum(x^2) 在 [0,1]^N 上的積分
    # 理論值：對於 f(x) = x1^2 + ... + xn^2 在 [0,1]^N
    # 積分結果應該是 N/3
    
    N = 4  # 設定維度 (你可以改成 4, 5, 6 試試看)
    
    def target_func(point):
        return np.sum(np.array(point) ** 2)

    # 設定邊界：每個維度都是 [0, 1]
    bounds = [[0, 1]] * N 
    
    integrator = N_Dimensional_Integrator(target_func, bounds)
    
    print(f"理論值應該是: {N/3:.5f}\n")

    # --- 1. 執行黎曼積分 ---
    start = time.time()
    # 注意：如果 N 很大，grid_points 必須設很小，否則記憶體會爆
    riemann_result = integrator.riemann_integration(grid_points_per_dim=50) 
    print(f"黎曼積分結果: {riemann_result:.5f}")
    print(f"耗時: {time.time() - start:.4f} 秒\n")

    # --- 2. 執行蒙地卡羅積分 ---
    start = time.time()
    mc_result = integrator.monte_carlo_integration(num_samples=125000) # 為了公平，讓運算次數接近 (50^3 = 125000)
    print(f"蒙地卡羅結果: {mc_result:.5f}")
    print(f"耗時: {time.time() - start:.4f} 秒")
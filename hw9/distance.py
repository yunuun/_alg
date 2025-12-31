def min_edit_distance(str1, str2):
    """
    計算兩個字串之間的最小編輯距離 (Levenshtein Distance)
    :param str1: 來源字串
    :param str2: 目標字串
    :return: 最小編輯次數
    """
    m = len(str1)
    n = len(str2)

    # 建立一個 (m+1) x (n+1) 的二維陣列 (DP Table)
    # dp[i][j] 代表 str1 的前 i 個字元轉換成 str2 的前 j 個字元所需的最小距離
    dp = [[0 for x in range(n + 1)] for x in range(m + 1)]

    # 初始化邊界條件
    # 如果 str2 是空的，str1 需要刪除 i 個字元才能變空
    for i in range(m + 1):
        dp[i][0] = i

    # 如果 str1 是空的，需要插入 j 個字元才能變成 str2
    for j in range(n + 1):
        dp[0][j] = j

    # 開始填滿 DP 表格
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # 如果當前字元相同，不需要操作，距離等於左上角的值
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                # 如果不同，取三種操作中的最小值 + 1
                dp[i][j] = 1 + min(
                    dp[i - 1][j],    # 刪除 (Deletion)
                    dp[i][j - 1],    # 插入 (Insertion)
                    dp[i - 1][j - 1] # 替換 (Substitution)
                )

    # 右下角的值即為最終結果
    return dp[m][n]

# --- 測試範例 ---
if __name__ == "__main__":
    s1 = "distance"
    s2 = "assistant"
    
    dist = min_edit_distance(s1, s2)
    print(f"'{s1}' 轉換成 '{s2}' 的最小編輯距離為: {dist}")
    # 結果應為 3 (h->r, 刪除 r, e->s ??? 實際上是: h->r, delete r, delete e... 不，最優解是: horse -> rorse (h換r) -> rose (刪r) -> ros (刪e))
    # 更正範例解釋: horse -> rorse (sub 'h'->'r') -> rose (del 'r') -> ros (del 'e') => 3
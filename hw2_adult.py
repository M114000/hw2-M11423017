import pandas as pd
import time
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt


# 官方 15 個欄位名稱
column_names = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income"
]

# -----------------------------
# 讀取已切割好的 CSV
# -----------------------------
train_df = pd.read_csv("train_split_data.csv")
test_df = pd.read_csv("test_split_data.csv")

# -----------------------------
# 如果想要從原始 adult.data 重新切割，可以取消註解以下程式
# -----------------------------
# try:
#     data_df = pd.read_csv(
#         "adult.data",
#         header=None,
#         names=column_names,
#         sep=', ',
#         engine='python',
#         na_values='?'
#     )
#     print(f"成功讀取 adult.data，總共 {len(data_df)} 筆資料。")
# except FileNotFoundError:
#     print("錯誤：找不到 'adult.data' 檔案。")
#
# # --- 80/20 切割 ---
# if 'data_df' in locals():
#     train_df, test_df = train_test_split(
#         data_df,
#         test_size=0.2,
#         random_state=42
#     )
#
#     # --- 直接刪除缺失值 ---
#     train_df = train_df.dropna()
#     test_df = test_df.dropna()
#
#     # --- 儲存切割後資料 ---
#     train_df.to_csv("train_split_data.csv", index=False)
#     test_df.to_csv("test_split_data.csv", index=False)

# -----------------------------
# 模型訓練與績效計算
# -----------------------------
target = "hours-per-week"
features = [col for col in train_df.columns if col != target]

X_train = train_df[features].copy()
y_train = train_df[target].copy()
X_test = test_df[features].copy()
y_test = test_df[target].copy()

# 編碼類別欄位
cat_cols = X_train.select_dtypes(include=['object']).columns
for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# 標準化數值欄位
num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# 定義模型
models = {
    "KNN": KNeighborsRegressor(n_neighbors=5),
    "SVR": SVR(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

results = []

# 訓練模型並計算績效指標
for name, model in models.items():
    start_time = time.time()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elapsed_time = time.time() - start_time

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    results.append((name, rmse, mape, r2, elapsed_time))

# 顯示結果
print(f"{'Model':<15} {'RMSE':<10} {'MAPE(%)':<10} {'R2':<10} {'Time(s)':<10}")
for r in results:
    print(f"{r[0]:<15} {r[1]:<10.3f} {r[2]:<10.2f} {r[3]:<10.3f} {r[4]:<10.3f}")

# -----------------------------
# 可視化績效柱狀圖（改成四張圖）
# -----------------------------
models_names = [r[0] for r in results]
rmse_values = [r[1] for r in results]
mape_values = [r[2] for r in results]
r2_values = [r[3] for r in results]
time_values = [r[4] for r in results]

# 四個指標名稱與資料
metrics = {
    "RMSE": rmse_values,
    "MAPE (%)": mape_values,
    "R²": r2_values,
    "Time (s)": time_values
}

# 顏色設定
colors = ['#66b3ff', '#ffb366', '#99cc99', '#ff9999']

# 建立畫布
plt.figure(figsize=(12, 7))

# x 軸位置設定
x = np.arange(len(models_names))  # 模型位置
bar_width = 0.2                   # 每根柱子的寬度

# 每個指標偏移
for i, (metric_name, values) in enumerate(metrics.items()):
    plt.bar(x + i * bar_width, values, width=bar_width, color=colors[i],
            label=metric_name, edgecolor='black', linewidth=1)

# x 軸與標籤
plt.xticks(x + bar_width * 1.5, models_names, fontsize=12)
plt.xlabel('Model', fontsize=14)
plt.ylabel('Metric Value', fontsize=14)
plt.title('Performance Comparison of Four Regression Models\nTarget Variable: hours-per-week',
          fontsize=16, fontweight='bold', pad=15)

# 顯示數值標籤
for i, (metric_name, values) in enumerate(metrics.items()):
    for j, v in enumerate(values):
        plt.text(x[j] + i * bar_width, v + (max(values)*0.02), f'{v:.2f}',
                 ha='center', va='bottom', fontsize=10, rotation=0)

# 圖例與美化
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
print(f"{'Model':<15} {'RMSE':<10} {'MAPE(%)':<10} {'R2':<10} {'Time(s)':<10}")
for r in results:
    print(f"{r[0]:<15} {r[1]:<10.3f} {r[2]:<10.2f} {r[3]:<10.3f} {r[4]:<10.3f}")
plt.savefig('performance_hours_per_week.png', dpi=300, bbox_inches='tight')

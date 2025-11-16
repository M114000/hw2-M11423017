import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

# ===============================
# 1. 讀取資料
# ===============================
df = pd.read_csv("BostonHousing.csv")
df.columns = df.columns.str.strip()
target_col = "medv"
X = df.drop(columns=[target_col])
y = df[target_col]

print("模型說明: XGBoost + KFold + SHAP 特徵刪減比較")

# ===============================
# 2. MAPE 定義
# ===============================
def mape(y_true, y_pred):
    y_true_safe = np.where(y_true == 0, 1e-10, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100

# ===============================
# 3. K-fold 評估函數
# ===============================
def run_kfold_and_plot(X_input, y_input, title_suffix="all_features"):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    mape_list, rmse_list, r2_list = [], [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_input), 1):
        X_train, X_test = X_input.iloc[train_idx], X_input.iloc[test_idx]
        y_train, y_test = y_input.iloc[train_idx], y_input.iloc[test_idx]

        model = xgb.XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mape_list.append(mape(y_test, y_pred))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_list.append(r2_score(y_test, y_pred))

        print(f"Fold {fold}: MAPE={mape_list[-1]:.4f}, RMSE={rmse_list[-1]:.4f}, R2={r2_list[-1]:.4f}")

    # 平均
    print(f"\n[{title_suffix}] 5-Fold 平均績效：")
    print(f"MAPE 平均 = {np.mean(mape_list):.4f}")
    print(f"RMSE 平均 = {np.mean(rmse_list):.4f}")
    print(f"R2   平均 = {np.mean(r2_list):.4f}")

    folds = range(1, 6)

    # --- 折線圖：MAPE ---
    plt.figure(figsize=(7, 5))
    plt.plot(folds, mape_list, marker='o', label="MAPE per fold")
    plt.axhline(np.mean(mape_list), linestyle="--", color="red", label="Average MAPE")
    plt.title(f"5-Fold MAPE ({title_suffix})")
    plt.xlabel("Fold")
    plt.ylabel("MAPE (%)")
    plt.xticks(folds)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title_suffix}_mape.png", dpi=300)
    plt.close()

    # --- 折線圖：RMSE ---
    plt.figure(figsize=(7, 5))
    plt.plot(folds, rmse_list, marker='o', label="RMSE per fold")
    plt.axhline(np.mean(rmse_list), linestyle="--", color="red", label="Average RMSE")
    plt.title(f"5-Fold RMSE ({title_suffix})")
    plt.xlabel("Fold")
    plt.ylabel("RMSE")
    plt.xticks(folds)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title_suffix}_rmse.png", dpi=300)
    plt.close()

    # --- 折線圖：R2 ---
    plt.figure(figsize=(7, 5))
    plt.plot(folds, r2_list, marker='o', label="R2 per fold")
    plt.axhline(np.mean(r2_list), linestyle="--", color="red", label="Average R2")
    plt.title(f"5-Fold R2 ({title_suffix})")
    plt.xlabel("Fold")
    plt.ylabel("R2")
    plt.ylim(0,1)
    plt.xticks(folds)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{title_suffix}_r2.png", dpi=300)
    plt.close()

    print(f"{title_suffix} 折線圖已生成！\n")
    return mape_list, rmse_list, r2_list

# ===============================
# 4. 全部特徵 K-Fold
# ===============================
print("\n===== 使用全部特徵進行 K-Fold =====")
m_all, rm_all, r2_all = run_kfold_and_plot(X, y, "all_features")

# ===============================
# 5. SHAP 特徵重要性
# ===============================
print("\n===== 開始計算 SHAP 特徵重要性 =====")
model_full = xgb.XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42
)
model_full.fit(X, y)

explainer = shap.TreeExplainer(model_full)
shap_values = explainer.shap_values(X)

# 平均 SHAP 絕對值
shap_abs_mean = np.mean(np.abs(shap_values), axis=0)
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "SHAP Importance": shap_abs_mean
}).sort_values("SHAP Importance", ascending=False)

print("\n=== SHAP 特徵重要性排名（前 10） ===")
print(importance_df.head(10))

# SHAP summary 圖
shap.summary_plot(shap_values, X, feature_names=X.columns, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=300)
plt.close()

# SHAP bar 圖
shap.summary_plot(shap_values, X, feature_names=X.columns, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig("shap_bar.png", dpi=300)
plt.close()

print("已輸出：shap_summary.png、shap_bar.png")

# ===============================
# 6. 取 SHAP 前 5 特徵
# ===============================
top5_features = importance_df["Feature"].head(5).tolist()
print("\n=== SHAP 前五名特徵 ===")
print(top5_features)
X_top5 = X[top5_features]
print("X_top5 shape:", X_top5.shape)

# ===============================
# 7. 前 5 特徵 K-Fold
# ===============================
print("\n===== 使用 SHAP 前 5 特徵進行 K-Fold =====")
m_5, rm_5, r2_5 = run_kfold_and_plot(X_top5, y, "shap_top5_features")

# ===============================
# 8. 最終績效比較
# ===============================
print("\n==============================")
print("   全特徵 vs SHAP 前 5 特徵")
print("==============================")
print(f"MAPE： All={np.mean(m_all):.4f}  Top5={np.mean(m_5):.4f}")
print(f"RMSE： All={np.mean(rm_all):.4f}  Top5={np.mean(rm_5):.4f}")
print(f"R2：   All={np.mean(r2_all):.4f}  Top5={np.mean(r2_5):.4f}")

# ===============================
# 9. 全特徵 vs 前5特徵折線圖比較
# ===============================
folds = range(1, 6)

# MAPE 比較
plt.figure(figsize=(7, 5))
plt.plot(folds, m_all, marker='o', label='All Features')
plt.plot(folds, m_5, marker='s', label='SHAP Top5')
plt.axhline(np.mean(m_all), linestyle='--', color='blue', alpha=0.5, label='All Avg')
plt.axhline(np.mean(m_5), linestyle='--', color='orange', alpha=0.5, label='Top5 Avg')
plt.title("MAPE Comparison: All Features vs SHAP Top5")
plt.xlabel("Fold")
plt.ylabel("MAPE (%)")
plt.xticks(folds)
plt.legend()
plt.tight_layout()
plt.savefig("comparison_mape.png", dpi=300)
plt.close()

# RMSE 比較
plt.figure(figsize=(7, 5))
plt.plot(folds, rm_all, marker='o', label='All Features')
plt.plot(folds, rm_5, marker='s', label='SHAP Top5')
plt.axhline(np.mean(rm_all), linestyle='--', color='blue', alpha=0.5, label='All Avg')
plt.axhline(np.mean(rm_5), linestyle='--', color='orange', alpha=0.5, label='Top5 Avg')
plt.title("RMSE Comparison: All Features vs SHAP Top5")
plt.xlabel("Fold")
plt.ylabel("RMSE")
plt.xticks(folds)
plt.legend()
plt.tight_layout()
plt.savefig("comparison_rmse.png", dpi=300)
plt.close()

# R2 比較
plt.figure(figsize=(7, 5))
plt.plot(folds, r2_all, marker='o', label='All Features')
plt.plot(folds, r2_5, marker='s', label='SHAP Top5')
plt.axhline(np.mean(r2_all), linestyle='--', color='blue', alpha=0.5, label='All Avg')
plt.axhline(np.mean(r2_5), linestyle='--', color='orange', alpha=0.5, label='Top5 Avg')
plt.title("R2 Comparison: All Features vs SHAP Top5")
plt.xlabel("Fold")
plt.ylabel("R2")
plt.ylim(0,1)
plt.xticks(folds)
plt.legend()
plt.tight_layout()
plt.savefig("comparison_r2.png", dpi=300)
plt.close()

print("✅ 全特徵 vs 前5特徵比較折線圖已生成：comparison_mape.png / comparison_rmse.png / comparison_r2.png")

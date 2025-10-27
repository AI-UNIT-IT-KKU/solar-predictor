# =========================================================
# 📦 Imports
# =========================================================
import pandas as pd
import numpy as np
import re
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from math import sqrt
from scipy.stats import spearmanr
from sklearn.model_selection import (
    GroupKFold,
    GridSearchCV,
    TimeSeriesSplit,
    train_test_split
)
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

sns.set_style("whitegrid")

# =========================================================
# 🧩 Column Standardization Utilities
# =========================================================
def canonicalize_column_name(name: str) -> str:
    """
    Normalize column names across multiple files.

    Steps:
        - Remove numeric suffixes like '.1', '.2', etc.
        - Add missing spaces before parentheses.
        - Normalize unit casing (e.g., 'W/M2' → 'W/m2').
        - Collapse multiple spaces/newlines/tabs.
        - Strip leading/trailing whitespace.

    Args:
        name (str): Raw column name.

    Returns:
        str: Cleaned and standardized column name.
    """
    s = str(name)
    s = re.sub(r"\.(\d+)$", "", s)
    s = re.sub(r"(?<=\w)\(", " (", s)
    s = re.sub(r"\s+", " ", s)
    s = s.replace("W/M2", "W/m2")
    s = s.strip()
    return s


def standardize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply column name cleaning to a DataFrame."""
    df = df.copy()
    df.columns = [canonicalize_column_name(c) for c in df.columns]
    return df


# =========================================================
# 🧹 Outlier Removal Utility
# =========================================================
def remove_outliers_iqr(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from numeric columns using the IQR method.
    Works generically for any numeric dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.
        multiplier (float): IQR multiplier (1.5 typical, 3 for lenient filtering).

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - multiplier * IQR
        upper = Q3 + multiplier * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df


# =========================================================
# ⚙️ Data Loading & Integration
# =========================================================
def load_and_merge_training_data(file_list, resample_freq: str = None) -> pd.DataFrame:
    """
    Load multiple files, clean columns, remove outliers, align on timestamp,
    and compute per-timestamp median across devices.

    Args:
        file_list (list): List of pickle file paths.
        resample_freq (str): Optional pandas frequency string (e.g., '1min', '5min')
                             to resample each file before merging.

    Returns:
        pd.DataFrame: Clean, aligned, and merged dataset.
    """
    dfs = []
    for f in file_list:
        try:
            df = pd.read_pickle(f)
            df = standardize_dataframe_columns(df)

            # Ensure timestamp column exists and is parsed
            if "timestamp" not in df.columns and "Time" in df.columns:
                df = df.rename(columns={"Time": "timestamp"})
            if "timestamp" not in df.columns:
                raise ValueError(f"{f} has no timestamp/Time column after standardization.")

            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

            # Remove outliers
            df = remove_outliers_iqr(df, multiplier=1.5)

            # Optionally resample to uniform time intervals
            if resample_freq:
                df = df.set_index("timestamp").resample(resample_freq).mean().reset_index()

            # Deduplicate timestamps by taking median of numeric columns
            num_cols = df.select_dtypes(include="number").columns.tolist()
            agg = {c: "median" for c in num_cols}
            df = df.groupby("timestamp", as_index=False).agg(agg)

            df = df.set_index("timestamp")
            dfs.append(df)
            print(f"✅ Loaded {f}: {df.shape}")
        except Exception as e:
            print(f"❌ Failed to load {f}: {type(e).__name__} - {e}")

    if len(dfs) == 0:
        raise ValueError("No files could be successfully loaded.")

    # Align all data on timestamp (outer join), then compute median across devices
    wide = pd.concat(dfs, axis=1, keys=[f"S{i+1}" for i in range(len(dfs))])
    merged = wide.groupby(level=1, axis=1).median().reset_index()

    print(f"\n✅ Merged {len(dfs)} files → Final shape: {merged.shape}")
    return merged

# =========================================================
# 2️⃣ Data Cleaning
# =========================================================
def clean_dataset(
    df: pd.DataFrame,
    target_col: str = "active_power",
    time_col: str = "timestamp",
    phys_limits: dict = None,
    fix_negatives: bool = True,
    interpolate_time: bool = True,
    day_hours: tuple = (6, 18),
    mode: str = "train",
) -> pd.DataFrame:
    """
    ✅ FIXED: Removed bfill() to prevent data leakage.
    
    Simplified cleaning for time-series data using only physical checks.
    Tracks NaN fixes, physical clipping, and timestamp validity.
    """

    df = df.copy()
    summary = {}

    # 🧮 Count missing values before cleaning
    nans_before = int(df.isna().sum().sum())

    # 1️⃣ Fix negatives in target column
    if target_col not in df.columns:
        raise ValueError(f"Expected '{target_col}' column.")
    if fix_negatives:
        negs = (df[target_col] < 0).sum()
        if negs:
            df[target_col] = np.where(df[target_col] < 0, 0, df[target_col])
        summary["negatives_fixed"] = int(negs)

    # 2️⃣ Parse timestamp and add hour
    if time_col not in df.columns:
        raise ValueError(f"Expected '{time_col}' column.")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    before_time = len(df)
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    summary["invalid_timestamps_removed"] = before_time - len(df)
    df["hour"] = df[time_col].dt.hour
    daytime = df["hour"].between(day_hours[0], day_hours[1])

    # 3️⃣ Detect suspicious daytime zeros
    poa_cols = [c for c in df.columns if any(k in c for k in ["Irradiance", "GHI", "POA"])]
    if poa_cols:
        poa_mean = df[poa_cols].mean(axis=1)
        mask = (poa_mean > 100) & (df[target_col] <= 0) & daytime
        summary["suspicious_daytime_zeros"] = int(mask.sum())
        df.loc[mask, target_col] = np.nan

    # 4️⃣ Apply physical limits
    summary["phys_clipped"] = {}
    for col, limits in (phys_limits or {}).items():
        if col in df.columns and limits:
            low, high = limits
            before = df[col].copy()
            df[col] = df[col].clip(low, high)
            clipped = int((before != df[col]).sum())
            if clipped:
                summary["phys_clipped"][col] = clipped

    # 5️⃣ Interpolation (✅ FIXED - No backward fill!)
    df = df.set_index(time_col)
    num_cols = df.select_dtypes(include="number").columns
    if interpolate_time and num_cols.size > 0:
        if mode == "train":
            # ✅ CRITICAL FIX: Only forward interpolation + forward fill
            df[num_cols] = df[num_cols].interpolate(method="time", limit_direction="forward")
            df[num_cols] = df[num_cols].ffill()  # ONLY forward fill - NO bfill!
        else:
            # Test/production: conservative interpolation
            df[num_cols] = df[num_cols].interpolate(method="time", limit=1, limit_direction="forward")
            df[num_cols] = df[num_cols].ffill()  # ONLY forward fill

    # 6️⃣ Fill for non-numeric columns
    non_num_cols = [c for c in df.columns if c not in num_cols]
    if non_num_cols:
        df[non_num_cols] = df[non_num_cols].ffill()  # Only forward

    df = df.reset_index()

    # 🧮 Count NaN after cleaning
    nans_after = int(df.isna().sum().sum())
    nans_added = summary.get("suspicious_daytime_zeros", 0)
    nans_filled = nans_added + max(0, nans_before - nans_after)

    summary["nans_before"] = nans_before
    summary["remaining_nans"] = nans_after
    summary["nans_fixed"] = nans_filled
    summary["nans_introduced"] = max(0, nans_after - nans_before)

    # 🧾 Summary
    print(f"\n✅ [{mode.upper()}] Cleaning Summary :")
    for k, v in summary.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                print(f"   • {k} → {subk}: {subv:,} values clipped")
        else:
            print(f"   • {k}: {v:,}")
    print(f"✅ Final shape: {df.shape[0]:,} rows × {df.shape[1]:,} cols.\n")

    return df

# =========================================================
# 🧹 Feature Dropping Utility
# =========================================================
def drop_high_target_corr_features(df, target="active_power", corr_thresh=0.95):
    """
    Remove features that are TOO correlated with the target (|corr| > threshold).
    Useful when some inputs are almost duplicates of the target (data leakage).
    """
    df_num = df.select_dtypes(include="number").copy()
    if target not in df_num.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    corr_target = df_num.corrwith(df_num[target]).abs().sort_values(ascending=False)
    high_corr_features = corr_target[corr_target > corr_thresh].index.tolist()

    # Drop all except the target itself
    high_corr_features = [f for f in high_corr_features if f != target]

    print(f"⚠️ Found {len(high_corr_features)} features with |corr| > {corr_thresh} to target:")
    for f in high_corr_features:
        print(f" - {f}: corr={corr_target[f]:.4f}")

    df_clean = df.drop(columns=high_corr_features, errors="ignore")
    return df_clean, high_corr_features

# =========================================================
# 📊 Feature Selection
# =========================================================
def select_features(df, target="active_power", top_k=10):
    """
    Return top features by PCC & MI + agreement stats (Spearman, Jaccard).
    """
    features = [c for c in df.select_dtypes(include="number").columns if c != target]
    if len(features) == 0:
        raise ValueError("No numeric features found.")

    pcc_scores = df[features].corrwith(df[target], method="pearson").abs()
    mi_scores = mutual_info_regression(df[features], df[target], random_state=42)
    mi_series = pd.Series(mi_scores, index=features)

    ranks = pd.DataFrame({"PCC_rank": (-pcc_scores).rank(), "MI_rank": (-mi_series).rank()}).dropna()
    rho, _ = spearmanr(ranks["PCC_rank"], ranks["MI_rank"])

    top_pcc = pcc_scores.sort_values(ascending=False).head(top_k).index.tolist()
    top_mi = mi_series.sort_values(ascending=False).head(top_k).index.tolist()
    jaccard = len(set(top_pcc) & set(top_mi)) / len(set(top_pcc) | set(top_mi)) if (set(top_pcc) | set(top_mi)) else 0.0

    print(f"📊 Spearman correlation (PCC vs MI ranks): {rho:.3f}")
    print(f"🧩 Jaccard similarity (top-{top_k}): {jaccard:.3f}")
    return top_pcc, top_mi, rho, jaccard

# =========================================================
#  Saudi Arabia Season Definition
# =========================================================
def get_saudi_season(month):
    """
    ✅ FIXED: Define seasons according to Saudi Arabia climate.
    - Winter: November to February (cooler months)
    - Spring: March to April (transition)
    - Summer: May to September (extremely hot)
    - Fall: October (short transition)
    """
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:  # 10
        return "fall"

# =========================================================
# ⚡ XGBRegressor Wrapper (no early stopping, stable for CV)
# =========================================================
class XGBStable(XGBRegressor):
    """
    Stable wrapper for XGBRegressor — avoids early stopping inside GridSearchCV.
    This ensures consistent training across all folds (better for time-series CV).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def fit(self, X, y, **fit_params):
        # standard full training (no internal validation split)
        return super().fit(X, y, **fit_params)

# =========================================================
# ⚙️ Grid Search (Time-series CV)
# =========================================================
def balanced_rmse(est, Xv, yv):
    """Negative RMSE (so higher is better for GridSearchCV)."""
    y_pred = est.predict(Xv)
    rmse = np.sqrt(mean_squared_error(yv, y_pred))
    return -rmse

def run_timeSeries_gridsearch_cv(
    model, param_grid, X, y,
    name="Model", cv_splits=5,
    scoring=balanced_rmse, verbose=1, n_jobs=-1
):
    """
    GridSearchCV wrapper using TimeSeriesSplit (best for sequential data).
    Compatible with XGBWithEarlyStopping (no need for a separate final fit).
    """
    print(f"\n🚀 Running GridSearchCV (TimeSeriesSplit={cv_splits}) for {name} ...")
    cv = TimeSeriesSplit(n_splits=cv_splits)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True,
        refit=True  # train best_estimator_ on the full data at the end
    )

    grid.fit(X, y)
    print(f"✅ Best params for {name}: {grid.best_params_}")
    print(f"🏆 Best CV RMSE: {-grid.best_score_:.4f}")
    return grid

def run_multiple_timeSeries_gridsearches(models_with_params, X, y, cv_splits=5):
    """
    Run TS-aware GridSearch for multiple models.
    """
    results = {}
    for name, (model, params) in models_with_params.items():
        results[name] = run_timeSeries_gridsearch_cv(
            model, params, X, y, name=name, cv_splits=cv_splits
        )
    return results

from sklearn.model_selection import RandomizedSearchCV

# =========================================================
# ⚙️ Randomized Search (Time-series CV)
# =========================================================
def run_timeSeries_randomsearch_cv(
    model, param_distributions, X, y,
    name="Model", cv_splits=5,
    scoring=balanced_rmse, verbose=1, n_jobs=-1,
    n_iter=30, random_state=42
):
    """
    RandomizedSearchCV wrapper using TimeSeriesSplit.
    Faster alternative to GridSearchCV for sequential data.
    """
    print(f"\n🚀 Running RandomizedSearchCV (TimeSeriesSplit={cv_splits}) for {name} ...")
    cv = TimeSeriesSplit(n_splits=cv_splits)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        random_state=random_state,
        refit=True  # train best_estimator_ on full data at the end
    )

    search.fit(X, y)
    print(f"✅ Best params for {name}: {search.best_params_}")
    print(f"🏆 Best CV RMSE: {-search.best_score_:.4f}")
    return search


def run_multiple_timeSeries_randomsearches(models_with_params, X, y, cv_splits=5, n_iter=30):
    """
    Run TS-aware RandomizedSearch for multiple models.
    """
    results = {}
    for name, (model, params) in models_with_params.items():
        results[name] = run_timeSeries_randomsearch_cv(
            model, params, X, y, name=name, cv_splits=cv_splits, n_iter=n_iter
        )
    return results


# =========================================================
# 📈 Evaluation (Temporal CV summary)
# =========================================================
def evaluate_best_model(grid, X, y, groups, name, group_labels=None):
    """
    Temporal evaluation summary with TimeSeriesSplit on the chosen best_estimator_.
    ✅ Updated to include both Train and Test metrics per fold.
    """
    model_best = grid.best_estimator_
    tscv = TimeSeriesSplit(n_splits=4)
    rows = []

    for i, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        model_best.fit(X_tr, y_tr)

        y_pred_tr = model_best.predict(X_tr)
        y_pred_te = model_best.predict(X_te)

        rmse_tr = np.sqrt(mean_squared_error(y_tr, y_pred_tr))
        r2_tr = r2_score(y_tr, y_pred_tr)
        rmse_te = np.sqrt(mean_squared_error(y_te, y_pred_te))
        r2_te = r2_score(y_te, y_pred_te)

        rows.append({
            "Fold": group_labels[i] if group_labels and i < len(group_labels) else f"Fold-{i+1}",
            "Train_RMSE": rmse_tr,
            "Train_R2": r2_tr,
            "Test_RMSE": rmse_te,
            "Test_R2": r2_te
        })

    df_eval = pd.DataFrame(rows)
    print(f"\n📊 {name} – Fold Results:")
    print(df_eval)
    print(f"\n{name} → Avg Train RMSE={df_eval['Train_RMSE'].mean():.4f}, "
          f"Avg Test RMSE={df_eval['Test_RMSE'].mean():.4f}, "
          f"Δ={df_eval['Test_RMSE'].mean() - df_eval['Train_RMSE'].mean():.4f}")

    return df_eval


def compare_models_results(evals):
    """
    Compact comparison table across models (average Train/Test metrics)
    + bar plot comparing RMSE between models.
    """
    summary = pd.DataFrame({
        "Model": list(evals.keys()),
        "Train_RMSE": [e["Train_RMSE"].mean() for e in evals.values()],
        "Test_RMSE": [e["Test_RMSE"].mean() for e in evals.values()],
        "Train_R2": [e["Train_R2"].mean() for e in evals.values()],
        "Test_R2": [e["Test_R2"].mean() for e in evals.values()],
    }).sort_values("Test_RMSE")

    summary["RMSE_Gap"] = summary["Test_RMSE"] - summary["Train_RMSE"]

    print("\n================= GRIDSEARCH SUMMARY =================")
    print(summary.to_string(index=False))

    # =========================================================
    # 📊 Visualization: Compare all models (Train vs Test RMSE)
    # =========================================================
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8,6))
    plot_df = summary.melt(
        id_vars="Model",
        value_vars=["Train_RMSE", "Test_RMSE"],
        var_name="Dataset",
        value_name="RMSE"
    )

    sns.barplot(
        data=plot_df,
        x="Model",
        y="RMSE",
        hue="Dataset",
        palette=["#4CAF50", "#3C94E7"]
    )

    plt.title("📉 RMSE Comparison Across Models (Train vs Test)")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.legend(title="Dataset", loc="upper right")
    plt.tight_layout()
    plt.show()

    return summary

# =========================================================
# 🌤️ Leave-One-Season-Out (Analytical only)
# =========================================================
def evaluate_leave_one_season_out_FIXED(grid, X, y, season_series, name):
    """
    Train on 3 seasons, test on the held-out season — purely for analytical insight.
    """
    model_best = grid.best_estimator_
    df = pd.DataFrame({
        "season": season_series.values,
        "y_true": y.values if hasattr(y, "values") else y,
    })
    df_X = X.copy()
    rows = []
    unique_seasons = df["season"].unique()

    print(f"\n######## LEAVE-ONE-SEASON-OUT (Test-Focused) for {name} ########")
    for test_season in unique_seasons:
        tr_idx = df[df["season"] != test_season].index
        te_idx = df[df["season"] == test_season].index

        X_tr, X_te = df_X.iloc[tr_idx], df_X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        model_best.fit(X_tr, y_tr)
        y_pred_te = model_best.predict(X_te)

        rmse_te = np.sqrt(mean_squared_error(y_te, y_pred_te))
        r2_te = r2_score(y_te, y_pred_te)
        mae_te = np.mean(np.abs(y_te - y_pred_te))
        mean_power_te = y_te.mean()
        rmse_te_pct = 100 * rmse_te / mean_power_te if mean_power_te != 0 else np.nan

        rows.append({
            "Season": str(test_season).upper(),
            "Test_RMSE": rmse_te,
            "Test_RMSE_%": rmse_te_pct,
            "Test_R2": r2_te,
            "Test_MAE": mae_te,
        })

    df_summary = pd.DataFrame(rows).sort_values("Test_RMSE_%", ascending=False)
    print(df_summary)
    return df_summary

# =========================================================
#  📊 Feature Importance Plotter
# =========================================================
def plot_feature_importance(model, X, title="Feature Importance", top_n=15):
    """
    Generic feature importance plotter for both tree-based and linear models.
    """
    plt.figure(figsize=(8, 6))

    # Case 1: Tree-based models (e.g., XGBoost)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feature_names = getattr(X, "columns", [f"f{i}" for i in range(len(importances))])
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}) \
                   .sort_values("Importance", ascending=False).head(top_n)
        sns.barplot(data=imp_df, x="Importance", y="Feature", palette="crest")
        plt.title(title)
        plt.tight_layout()

    # Case 2: Linear models (e.g., Ridge)
    elif hasattr(model, "coef_"):
        coef = np.abs(np.ravel(model.coef_))
        feature_names = getattr(X, "columns", [f"f{i}" for i in range(len(coef))])
        imp_df = pd.DataFrame({"Feature": feature_names, "Importance": coef}) \
                   .sort_values("Importance", ascending=False).head(top_n)
        sns.barplot(data=imp_df, x="Importance", y="Feature", palette="crest")
        plt.title(title + " (Linear Coefficients)")
        plt.tight_layout()

    else:
        print("⚠️ No feature importance or coefficients found for this model.")
        return

    plt.show()

# =========================================================
# 📈 Visualization – Predictions vs Actual & Residuals
# =========================================================
def plot_predictions_vs_actual(model, X, y, title="Predictions vs Actual", sample_size=None):
    """
    Visualize model performance: predicted vs actual values and residuals.
    ✅ Final version (display-only):
        - Removes MAE
        - Shows random sample nicely
        - Does not return anything (cleaner output in Jupyter)
    """
    from sklearn.metrics import mean_squared_error, r2_score
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    # ✅ Generate predictions
    y_pred = model.predict(X)

    # ✅ Random sample for clarity
    if sample_size and len(y) > sample_size:
        idx = np.random.choice(len(y), sample_size, replace=False)
        y_true_sample = np.array(y)[idx]
        y_pred_sample = np.array(y_pred)[idx]
    else:
        y_true_sample = np.array(y)
        y_pred_sample = np.array(y_pred)

    # ✅ Compute metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    # ✅ Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Predicted vs Actual ---
    ax1.scatter(y_true_sample, y_pred_sample, alpha=0.5, edgecolor='k', s=30)
    ax1.plot(
        [y_true_sample.min(), y_true_sample.max()],
        [y_true_sample.min(), y_true_sample.max()],
        'r--', lw=2, label='Perfect prediction'
    )
    ax1.set_xlabel('Actual Values', fontsize=12)
    ax1.set_ylabel('Predicted Values', fontsize=12)
    ax1.set_title(f'{title}\nRMSE={rmse:.2f} | R²={r2:.3f}', fontsize=13)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # --- Residuals ---
    residuals = y_true_sample - y_pred_sample
    ax2.scatter(y_pred_sample, residuals, alpha=0.5, edgecolor='k', s=30)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Values', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Residual Plot', fontsize=13)
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # ✅ Print metrics summary
    print(f"\n📊 {title} Metrics:")
    print(f"   RMSE: {rmse:.4f}")
    print(f"   R²  : {r2:.4f}")

    # ✅ Display random sample (nice formatted)
    results_df = pd.DataFrame({
        "Actual": np.array(y),
        "Predicted": np.array(y_pred),
        "Residual": np.array(y) - np.array(y_pred)
    })
    sample_to_display = results_df.sample(n=min(10, len(results_df)), random_state=42).reset_index(drop=True)
    print("\n📋 Random Sample of Predictions:")
    print(sample_to_display.round(2).to_string(index=False))

# =========================================================
# ⚡ Evaluate RMSE Against System Capacity
# =========================================================
def evaluate_rmse_against_capacity(model, X, y, capacity_kw):
    """
    Evaluate model RMSE as % of system capacity.
    Useful for understanding error relative to system size.
    """
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    y_pred = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    rmse_pct = 100 * rmse / capacity_kw
    mae_pct = 100 * mae / capacity_kw

    print(f"\n⚡ RMSE vs Capacity Analysis:")
    print(f"   RMSE = {rmse:.2f} W  ({rmse_pct:.2f}% of capacity)")
    print(f"   MAE  = {mae:.2f} W  ({mae_pct:.2f}% of capacity)")
    print(f"   R²   = {r2:.4f}")

    return {
        "RMSE_W": rmse,
        "MAE_W": mae,
        "R2": r2,
        "RMSE_%": rmse_pct,
        "MAE_%": mae_pct
    }
# =========================================================
# 📊 Plot Predictions: Train vs Tes
# =========================================================
def plot_predictions_train_vs_test(model, X_train, y_train, X_test, y_test, title="Train vs Test Comparison", sample_size=300):
    """
    📊 Compare model predictions on Train vs Test data.
    Draws both in one figure with metrics and random sampling for clarity.
    Works perfectly with XGBoost or any regression model.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_squared_error, r2_score
    import pandas as pd

    # ✅ Generate predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # ✅ Random sampling (to avoid overplotting)
    if sample_size:
        idx_train = np.random.choice(len(y_train), min(sample_size, len(y_train)), replace=False)
        idx_test = np.random.choice(len(y_test), min(sample_size, len(y_test)), replace=False)
    else:
        idx_train = np.arange(len(y_train))
        idx_test = np.arange(len(y_test))

    # ✅ Compute metrics
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    r2_train = r2_score(y_train, y_pred_train)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_test = r2_score(y_test, y_pred_test)

    # ✅ Combined scatter plot (Predicted vs Actual)
    plt.figure(figsize=(8, 8))
    plt.scatter(
        y_train.iloc[idx_train], y_pred_train[idx_train],
        alpha=0.4, color='royalblue',
        label=f"Train (RMSE={rmse_train:.1f}, R²={r2_train:.3f})"
    )
    plt.scatter(
        y_test.iloc[idx_test], y_pred_test[idx_test],
        alpha=0.5, color='tomato',
        label=f"Test (RMSE={rmse_test:.1f}, R²={r2_test:.3f})"
    )

    # ✅ Perfect prediction line
    all_vals = np.concatenate([y_train, y_test])
    lims = [all_vals.min(), all_vals.max()]
    plt.plot(lims, lims, 'k--', lw=2, label="Perfect Prediction")

    # ✅ Labels and layout
    plt.xlabel("Actual Power (W)")
    plt.ylabel("Predicted Power (W)")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ✅ Print numeric summary
    print(f"\n📊 {title} Metrics:")
    print(f" 🟢 TRAIN → RMSE={rmse_train:,.2f}, R²={r2_train:.3f}")
    print(f" 🔵 TEST  → RMSE={rmse_test:,.2f}, R²={r2_test:.3f}")

    # ✅ Display separate random samples (avoids length mismatch)
    print("\n📋 Random sample from TRAIN predictions:")
    train_df = pd.DataFrame({
        "Actual_Train": np.array(y_train),
        "Pred_Train": np.array(y_pred_train),
        "Residual_Train": np.array(y_train) - np.array(y_pred_train)
    }).sample(n=min(10, len(y_train)), random_state=42)
    print(train_df.round(2).to_string(index=False))

    print("\n📋 Random sample from TEST predictions:")
    test_df = pd.DataFrame({
        "Actual_Test": np.array(y_test),
        "Pred_Test": np.array(y_pred_test),
        "Residual_Test": np.array(y_test) - np.array(y_pred_test)
    }).sample(n=min(10, len(y_test)), random_state=42)
    print(test_df.round(2).to_string(index=False))

# =========================================================
# 💾 Save / Load
# =========================================================
def save_model(model, filename):
    joblib.dump(model, filename)
    print(f"💾 Model saved → {filename}")

def load_model(filename):
    return joblib.load(filename)

"""
helpers.py
-----------
Reusable helper functions for time-series regression and predictive modeling.

Designed following clean-code principles for clarity, modularity, and reusability.
This module can be reused across different regression or time-series projects.

Author: (Your Name)
"""

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
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor


sns.set_style("whitegrid")

# =========================================================
# 0️⃣ Column Name Standardization
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
# 1️⃣ Data Loading & Integration
# =========================================================
def load_and_merge_training_data(file_list):
    """
    Load multiple device files, standardize columns, align on timestamp, and
    compute the per-timestamp average across devices for each numeric feature.

    Returns:
        pd.DataFrame with columns: ['timestamp', <averaged numeric features...>]
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

            # Standardize active power name if needed
            if "active_power" not in df.columns and "Control-PPC - Active power (1m)" in df.columns:
                df = df.rename(columns={"Control-PPC - Active power (1m)": "active_power"})

            # Deduplicate exact timestamps by averaging numeric columns
            num_cols = df.select_dtypes(include="number").columns.tolist()
            agg = {c: "mean" for c in num_cols}
            df = df.groupby("timestamp", as_index=False).agg(agg)

            # Set timestamp as index for clean alignment later
            df = df.set_index("timestamp")
            dfs.append(df)
            print(f"✅ Loaded {f}: {df.shape}")
        except Exception as e:
            print(f"❌ Failed to load {f}: {e}")

    if len(dfs) == 0:
        raise ValueError("No training files could be loaded.")

    # Align on timestamp index (outer join to avoid losing points); then average over devices
    wide = pd.concat(dfs, axis=1, keys=[f"S{i+1}" for i in range(len(dfs))])
    # Average across the device level (level=0 is device key, level=1 is feature name)
    averaged = wide.groupby(level=1, axis=1).mean()

    # Bring timestamp back as a column
    averaged = averaged.reset_index()

    # Ensure target name is active_power
    if "active_power" not in averaged.columns and "Control-PPC - Active power (1m)" in averaged.columns:
        averaged = averaged.rename(columns={"Control-PPC - Active power (1m)": "active_power"})

    print(f"\n✅ Averaged {len(dfs)} files → Final shape: {averaged.shape}")
    return averaged


# =========================================================
# 2️⃣ Data Cleaning
# =========================================================
def clean_dataset(
    df: pd.DataFrame,
    target_col: str = "active_power",
    time_col: str = "timestamp",
    phys_limits: dict = None,
    clip_iqr: bool = True,
    interpolate_time: bool = True,
    mode: str = "train"   
) -> pd.DataFrame:
    """
    Generic time-series cleaning function for solar / sensor datasets.
    Supports both training and testing modes.

    Steps:
    1. Fix negatives in target column.
    2. Parse/sort time and add 'hour' column.
    3. Handle suspicious daytime zeros.
    4. Clip to physical limits if provided.
    5. Apply IQR capping (optional).
    6. Time-based interpolation (depends on mode).
    7. Forward/backward fill for remaining gaps.
    """

    df = df.copy()

    # 1️⃣ Fix negatives in target column
    if target_col not in df.columns:
        raise ValueError(f"Expected '{target_col}' column.")
    negatives_before = (df[target_col] < 0).sum()
    df[target_col] = np.where(df[target_col] < 0, 0, df[target_col])
    if negatives_before:
        print(f"⚙️ Replaced {negatives_before:,} negative {target_col} values with 0.")

    # 2️⃣ Parse timestamp and add hour
    if time_col not in df.columns:
        raise ValueError(f"Expected '{time_col}' column.")
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    df["hour"] = df[time_col].dt.hour
    daytime = df["hour"].between(6, 18)

    # 3️⃣ Detect suspicious zeros (daytime zeros with good irradiance)
    poa_like = [c for c in df.columns if "Irradiance" in c or "GHI" in c or "POA" in c]
    if poa_like:
        poa_col = poa_like[0]
        mask = (df[poa_col] > 100) & (df[target_col] <= 0) & daytime
        count_bad = mask.sum()
        if count_bad:
            print(f"⚙️ Marked {count_bad:,} suspicious daytime zeros in '{target_col}' as NaN.")
            df.loc[mask, target_col] = np.nan

    # 4️⃣ Apply physical limits if provided
    if phys_limits:
        for col, (low, high) in phys_limits.items():
            if col in df.columns:
                before = df[col].copy()
                df[col] = df[col].clip(low, high)
                clipped = (before != df[col]).sum()
                if clipped:
                    print(f"⚙️ Clipped {clipped:,} values in {col} outside [{low}, {high}]")

    # 5️⃣ IQR capping (light outlier suppression)
    if clip_iqr:
        num_cols = df.select_dtypes(include="number").columns.tolist()
        for col in num_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            if pd.isna(IQR) or IQR == 0:
                continue
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            df[col] = df[col].clip(lower, upper)

    # 6️⃣ Interpolation (depends on mode)
    df = df.set_index(time_col)
    num_cols = df.select_dtypes(include="number").columns.tolist()

    if interpolate_time and num_cols:
        if mode == "train":
            
            df[num_cols] = df[num_cols].interpolate(method="time", limit_direction="both")
            df[num_cols] = df[num_cols].ffill().bfill()
        else:
            
            df[num_cols] = df[num_cols].interpolate(method="time", limit=1, limit_direction="both")

    # 7️⃣ Fill for non-numeric columns
    non_num_cols = [c for c in df.columns if c not in num_cols]
    if non_num_cols:
        df[non_num_cols] = df[non_num_cols].ffill().bfill()

    df = df.reset_index()

    # 🧾 Summary
    print(f"✅ [{mode.upper()}] Cleaned successfully: {df.shape[0]:,} rows × {df.shape[1]:,} cols.")
    return df


# =========================================================
# 3️⃣ Smart EDA (Feature Redundancy Removal)
# =========================================================
def smart_feature_dedup(df, target="active_power", corr_thresh=0.99):
    """
    Remove highly correlated redundant features while keeping the most relevant to target.
    Returns the cleaned dataframe and a list of dropped columns.
    """
    df_num = df.select_dtypes(include="number").copy()
    features = [c for c in df_num.columns if c != target]
    corr = df_num[features].corr().abs()
    target_corr = df_num[features].corrwith(df_num[target]).abs()

    to_drop = set()
    dropped_pairs = []

    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            c1, c2 = features[i], features[j]
            if corr.loc[c1, c2] > corr_thresh:
                drop_col = c1 if target_corr[c1] < target_corr[c2] else c2
                to_drop.add(drop_col)
                dropped_pairs.append((c1, c2, corr.loc[c1, c2], drop_col))

    print(f"🧠 Removed {len(to_drop)} redundant features (corr > {corr_thresh})")
    if dropped_pairs:
        print("\n🔍 Dropped pairs (top 10):")
        for (a, b, corrv, dropc) in dropped_pairs[:10]:
            print(f" - {a} ↔ {b} | corr={corrv:.4f} → dropped {dropc}")

    return df.drop(columns=list(to_drop), errors="ignore"), list(to_drop)


# =========================================================
# 4️⃣ Feature Selection (PCC / MI / Spearman / Jaccard)
# =========================================================
def select_features(df, target="active_power", top_k=10):
    """
    Select top-K features using PCC and MI, compute Spearman rank correlation
    and Jaccard similarity between their top-K sets.
    """
    features = [c for c in df.select_dtypes(include="number").columns if c != target]

    # --- Compute PCC & MI ---
    pcc_scores = df[features].corrwith(df[target], method="pearson")
    mi_scores = mutual_info_regression(df[features], df[target], random_state=42)
    mi_series = pd.Series(mi_scores, index=features)

    # --- Spearman correlation between PCC and MI rankings ---
    ranks = pd.DataFrame({"PCC_rank": (-pcc_scores).rank(), "MI_rank": (-mi_series).rank()}).dropna()
    rho, _ = spearmanr(ranks["PCC_rank"], ranks["MI_rank"])
    print(f"📊 Spearman correlation (PCC vs MI ranks): {rho:.3f}")

    # --- Jaccard similarity ---
    top_pcc = pcc_scores.sort_values(ascending=False).head(top_k).index.tolist()
    top_mi = mi_series.sort_values(ascending=False).head(top_k).index.tolist()
    jaccard = len(set(top_pcc) & set(top_mi)) / len(set(top_pcc) | set(top_mi))
    print(f"🧩 Jaccard similarity (top-{top_k} sets): {jaccard:.3f}")

    # --- Visualization ---
    plt.figure(figsize=(6, 6))
    plt.scatter(ranks["PCC_rank"], ranks["MI_rank"], alpha=0.6, edgecolor="k")
    plt.plot([0, len(features)], [0, len(features)], "r--")
    plt.title(f"PCC vs MI Rank Comparison\nSpearman={rho:.3f}")
    plt.xlabel("PCC Feature Rank")
    plt.ylabel("MI Feature Rank")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return top_pcc, top_mi, rho, jaccard


# =========================================================
# 5️⃣ Cross-validation Evaluation
# =========================================================
def eval_cv(model, X, y, groups, label="Model"):
    """
    Perform GroupKFold CV and print fold metrics.
    """
    gkf = GroupKFold(n_splits=4)
    results = []

    for i, (tr, te) in enumerate(gkf.split(X, y, groups)):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]

        model.fit(Xtr, ytr)
        pred_tr, pred_te = model.predict(Xtr), model.predict(Xte)
        rmse_tr, rmse_te = sqrt(mean_squared_error(ytr, pred_tr)), sqrt(mean_squared_error(yte, pred_te))
        r2_tr, r2_te = r2_score(ytr, pred_tr), r2_score(yte, pred_te)

        print(f"[{label}] Fold {i}: Train RMSE={rmse_tr:,.2f}, Test RMSE={rmse_te:,.2f}, R²={r2_te:.3f}")
        results.append({"Fold": i, "RMSE_Train": rmse_tr, "RMSE_Test": rmse_te, "R2_Train": r2_tr, "R2_Test": r2_te})

    return pd.DataFrame(results)


# =========================================================
# 6️⃣ GridSearch
# =========================================================
def balanced_rmse(est, Xv, yv):
    """Custom RMSE scorer that rewards feature diversity if available."""
    y_pred = est.predict(Xv)
    rmse = np.sqrt(mean_squared_error(yv, y_pred))

    diversity = 1.0
    if hasattr(est, "feature_importances_"):
        fi = est.feature_importances_
        if np.sum(fi) > 0:
            fi = fi / np.sum(fi)
            diversity = (fi > np.mean(fi) * 0.3).sum() / len(fi)
            diversity = max(diversity, 0.3)

    adjusted_rmse = rmse / (1 + 0.2 * diversity)
    return -adjusted_rmse


def run_gridsearch_cv(model, param_grid, X, y, groups=None, name="Model",
                      cv_splits=4, scoring=balanced_rmse, verbose=1, n_jobs=-1):
    """Generic GridSearchCV runner supporting grouped or normal CV."""
    print(f"\n🚀 Running GridSearchCV for {name} ...")
    cv = GroupKFold(n_splits=cv_splits) if groups is not None else cv_splits

    grid = GridSearchCV(
        model,
        param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose
    )

    if groups is not None:
        grid.fit(X, y, groups=groups)
    else:
        grid.fit(X, y)

    print(f"✅ Best params for {name}: {grid.best_params_}")
    print(f"🏆 Best CV RMSE (balanced): {-grid.best_score_:.2f}")
    return grid


def run_multiple_gridsearches(models_with_params, X, y, groups=None):
    """
    Run GridSearchCV for multiple models.
    
    Parameters
    ----------
    models_with_params : dict
        Dictionary {model_name: (model_object, param_grid)}
    X, y : DataFrame
        Features and target.
    groups : array-like, optional
        Groups for GroupKFold (if applicable)
    """
    results = {}
    for name, (model, params) in models_with_params.items():
        results[name] = run_gridsearch_cv(model, params, X, y, groups, name)
    return results


# =========================================================
# 7️⃣ Model Evaluation & Visualization
# =========================================================
def evaluate_best_model(grid, X, y, groups, name, group_labels=None):
    """
    Evaluate the best model from GridSearchCV using GroupKFold.

    Parameters
    ----------
    grid : fitted GridSearchCV
        The grid search object with best_estimator_.
    X, y : pandas.DataFrame, pandas.Series
        Features and target data.
    groups : array-like
        Group labels for the CV splits (e.g., seasons, years).
    name : str
        Model name (for printing).
    group_labels : list, optional
        Custom labels for each group (e.g., ['Winter', 'Spring', ...]).
        If None, will just print Fold 1, Fold 2, ...
    """
    model_best = grid.best_estimator_
    gkf = GroupKFold(n_splits=4)
    rows = []

    print(f"\n################ EVALUATION for {name} #################")

    for i, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model_best.fit(X_train, y_train)
        y_pred_train = model_best.predict(X_train)
        y_pred_test = model_best.predict(X_test)

        rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        season_name = group_labels[i] if group_labels else f"Fold {i+1}"
        print(f"[{name}] {season_name}: Train RMSE={rmse_train:,.2f}, "
              f"Test RMSE={rmse_test:,.2f} | Train R²={r2_train:.4f}, Test R²={r2_test:.4f}")

        rows.append({
            "Fold": i+1,
            "Group": season_name,
            "RMSE_Train": rmse_train,
            "RMSE_Test": rmse_test,
            "R2_Train": r2_train,
            "R2_Test": r2_test
        })

    df_eval = pd.DataFrame(rows)
    print(f"\n[{name}] AVG → Train RMSE={df_eval['RMSE_Train'].mean():,.2f} | "
          f"Test RMSE={df_eval['RMSE_Test'].mean():,.2f} | "
          f"Train R²={df_eval['R2_Train'].mean():.4f} | "
          f"Test R²={df_eval['R2_Test'].mean():.4f}")

    return df_eval


def compare_models_results(evals):
    """Compare model metrics visually."""
    summary = pd.DataFrame({
        "Model": list(evals.keys()),
        "Train_RMSE": [e["RMSE_Train"].mean() for e in evals.values()],
        "Test_RMSE": [e["RMSE_Test"].mean() for e in evals.values()],
        "Train_R2": [e["R2_Train"].mean() for e in evals.values()],
        "Test_R2": [e["R2_Test"].mean() for e in evals.values()]
    })

    print("\n================= FINAL GRIDSEARCH SUMMARY =================")
    print(summary.to_string(index=False))

    plt.figure(figsize=(8, 5))
    sns.barplot(data=summary.melt(id_vars="Model", value_vars=["Train_RMSE", "Test_RMSE"]),
                x="Model", y="value", hue="variable", palette="crest")
    plt.title("RMSE Comparison after Hyperparameter Tuning")
    plt.tight_layout()
    plt.show()

    return summary


# =========================================================
# 8️⃣ Final Model Training & Feature Importance
# =========================================================
def train_xgb_model(X, y, params=None, verbose=True):
    """
    General XGBoost training function.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series or np.array): Target.
        params (dict, optional): Custom hyperparameters.
        verbose (bool): Whether to print confirmation message.

    Returns:
        XGBRegressor: Trained XGBoost model.
    """
    default_params = {
        'colsample_bylevel': 0.7, 
        'colsample_bytree': 0.6, 
        'learning_rate': 0.02,
        'max_depth': 5, 
        'min_child_weight': 5, 
        'n_estimators': 800, 
        'subsample': 0.8,
        'reg_lambda': 2.0, 
        'random_state': 42, 
        'n_jobs': -1
    }

    if params:
        default_params.update(params)

    model = XGBRegressor(**default_params)
    model.fit(X, y)

    if verbose:
        print("✅ XGBoost model trained successfully!")
    return model



def plot_feature_importance(model, X, title="XGBoost Feature Importance"):
    """Plot top 15 features for trained XGBoost model."""
    imp = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(15)

    plt.figure(figsize=(7, 6))
    sns.barplot(data=imp, x="Importance", y="Feature", palette="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(model, X, y, title="Predicted vs Actual (XGBoost)", sample_size=500):
    """
    Visualize model performance using both scatter plot (all data)
    and line plot (sampled data for detailed comparison).

    Args:
        model: Trained model (e.g., XGBRegressor)
        X (pd.DataFrame): Input features
        y (pd.Series or np.array): True target values
        title (str): Plot title
        sample_size (int): Number of samples for line plot
    """
    # ✅ Generate predictions
    y_pred = model.predict(X)

    # ✅ Compute metrics
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    mae = np.mean(np.abs(y - y_pred))

    # ==========================
    # 🔹 1. Scatter plot (ALL DATA)
    # ==========================
    plt.figure(figsize=(7, 7))
    plt.scatter(y, y_pred, alpha=0.4, edgecolor="k", s=20)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2, label="Ideal line")
    plt.xlabel("Actual Values", fontsize=12)
    plt.ylabel("Predicted Values", fontsize=12)
    plt.title(f"{title}\nRMSE={rmse:,.2f} | MAE={mae:,.2f} | R²={r2:.3f}", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ==========================
    # 🔹 2. Line plot (SAMPLED DATA)
    # ==========================
    if len(y) > sample_size:
        idx = np.linspace(0, len(y) - 1, sample_size, dtype=int)
        y_plot = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
        y_pred_plot = y_pred[idx]
        x_axis = X.index[idx] if hasattr(X, "index") else np.arange(len(y_plot))
    else:
        y_plot, y_pred_plot = y, y_pred
        x_axis = X.index if hasattr(X, "index") else np.arange(len(y))

    # If there’s a timestamp column, use it as X-axis
    if "timestamp" in X.columns:
        x_axis = pd.to_datetime(X["timestamp"].iloc[idx] if len(y) > sample_size else X["timestamp"])

    # Compute absolute error for visualization
    abs_error = np.abs(y_plot - y_pred_plot)

    plt.figure(figsize=(12, 6))
    plt.plot(x_axis, y_plot, label="Actual", linewidth=2)
    plt.plot(x_axis, y_pred_plot, label="Predicted", linewidth=2, alpha=0.8)
    plt.fill_between(range(len(y_plot)), y_plot, y_pred_plot, color="gray", alpha=0.15, label="Error gap")
    plt.title(f"Sampled Prediction vs Actual\nMean Abs Error={abs_error.mean():.2f}", fontsize=13)
    plt.xlabel("Time" if "timestamp" in X.columns else "Sample Index")
    plt.ylabel("Target Value")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ==========================
    # 🔹 3. Print numeric comparison (first 10)
    # ==========================
    preview = pd.DataFrame({
        "Actual": np.round(y_plot[:10].values, 3),
        "Predicted": np.round(y_pred_plot[:10], 3),
        "Difference": np.round(y_pred_plot[:10] - y_plot[:10].values, 3)
    })
    print("🔍 Sample numeric comparison (first 10 rows):")
    print(preview)

    # 🔹 Optionally return performance metrics
    return {"RMSE": rmse, "R2": r2}


# =========================================================
# 9️⃣ Model Persistence
# =========================================================
def save_model(model, filename):
    """Save model to disk."""
    joblib.dump(model, filename)
    print(f"💾 Model saved → {filename}")


def load_model(filename):
    """Load model from disk."""
    return joblib.load(filename)

# =========================================================
# 🔟 RMSE vs System Capacity Evaluation
# =========================================================
def evaluate_rmse_against_capacity(df, rmse_values, target_col="active_power", irradiance_col=None):
    """
    Evaluate RMSE values as a percentage of estimated system capacity.

    Steps:
      1. Estimate capacity from top quantiles of the target during sunny hours.
      2. Compute RMSE % of capacity overall and per season if provided.
    
    Args:
        df (pd.DataFrame): Dataset containing timestamp + active_power + irradiance.
        rmse_values (float or dict): RMSE (overall or per-season dict).
        target_col (str): Column name for target variable.
        irradiance_col (str): Column name for irradiance; if None, auto-detected.
    
    Returns:
        dict: Contains capacity estimation and RMSE % results.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    # Auto-detect irradiance column if not provided
    if irradiance_col is None:
        for c in df.columns:
            if "Irradiance" in c and "(W/m2)" in c:
                irradiance_col = c
                break

    if irradiance_col is None or irradiance_col not in df.columns:
        raise ValueError("Could not find a valid irradiance column in dataset.")

    # Filter to sunny daytime
    mask = df["hour"].between(8, 16) & (df[irradiance_col] > 200) & df[target_col].notna()
    subset = df.loc[mask, target_col]

    # Capacity estimation
    cap_p99 = subset.quantile(0.99)
    cap_p995 = subset.quantile(0.995)
    cap_top100 = subset.nlargest(100).median()
    cap_est = cap_p995

    print("🔎 Capacity estimation (Watts):")
    print(f" - P99.0  : {cap_p99:,.0f}")
    print(f" - P99.5  : {cap_p995:,.0f}")
    print(f" - Median of top-100: {cap_top100:,.0f}")
    print(f"\n✅ Chosen capacity estimate: {cap_est:,.0f} W")

    # Helper
    def pct(val): return 100 * val / cap_est if cap_est > 0 else np.nan

    # Compute RMSE % of capacity
    print("\n📏 RMSE as % of capacity:")
    if isinstance(rmse_values, dict):
        for s, v in rmse_values.items():
            print(f" - {s.capitalize():7s}: {pct(v):.2f}%")
        overall = np.mean(list(rmse_values.values()))
        print(f" - Overall: {pct(overall):.2f}%")
    else:
        print(f" - Overall RMSE ({rmse_values:,.2f}) → {pct(rmse_values):.2f}%")

    return {
        "capacity_est": cap_est,
        "rmse_percent": pct(rmse_values if isinstance(rmse_values, (int, float)) else np.mean(list(rmse_values.values())))
    }


# =========================================================
# 1️⃣1️⃣ Preprocess New Data for Deployment
# =========================================================
def preprocess_new_data(df_raw):
    """Apply same cleaning pipeline to unseen data."""
    df_clean = clean_dataset(df_raw)
    df_clean = smart_feature_dedup(df_clean)
    return df_clean

"""
Healthcare Data Analytics Project — Diabetes Risk Analysis
Full Analysis: Cleaning + EDA + Visualizations + ML Model
Author: Your Name | MSc Biotechnology | Aspiring Data Analyst

Domain note: As a Biotechnology MSc graduate, I knew that
glucose, blood pressure, and BMI values of 0 are biologically
impossible in living patients — they are recording errors.
This informed a clinically-appropriate imputation strategy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
import os
import warnings
warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.size": 11,
                     "axes.titlesize": 13, "axes.titleweight": "bold"})
os.makedirs("outputs/figures", exist_ok=True)

RAW = "data/diabetes.csv"


# ══════════════════════════════════════════
#  STEP 1: CLEAN — CLINICALLY-INFORMED
# ══════════════════════════════════════════
def clean_healthcare(filepath):
    print("=" * 55)
    print("  HEALTHCARE DATA CLEANING")
    print("=" * 55)

    df = pd.read_csv(filepath)
    print(f"\n[1] Raw shape    : {df.shape}")
    print(f"    Outcome dist : {df['Outcome'].value_counts().to_dict()}")
    print(f"    Diabetes rate: {df['Outcome'].mean():.1%}")

    print(f"\n[2] Checking for biologically-impossible zero values:")
    zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_cols:
        n_zeros = (df[col] == 0).sum()
        if n_zeros > 0:
            print(f"    {col:<20}: {n_zeros:>3} zeros ({n_zeros/len(df)*100:.1f}%) → replacing with NaN")

    # ── Replace impossible zeros with NaN
    df[zero_cols] = df[zero_cols].replace(0, np.nan)

    # ── Domain-aware imputation: median per Outcome group
    print(f"\n[3] Imputing with outcome-group medians (clinically informed):")
    for col in zero_cols:
        medians = df.groupby("Outcome")[col].median()
        df[col] = df.groupby("Outcome")[col].transform(lambda x: x.fillna(x.median()))
        print(f"    {col:<20}: diabetic median={medians.get(1,0):.1f}, "
              f"healthy median={medians.get(0,0):.1f}")

    # ── Feature engineering
    df["Age_Group"] = pd.cut(df["Age"],
        bins=[20, 30, 40, 50, 60, 100],
        labels=["21-30", "31-40", "41-50", "51-60", "60+"])

    df["BMI_Category"] = pd.cut(df["BMI"],
        bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"])

    df["Glucose_Category"] = pd.cut(df["Glucose"],
        bins=[0, 99, 125, 200],
        labels=["Normal", "Pre-diabetic", "Diabetic-range"])

    print(f"\n[4] Final cleaned shape: {df.shape}")
    print(f"    New features: Age_Group, BMI_Category, Glucose_Category")

    df.to_csv("data/diabetes_clean.csv", index=False)
    print(f"\n[✓] Saved: data/diabetes_clean.csv")
    return df


# ══════════════════════════════════════════
#  STEP 2: EDA SUMMARY
# ══════════════════════════════════════════
def eda_summary(df):
    print("\n" + "=" * 55)
    print("  EDA SUMMARY")
    print("=" * 55)

    print("\n--- Clinical stats by Outcome ---")
    features = ["Glucose","BloodPressure","BMI","Age","Insulin",
                "DiabetesPedigreeFunction","Pregnancies"]
    stats = df.groupby("Outcome")[features].mean().round(2).T
    stats.columns = ["No Diabetes (0)", "Diabetes (1)"]
    stats["Difference"] = (stats["Diabetes (1)"] - stats["No Diabetes (0)"]).round(2)
    print(stats.to_string())

    print("\n--- Diabetes rate by Age Group ---")
    age_rate = df.groupby("Age_Group", observed=True)["Outcome"].mean().round(3) * 100
    print(age_rate.to_string())

    print("\n--- Diabetes rate by BMI Category ---")
    bmi_rate = df.groupby("BMI_Category", observed=True)["Outcome"].mean().round(3) * 100
    print(bmi_rate.to_string())

    print("\n--- Correlation with Outcome ---")
    num_cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                "Insulin","BMI","DiabetesPedigreeFunction","Age"]
    corr = df[num_cols + ["Outcome"]].corr()["Outcome"].drop("Outcome").sort_values(ascending=False)
    print(corr.round(3).to_string())


# ══════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════
COLORS = {0: "#5DCAA5", 1: "#E24B4A"}
LABELS = {0: "No Diabetes", 1: "Diabetes"}


def chart_outcome_distribution(df):
    counts = df["Outcome"].value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar chart
    bars = axes[0].bar([LABELS[0], LABELS[1]], counts.values,
                       color=[COLORS[0], COLORS[1]],
                       edgecolor="white", linewidth=0.5, width=0.5)
    for bar, (i, val) in zip(bars, counts.items()):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 5,
                     f"{val}\n({val/len(df)*100:.1f}%)",
                     ha="center", fontsize=11, fontweight="bold")
    axes[0].set_title("Diabetes Prevalence in Dataset")
    axes[0].set_ylabel("Number of Patients")
    axes[0].set_ylim(0, counts.max() * 1.2)
    axes[0].grid(axis="y", alpha=0.3); axes[0].set_axisbelow(True)

    # Glucose distribution by outcome
    for outcome in [0, 1]:
        subset = df[df["Outcome"] == outcome]["Glucose"]
        axes[1].hist(subset, bins=30, alpha=0.65,
                     color=COLORS[outcome], label=LABELS[outcome],
                     edgecolor="white", linewidth=0.3)
    axes[1].set_title("Glucose Distribution by Diabetes Status")
    axes[1].set_xlabel("Glucose Level (mg/dL)")
    axes[1].set_ylabel("Count"); axes[1].legend()
    axes[1].grid(alpha=0.3); axes[1].set_axisbelow(True)

    plt.suptitle("Diabetes Dataset Overview", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("outputs/figures/01_outcome_overview.png", bbox_inches="tight")
    plt.close()
    print("  [✓] Chart 1: 01_outcome_overview.png")


def chart_clinical_boxplots(df):
    features = ["Glucose", "BMI", "Age", "BloodPressure",
                "Insulin", "DiabetesPedigreeFunction"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    for ax, feat in zip(axes.flatten(), features):
        data_plot = [df[df["Outcome"] == 0][feat].dropna(),
                     df[df["Outcome"] == 1][feat].dropna()]
        bp = ax.boxplot(data_plot, patch_artist=True,
                        medianprops={"color":"white","linewidth":2},
                        flierprops={"marker":"o","markersize":3,"alpha":0.4})
        for patch, color in zip(bp["boxes"], [COLORS[0], COLORS[1]]):
            patch.set_facecolor(color); patch.set_alpha(0.8)
        ax.set_xticklabels([LABELS[0], LABELS[1]])
        ax.set_title(feat, fontsize=11)
        ax.grid(axis="y", alpha=0.3); ax.set_axisbelow(True)

    plt.suptitle("Clinical Feature Distributions by Diabetes Status",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("outputs/figures/02_clinical_boxplots.png", bbox_inches="tight")
    plt.close()
    print("  [✓] Chart 2: 02_clinical_boxplots.png")


def chart_correlation_heatmap(df):
    num_cols = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                "Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
    labels   = ["Pregnancies","Glucose","Blood\nPressure","Skin\nThickness",
                "Insulin","BMI","Pedigree\nFunction","Age","Outcome"]
    corr = df[num_cols].corr()
    corr.index = labels; corr.columns = labels

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, square=True, linewidths=0.6,
                linecolor="white", ax=ax,
                cbar_kws={"shrink": 0.82},
                annot_kws={"size": 9})
    ax.set_title("Feature Correlation Heatmap — Diabetes Dataset", pad=15)
    plt.xticks(rotation=30, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig("outputs/figures/03_correlation_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  [✓] Chart 3: 03_correlation_heatmap.png")


def chart_age_bmi_analysis(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Diabetes rate by age group
    age_rate = (df.groupby("Age_Group", observed=True)["Outcome"]
                  .mean().reset_index())
    age_rate["rate_pct"] = age_rate["Outcome"] * 100
    colors_age = ["#AFA9EC","#7F77DD","#534AB7","#3C3489","#26215C"]
    bars = axes[0].bar(age_rate["Age_Group"].astype(str),
                       age_rate["rate_pct"], color=colors_age,
                       edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, age_rate["rate_pct"]):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5, f"{val:.1f}%",
                     ha="center", fontsize=10, fontweight="bold")
    axes[0].set_title("Diabetes Rate by Age Group")
    axes[0].set_xlabel("Age Group"); axes[0].set_ylabel("Diabetes Rate (%)")
    axes[0].set_ylim(0, age_rate["rate_pct"].max() * 1.2)
    axes[0].grid(axis="y", alpha=0.3); axes[0].set_axisbelow(True)

    # Diabetes rate by BMI category
    bmi_rate = (df.groupby("BMI_Category", observed=True)["Outcome"]
                  .mean().reset_index())
    bmi_rate["rate_pct"] = bmi_rate["Outcome"] * 100
    colors_bmi = ["#9FE1CB","#5DCAA5","#1D9E75","#0F6E56"]
    bars2 = axes[1].bar(bmi_rate["BMI_Category"].astype(str),
                        bmi_rate["rate_pct"], color=colors_bmi,
                        edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars2, bmi_rate["rate_pct"]):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.5, f"{val:.1f}%",
                     ha="center", fontsize=10, fontweight="bold")
    axes[1].set_title("Diabetes Rate by BMI Category")
    axes[1].set_xlabel("BMI Category"); axes[1].set_ylabel("Diabetes Rate (%)")
    axes[1].set_ylim(0, bmi_rate["rate_pct"].max() * 1.2)
    axes[1].grid(axis="y", alpha=0.3); axes[1].set_axisbelow(True)

    plt.suptitle("Risk Factor Analysis — Age & BMI", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("outputs/figures/04_age_bmi_analysis.png", bbox_inches="tight")
    plt.close()
    print("  [✓] Chart 4: 04_age_bmi_analysis.png")


def chart_glucose_bmi_scatter(df):
    fig, ax = plt.subplots(figsize=(10, 7))
    for outcome in [0, 1]:
        subset = df[df["Outcome"] == outcome]
        ax.scatter(subset["Glucose"], subset["BMI"],
                   c=COLORS[outcome], label=LABELS[outcome],
                   alpha=0.55, s=35, edgecolors="none")

    ax.set_xlabel("Glucose Level (mg/dL)", labelpad=10)
    ax.set_ylabel("BMI", labelpad=10)
    ax.set_title("Glucose vs BMI — Colored by Diabetes Status", pad=15)
    ax.legend(markerscale=1.5, fontsize=10)
    ax.axvline(126, color="#854F0B", linewidth=1.2, linestyle="--",
               alpha=0.7, label="Diabetes threshold (126 mg/dL)")
    ax.axhline(30, color="#A32D2D", linewidth=1.2, linestyle=":",
               alpha=0.7)
    ax.text(127, ax.get_ylim()[1]*0.95, "Glucose threshold\n(126 mg/dL)",
            fontsize=8, color="#854F0B")
    ax.text(ax.get_xlim()[0]+1, 30.5, "Obese threshold (BMI=30)",
            fontsize=8, color="#A32D2D")
    ax.grid(alpha=0.3); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("outputs/figures/05_glucose_bmi_scatter.png", bbox_inches="tight")
    plt.close()
    print("  [✓] Chart 5: 05_glucose_bmi_scatter.png")


# ══════════════════════════════════════════
#  STEP 3: LOGISTIC REGRESSION MODEL
# ══════════════════════════════════════════
def build_model(df):
    print("\n" + "=" * 55)
    print("  LOGISTIC REGRESSION MODEL")
    print("=" * 55)

    features = ["Glucose","BMI","Age","Pregnancies",
                "BloodPressure","Insulin","DiabetesPedigreeFunction"]
    X = df[features].copy()
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_sc, y_train)

    y_pred      = model.predict(X_test_sc)
    y_pred_prob = model.predict_proba(X_test_sc)[:, 1]
    auc         = roc_auc_score(y_test, y_pred_prob)

    print(f"\n[Results]")
    print(f"  Training samples : {len(X_train)}")
    print(f"  Test samples     : {len(X_test)}")
    print(f"  ROC-AUC Score    : {auc:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Diabetes','Diabetes'])}")

    # Feature importance
    coef_df = pd.DataFrame({
        "Feature": features,
        "Coefficient": model.coef_[0]
    }).sort_values("Coefficient", key=abs, ascending=False)
    print("Feature importance (absolute coefficient):")
    print(coef_df.to_string(index=False))

    # ── Chart 6: Confusion matrix + ROC curve
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                ax=axes[0], linewidths=0.5,
                xticklabels=["No Diabetes","Diabetes"],
                yticklabels=["No Diabetes","Diabetes"],
                annot_kws={"size": 14, "weight": "bold"})
    axes[0].set_ylabel("Actual"); axes[0].set_xlabel("Predicted")
    axes[0].set_title(f"Confusion Matrix\n(Accuracy: {(y_pred==y_test).mean():.1%})")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    axes[1].plot(fpr, tpr, color="#7F77DD", linewidth=2.5,
                 label=f"ROC Curve (AUC = {auc:.3f})")
    axes[1].plot([0,1], [0,1], "k--", linewidth=1, alpha=0.5, label="Random (AUC = 0.5)")
    axes[1].fill_between(fpr, tpr, alpha=0.1, color="#7F77DD")
    axes[1].set_xlabel("False Positive Rate"); axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve — Logistic Regression")
    axes[1].legend(loc="lower right", fontsize=10)
    axes[1].grid(alpha=0.3); axes[1].set_axisbelow(True)

    plt.suptitle("Model Evaluation — Diabetes Prediction",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/figures/06_model_evaluation.png", bbox_inches="tight")
    plt.close()
    print("\n  [✓] Chart 6: 06_model_evaluation.png")

    return model, scaler


# ══════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════
if __name__ == "__main__":
    print("\n=== HEALTHCARE DATA ANALYTICS ===\n")
    df = clean_healthcare(RAW)
    eda_summary(df)

    print("\nGenerating charts...")
    chart_outcome_distribution(df)
    chart_clinical_boxplots(df)
    chart_correlation_heatmap(df)
    chart_age_bmi_analysis(df)
    chart_glucose_bmi_scatter(df)

    model, scaler = build_model(df)

    print("\n[✓] All 6 charts saved to outputs/figures/")
    print("[✓] Healthcare analysis complete!\n")

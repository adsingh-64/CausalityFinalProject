# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, log_loss
import sklearn
import os
from matplotlib.pyplot import hist
import matplotlib.pyplot as plt

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# %%
df = pd.read_csv("1989.csv")

# %%
df.head()

# %%
CONFIG = {
    "columns": {
        "R0000100": "ID",
        "R0618300": "AFQT",
        "R2925010": "hourly_pay",
        "R2928400": "job_educ_req",
        "T1215600": "educ"
    }, 
    "inplace": True
}

df.rename(**CONFIG)

# %%
df = df[(df['AFQT'] >= 0) & (df['hourly_pay'] >= 0) & (df['educ'] >= 0) & (df['job_educ_req'] >= 0)].reset_index(drop=True)

# %%
overqualified_list = []
for i, level in enumerate(df["educ"]):
    job_req = df["job_educ_req"].iloc[i]
    match level:
        case 0:
            overqualified = 0
        case 1:
            if job_req <= 2:
                overqualified = 1
            else:
                overqualified = 0
        case 2:
            if job_req <= 3:
                overqualified = 1
            else:
                overqualified = 0
        case 3:
            if job_req <= 4:
                overqualified = 1
            else:
                overqualified = 0
        case 4:
            if job_req <= 4:
                overqualified = 1
            else:
                overqualified = 0
        case _:
            if job_req <= 5:
                overqualified = 1
            else:
                overqualified = 0
    overqualified_list.append(overqualified)
df['overqualified'] = pd.Series(overqualified_list)
df.drop("job_educ_req", axis=1, inplace=True)
df["educ"] = (df["educ"] < 3).astype(int)

# %%
for _ in range(2): # there are two buggy data points with over $1mil/hour
    max_index = df["hourly_pay"].idxmax()
    df = df.drop(max_index).reset_index(drop=True)

# %%
# Check overlap condition P(A = 1|M=m, C=c) < 1 for every m, c
# A = educ, M = overqualified, C = AFQT
def check_overlap_with_confounders(df):
    """Check overlap condition with confounders"""
    # Discretize AFQT into quartiles for overlap check
    df['AFQT_quartile'] = pd.qcut(df['AFQT'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    
    overlap_stats = []
    for m in [0, 1]:
        for c in ['Q1', 'Q2', 'Q3', 'Q4']:
            subset = df[(df['overqualified'] == m) & (df['AFQT_quartile'] == c)]
            if len(subset) > 0:
                prob_a1 = subset['educ'].mean()
                overlap_stats.append({
                    'M': m,
                    'C': c,
                    'prob_A1': prob_a1,
                    'n': len(subset)
                })
    
    overlap_df = pd.DataFrame(overlap_stats)
    print("Overlap check (P(A=1|M,C) by group):")
    print(overlap_df)

    # Plot overlap visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bar plot
    x_labels = [f"M={row['M']}, C={row['C']}" for _, row in overlap_df.iterrows()]
    bars = ax.bar(x_labels, overlap_df['prob_A1'], 
                  color=['lightblue' if m == 0 else 'lightcoral' for m in overlap_df['M']])
    
    # Add horizontal lines for reference
    ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Problematic overlap (< 0.05)')
    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='Problematic overlap (> 0.95)')
    
    # Customize plot
    ax.set_ylabel('P(A=a\'|M,C)')
    ax.set_xlabel('Groups (M=Overqualified, C=AFQT Quartile)')
    ax.set_title('Overlap Check: Treatment Probability by Mediator and Confounder')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Check if any probability is too close to 0 or 1
    problematic = overlap_df[(overlap_df['prob_A1'] < 0.05) | (overlap_df['prob_A1'] > 0.95)]
    if len(problematic) > 0:
        print("\nWarning: Some groups have problematic overlap:")
        print(problematic)
    else:
        print("\nOverlap condition satisfied for all groups.")
    
    return overlap_df

overlap_check = check_overlap_with_confounders(df)

# %%
def att_aiptw_with_confounders(Q0, Q1, g, p_baseline, A, Y):
    """
    Double ML estimator for the ATT with baseline confounders
    This implements the ATT-AIPTW estimator with baseline propensity adjustment
    From equation in CLAUDE.md section on "Adjusting for Baseline Confounding"
    
    Args:
    Q0, Q1: outcome predictions under control/treatment
    g: propensity scores P(A=a'|M,C) 
    p_baseline: baseline propensity scores P(A=a'|C)
    A: treatment indicator
    Y: outcome
    
    Return: aiptw of ATT and its standard error
    """
    
    # number of observations
    n = Y.shape[0]
    
    # ATT-AIPTW estimator with baseline confounders 
    first_term = (A / p_baseline) * (Y - Q0)
    second_term = ((1 - A) * g) / (p_baseline * (1 - g)) * (Y - Q0)
    tau_hat = (first_term - second_term).mean()
    
    # Influence function for standard error
    phi = (A / p_baseline) * (Y - Q0) - ((1 - A) * g) / (p_baseline * (1 - g)) * (Y - Q0) - (A * tau_hat) / p_baseline
    std_hat = np.std(phi) / np.sqrt(n)
    
    return tau_hat, std_hat


# %%
def make_Q_model():
    """A function that returns a general ML q model for later use in k-folding"""
    # return RandomForestRegressor(random_state=RANDOM_SEED, n_estimators=500, max_depth=None)

    # Gradient Boosting Regressor
    # return GradientBoostingRegressor(random_state=RANDOM_SEED, n_estimators=100, max_depth=3, learning_rate=0.1)

    # Linear Regression (simple baseline)
    return LinearRegression()


# %%
def make_g_model():
    """A function that returns a g model for computing propensity scores"""
    # return RandomForestClassifier(n_estimators=100, max_depth=5)

    # return GradientBoostingClassifier(random_state=RANDOM_SEED, n_estimators=100, max_depth=3, learning_rate=0.1)

    return LogisticRegression(random_state=RANDOM_SEED, solver='liblinear', C=1.0)


# %%
def treatment_k_fold_fit_and_predict_with_confounders(
    make_model, M: pd.DataFrame, C: pd.DataFrame, A: np.array, n_splits: int
):
    """
    Implements K fold cross-fitting for the model predicting P(A=a'|M,C).
    
    Args:
    make_model: function that returns sklearn model
    M: dataframe of mediator variables
    C: dataframe of confounding variables  
    A: array of treatments
    n_splits: number of splits to use
    """
    
    # Combine mediator and confounders
    X = pd.concat([M, C], axis=1)
    
    predictions = np.full_like(A, np.nan, dtype=float)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    for train_index, test_index in kf.split(X, A):
        X_train = X.loc[train_index]
        A_train = A.loc[train_index]
        g = make_model()
        g.fit(X_train, A_train)

        # get predictions for split
        predictions[test_index] = g.predict_proba(X.loc[test_index])[:, 1]

    # sanity check that overlap holds
    assert np.isnan(predictions).sum() == 0
    return predictions


# %%
def baseline_propensity_k_fold_fit_and_predict(
    make_model, C: pd.DataFrame, A: np.array, n_splits: int
):
    """
    Implements K fold cross-fitting for the baseline propensity model P(A=a'|C).
    
    Args:
    make_model: function that returns sklearn model
    C: dataframe of confounding variables
    A: array of treatments  
    n_splits: number of splits to use
    """
    
    predictions = np.full_like(A, np.nan, dtype=float)
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    for train_index, test_index in kf.split(C, A):
        C_train = C.loc[train_index]
        A_train = A.loc[train_index]
        p_baseline = make_model()
        p_baseline.fit(C_train, A_train)

        # get predictions for split
        predictions[test_index] = p_baseline.predict_proba(C.loc[test_index])[:, 1]

    # sanity check that overlap holds
    assert np.isnan(predictions).sum() == 0
    return predictions


# %%
def outcome_k_fold_fit_and_predict_with_confounders(
    make_model,
    M: pd.DataFrame,
    C: pd.DataFrame, 
    y: np.array,
    A: np.array,
    n_splits: int,
    output_type: str,
):
    """
    Implements K fold cross-fitting for the model predicting E[Y|A,M,C].
    
    Args:
    make_model: function that returns sklearn model
    M: dataframe of mediator variables
    C: dataframe of confounding variables
    y: array of outcomes
    A: array of treatments
    n_splits: number of splits to use
    output_type: type of outcome, "binary" or "continuous"
    """
    
    # Combine mediator and confounders
    X = pd.concat([M, C], axis=1)
    
    predictions0 = np.full_like(A, np.nan, dtype=float)
    predictions1 = np.full_like(y, np.nan, dtype=float)
    if output_type == "binary":
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    elif output_type == "continuous":
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

    # include the treatment as input feature
    X_w_treatment = X.copy()
    X_w_treatment["A"] = A

    # for predicting effect under treatment / control status for each data point
    X0 = X_w_treatment.copy()
    X0["A"] = 0
    X1 = X_w_treatment.copy()
    X1["A"] = 1

    for train_index, test_index in kf.split(X_w_treatment, y):
        X_train = X_w_treatment.loc[train_index]
        y_train = y.loc[train_index]
        q = make_model()
        q.fit(X_train, y_train)

        if output_type == "binary":
            predictions0[test_index] = q.predict_proba(X0.loc[test_index])[:, 1]
            predictions1[test_index] = q.predict_proba(X1.loc[test_index])[:, 1]
        elif output_type == "continuous":
            predictions0[test_index] = q.predict(X0.loc[test_index])
            predictions1[test_index] = q.predict(X1.loc[test_index])

    assert np.isnan(predictions0).sum() == 0
    assert np.isnan(predictions1).sum() == 0
    return predictions0, predictions1

# %%
# Define variables with confounders
mediator = df[["overqualified"]]
confounder = df[["AFQT"]]  # AFQT percentile scores as confounder C
outcome = df["hourly_pay"]
treatment = df["educ"]

# %%
# Fit propensity model P(A=a'|M,C)
g = treatment_k_fold_fit_and_predict_with_confounders(
    make_g_model, M=mediator, C=confounder, A=treatment, n_splits=5
)

# %%
# Fit baseline propensity model P(A=a'|C)
p_baseline = baseline_propensity_k_fold_fit_and_predict(
    make_g_model, C=confounder, A=treatment, n_splits=5
)

# %%
hist(g, density=True, alpha=0.7, label='g(M,C)')
hist(p_baseline, density=True, alpha=0.7, label='p(C)')
plt.xlabel('Propensity Score')
plt.ylabel('Density')
plt.title('Propensity Score Distributions')
plt.legend()
plt.show()

# %%
# get conditional outcomes E[Y|A,M,C]
Q0_ml, Q1_ml = outcome_k_fold_fit_and_predict_with_confounders(
    make_Q_model,
    M=mediator,
    C=confounder,
    y=outcome,
    A=treatment,
    n_splits=5,
    output_type="continuous",
)

# %%
data_and_nuisance_estimates_ml = pd.DataFrame(
    {"Q0": Q0_ml, "Q1": Q1_ml, "g": g, "p_baseline": p_baseline, "A": treatment, "Y": outcome}
)
data_and_nuisance_estimates_ml

# %%
tau_hat, std_hat = att_aiptw_with_confounders(
    Q0=data_and_nuisance_estimates_ml["Q0"],
    Q1=data_and_nuisance_estimates_ml["Q1"], 
    g=data_and_nuisance_estimates_ml["g"],
    p_baseline=data_and_nuisance_estimates_ml["p_baseline"],
    A=data_and_nuisance_estimates_ml["A"],
    Y=data_and_nuisance_estimates_ml["Y"]
)

print(f"The confounder-adjusted NDE is {-tau_hat:.4f} with a std = {std_hat:.4f}")
print(f"Confidence interval: {-tau_hat:.4f} +/- {1.96*std_hat:.4f}")
print(f"95% CI: [{-tau_hat - 1.96*std_hat:.4f}, {-tau_hat + 1.96*std_hat:.4f}]")


# %%

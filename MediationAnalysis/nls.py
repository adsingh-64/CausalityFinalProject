# %%
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline 
import sklearn
import os
from matplotlib.pyplot import hist
import matplotlib.pyplot as plt

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# %%
df = pd.read_csv("1989.csv")

#%%
len(df)

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
# Check overlap condition P(A = 1|M=m) < 1 for every m
# A = educ, M = overqualified
prob_a1_given_m0 = df[df['overqualified'] == 0]['educ'].mean()
prob_a1_given_m1 = df[df['overqualified'] == 1]['educ'].mean()

plt.figure(figsize=(8, 6))
plt.bar(['M=0 (Not Overqualified)', 'M=1 (Overqualified)'], 
        [prob_a1_given_m0, prob_a1_given_m1], 
        color=['skyblue', 'lightcoral'])
plt.ylabel('P(A = 1 | M = m)')
plt.title('Probability of Not Having A Bachelor\'s Degree (A=1) Given Overqualification Status (M)')
plt.ylim(0, 1)
for i, v in enumerate([prob_a1_given_m0, prob_a1_given_m1]):
    plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.show()

# %%
def att_aiptw(Q0, Q1, g, A, Y, prob_t=None):
    """
    Double ML estimator for the ATT
    This uses the ATT specific scores, see equation 3.9 of https://www.econstor.eu/bitstream/10419/149795/1/869216953.pdf
    Return: aiptw of ATE and its standard error
    """

    # number of observations
    n = Y.shape[0]

    # estimate marginal probability of treatment
    if prob_t is None:
        prob_t = A.mean()

    # att aiptw
    tau_hat = (A * (Y - Q0) - (1 - A) * (g / (1 - g)) * (Y - Q0)).mean() / prob_t

    # influence curve and standard error of aiptw
    phi = (A * (Y - Q0) - (1 - A) * (g / (1 - g)) * (Y - Q0) - tau_hat * A) / prob_t
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
def treatment_k_fold_fit_and_predict(
    make_model, X: pd.DataFrame, A: np.array, n_splits: int
):
    """
    Implements K fold cross-fitting for the model predicting the treatment A.
    That is,
    1. Split data into K folds
    2. For each fold j, the model is fit on the other K-1 folds
    3. The fitted model is used to make predictions for each data point in fold j
    Returns an array containing the predictions

    Args:
    model: function that returns sklearn model (which implements fit and predict_prob)
    X: dataframe of variables to adjust for
    A: array of treatments
    n_splits: number of splits to use
    """

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
def outcome_k_fold_fit_and_predict(
    make_model,
    X: pd.DataFrame,
    y: np.array,
    A: np.array,
    n_splits: int,
    output_type: str,
):
    """
    Implements K fold cross-fitting for the model predicting the outcome Y.
    That is,
    1. Split data into K folds
    2. For each fold j, the model is fit on the other K-1 folds
    3. The fitted model is used to make predictions for each data point in fold j
    Returns two arrays containing the predictions for all units untreated, all units treated

    Args:
    model: function that returns sklearn model (that implements fit and either predict_prob or predict)
    X: dataframe of variables to adjust for
    y: array of outcomes
    A: array of treatments
    n_splits: number of splits to use
    output_type: type of outcome, "binary" or "continuous"
    """

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
confounders = df[["overqualified"]]
outcome = df["hourly_pay"]
treatment = df["educ"]

# %%
g = treatment_k_fold_fit_and_predict(
    make_g_model, X=confounders, A=treatment, n_splits=5
)

# %%
hist(g, density=True)

# %%
# get conditional outcomes
Q0_ml, Q1_ml = outcome_k_fold_fit_and_predict(
    make_Q_model,
    X=confounders,
    y=outcome,
    A=treatment,
    n_splits=5,
    output_type="continuous",
)

# %%
data_and_nuisance_estimates_ml = pd.DataFrame(
    {"Q0": Q0_ml, "Q1": Q1_ml, "g": g, "A": treatment, "Y": outcome}
)
data_and_nuisance_estimates_ml

# %%
tau_hat, std_hat = att_aiptw(**data_and_nuisance_estimates_ml)
print(
    f"Outcome model: GradientBoostingRegressor | propensity score model: GradientBoostingClassifier The NDE is {-tau_hat} with a std = {std_hat}"
)
print(f"Confidence interval: {-tau_hat} +/- {1.96*std_hat}")

# %%

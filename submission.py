
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss

from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# SEARCH FILES

print("Searching Kaggle input folders...\n")

all_files = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        full_path = os.path.join(dirname, filename)

        print(full_path)

        all_files.append(full_path)

# DETECT FILES

train_path = None
schedule_path = None
sample_path = None
public_lb_path = None

for file in all_files:

    lower = os.path.basename(file).lower()

    if lower == "train_ipl.csv":
        train_path = file

    elif lower == "schedule.csv":
        schedule_path = file

    elif lower == "sample_submission.csv":
        sample_path = file

    elif lower == "public_lb_matches.csv":
        public_lb_path = file

print("\nDetected Files:\n")

print("Train:", train_path)
print("Schedule:", schedule_path)
print("Sample:", sample_path)
print("Public LB:", public_lb_path)

# LOAD DATA

train = pd.read_csv(
    train_path,
    low_memory=False
)

schedule = pd.read_csv(schedule_path)

sample = pd.read_csv(sample_path)

public_lb = pd.read_csv(public_lb_path)

print("\nLoaded Successfully\n")

print("Train Shape:", train.shape)
print("Schedule Shape:", schedule.shape)
print("Public LB Shape:", public_lb.shape)
print("Sample Shape:", sample.shape)

# STANDARDIZE TEAM NAMES

team_map = {

    "Royal Challengers Bangalore":
        "Royal Challengers Bengaluru",

    "Delhi Daredevils":
        "Delhi Capitals",

    "Kings XI Punjab":
        "Punjab Kings",

    "Rising Pune Supergiants":
        "Rising Pune Supergiant"
}

team_cols = [
    "Bat First",
    "Bat Second",
    "toss_winner",
    "match_won_by"
]

for col in team_cols:

    if col in train.columns:

        train[col] = train[col].replace(team_map)

# CREATE MATCH LEVEL DATA

innings1 = (
    train[train["Innings"] == 1]
    .groupby("Match ID")["Runs From Ball"]
    .sum()
)

innings2 = (
    train[train["Innings"] == 2]
    .groupby("Match ID")["Runs From Ball"]
    .sum()
)

second_wickets = (
    train[
        (train["Innings"] == 2)
        &
        (train["Wicket"] == 1)
    ]
    .groupby("Match ID")
    .size()
)

meta_cols = [

    "Match ID",
    "Date",
    "Venue",
    "city",

    "Bat First",
    "Bat Second",

    "toss_winner",
    "toss_decision",

    "match_won_by"
]

matches = train[meta_cols].drop_duplicates("Match ID")

matches = matches.merge(

    innings1.rename("innings1_runs"),

    on="Match ID",

    how="left"
)

matches = matches.merge(

    innings2.rename("innings2_runs"),

    on="Match ID",

    how="left"
)

matches = matches.merge(

    second_wickets.rename("innings2_wickets"),

    on="Match ID",

    how="left"
)

matches["innings2_wickets"] = (
    matches["innings2_wickets"]
    .fillna(0)
)

# REMOVE INVALID MATCHES

matches = matches[
    matches["match_won_by"].notna()
]

matches = matches[
    matches["match_won_by"] != "No Result"
]

matches = matches[
    matches["match_won_by"] != "Tie"
]

# TEAM A / TEAM B

matches["team_a"] = matches["Bat First"]

matches["team_b"] = matches["Bat Second"]


# REMOVE UNKNOWN WINNERS
# FIXES THE KeyError

matches = matches[
    matches["match_won_by"].isin(
        matches["team_a"]
    )
    |
    matches["match_won_by"].isin(
        matches["team_b"]
    )
]

# CREATE TARGET

def create_target(row):

    if row["match_won_by"] == row["team_a"]:

        margin = (
            row["innings1_runs"]
            -
            row["innings2_runs"]
        )

        if margin > 20:
            return "A_big"

        else:
            return "A_small"

    else:

        wickets_left = (
            10 - row["innings2_wickets"]
        )

        if wickets_left >= 6:
            return "B_big"

        else:
            return "B_small"

matches["target"] = matches.apply(
    create_target,
    axis=1
)

print("\nTarget Distribution:\n")

print(matches["target"].value_counts())

# SORT BY DATE

matches["Date"] = pd.to_datetime(
    matches["Date"],
    errors="coerce"
)

matches = matches.sort_values("Date")

# RECENT FORM FEATURES

team_history = {}

recent_form_a = []
recent_form_b = []

for idx, row in matches.iterrows():

    ta = row["team_a"]
    tb = row["team_b"]

    recent_a = team_history.get(ta, [])
    recent_b = team_history.get(tb, [])

    recent_form_a.append(

        np.mean(recent_a[-5:])
        if len(recent_a) > 0
        else 0.5
    )

    recent_form_b.append(

        np.mean(recent_b[-5:])
        if len(recent_b) > 0
        else 0.5
    )

    winner = row["match_won_by"]

    if winner == ta:

        team_history.setdefault(ta, []).append(1)
        team_history.setdefault(tb, []).append(0)

    else:

        team_history.setdefault(ta, []).append(0)
        team_history.setdefault(tb, []).append(1)

matches["recent_form_a"] = recent_form_a
matches["recent_form_b"] = recent_form_b

# TEAM WIN %

all_teams = pd.concat([

    matches[["team_a"]]
    .rename(columns={"team_a": "team"}),

    matches[["team_b"]]
    .rename(columns={"team_b": "team"})
])

team_total = (
    all_teams["team"]
    .value_counts()
    .to_dict()
)

team_wins = (
    matches["match_won_by"]
    .value_counts()
    .to_dict()
)

team_win_pct = {}

for team in team_total:

    wins = team_wins.get(team, 0)

    total = team_total.get(team, 1)

    team_win_pct[team] = wins / total

matches["team_a_win_pct"] = (
    matches["team_a"]
    .map(team_win_pct)
)

matches["team_b_win_pct"] = (
    matches["team_b"]
    .map(team_win_pct)
)

# HEAD TO HEAD FEATURES

h2h_dict = {}

h2h_team_a = []

for idx, row in matches.iterrows():

    ta = row["team_a"]
    tb = row["team_b"]

    pair = tuple(sorted([ta, tb]))

    if pair not in h2h_dict:

        h2h_dict[pair] = {
            ta: 0,
            tb: 0
        }

    total = sum(
        h2h_dict[pair].values()
    )

    if total == 0:

        h2h_team_a.append(0.5)

    else:

        h2h_team_a.append(

            h2h_dict[pair].get(ta, 0)
            /
            total
        )

    winner = row["match_won_by"]

    # SAFE UPDATE
    # FIXES UNKNOWN ERROR

    if winner in h2h_dict[pair]:

        h2h_dict[pair][winner] += 1

matches["h2h_team_a"] = h2h_team_a

# VENUE FEATURES

venue_avg_score = (
    matches.groupby("Venue")["innings1_runs"]
    .mean()
)

matches["venue_avg_score"] = (
    matches["Venue"]
    .map(venue_avg_score)
)

venue_chase_success = (

    matches.groupby("Venue")
    .apply(

        lambda x:

        np.mean(
            x["match_won_by"]
            ==
            x["team_b"]
        )
    )
)

matches["venue_chase_success"] = (
    matches["Venue"]
    .map(venue_chase_success)
)

# TOSS FEATURES

matches["team_a_won_toss"] = (

    matches["toss_winner"]
    ==
    matches["team_a"]

).astype(int)

matches["field_first"] = (

    matches["toss_decision"]
    ==
    "field"

).astype(int)

# SELECT FEATURES

feature_cols = [

    "team_a",
    "team_b",

    "Venue",
    "city",

    "team_a_win_pct",
    "team_b_win_pct",

    "recent_form_a",
    "recent_form_b",

    "h2h_team_a",

    "venue_avg_score",
    "venue_chase_success",

    "team_a_won_toss",
    "field_first"
]

X = matches[feature_cols].copy()

y = matches["target"]


# LABEL ENCODING


encoders = {}

cat_cols = [
    "team_a",
    "team_b",
    "Venue",
    "city"
]

for col in cat_cols:

    le = LabelEncoder()

    X[col] = le.fit_transform(
        X[col].astype(str)
    )

    encoders[col] = le

# TARGET ENCODING


target_encoder = LabelEncoder()

y_encoded = target_encoder.fit_transform(y)

print("\nEncoded Classes:\n")

for idx, cls in enumerate(target_encoder.classes_):

    print(idx, "=", cls)
  
# IMPUTE MISSING VALUES

imputer = SimpleImputer(
    strategy="median"
)

X = imputer.fit_transform(X)

# PREPARE FUTURE MATCHES

future = pd.concat([

    public_lb,
    schedule

], ignore_index=True)

future["team_a_win_pct"] = (
    future["team_a"]
    .map(team_win_pct)
    .fillna(0.5)
)

future["team_b_win_pct"] = (
    future["team_b"]
    .map(team_win_pct)
    .fillna(0.5)
)

future["recent_form_a"] = 0.5
future["recent_form_b"] = 0.5

future["h2h_team_a"] = 0.5

future["venue_avg_score"] = (
    future["venue"]
    .map(venue_avg_score)
    .fillna(matches["innings1_runs"].mean())
)

future["venue_chase_success"] = (
    future["venue"]
    .map(venue_chase_success)
    .fillna(0.5)
)

future["team_a_won_toss"] = 0

future["field_first"] = 1

future["Venue"] = future["venue"]

future_X = future[feature_cols].copy()

# APPLY LABEL ENCODING

for col in cat_cols:

    known_classes = set(
        encoders[col].classes_
    )

    future_X[col] = future_X[col].apply(

        lambda x:

        x
        if str(x) in known_classes
        else encoders[col].classes_[0]
    )

    future_X[col] = encoders[col].transform(
        future_X[col].astype(str)
    )

future_X = imputer.transform(future_X)

# STRATIFIED KFOLD

skf = StratifiedKFold(

    n_splits=5,
    shuffle=True,
    random_state=42
)

oof_probs = np.zeros((len(X), 4))

test_predictions = []

fold = 1

# TRAINING LOOP

for train_idx, valid_idx in skf.split(X, y_encoded):

    print(f"\nTraining Fold {fold}")

    X_train = X[train_idx]
    X_valid = X[valid_idx]

    y_train = y_encoded[train_idx]
    y_valid = y_encoded[valid_idx]

    # XGBOOST

    xgb = XGBClassifier(

        objective="multi:softprob",

        num_class=4,

        n_estimators=600,

        learning_rate=0.03,

        max_depth=7,

        subsample=0.8,

        colsample_bytree=0.8,

        eval_metric="mlogloss",

        random_state=42
    )

    calibrated_xgb = CalibratedClassifierCV(

        xgb,

        method="sigmoid",

        cv=3
    )

    calibrated_xgb.fit(
        X_train,
        y_train
    )

    xgb_valid = calibrated_xgb.predict_proba(
        X_valid
    )

    # CATBOOST

    cat = CatBoostClassifier(

        iterations=600,

        depth=7,

        learning_rate=0.03,

        loss_function="MultiClass",

        verbose=0,

        random_seed=42
    )

    cat.fit(
        X_train,
        y_train
    )

    cat_valid = cat.predict_proba(
        X_valid
    )

    # ENSEMBLE

    final_valid = (

        xgb_valid * 0.7

        +

        cat_valid * 0.3
    )

    oof_probs[valid_idx] = final_valid

    fold_loss = log_loss(
        y_valid,
        final_valid
    )

    print("Fold Log Loss:", fold_loss)

    # FUTURE PREDICTIONS

    xgb_test = calibrated_xgb.predict_proba(
        future_X
    )

    cat_test = cat.predict_proba(
        future_X
    )

    final_test = (

        xgb_test * 0.7

        +

        cat_test * 0.3
    )

    test_predictions.append(final_test)

    fold += 1

# FINAL CV SCORE

cv_score = log_loss(
    y_encoded,
    oof_probs
)

print("\nFINAL CV LOG LOSS:", cv_score)

# AVERAGE TEST PREDICTIONS

final_probs = np.mean(
    test_predictions,
    axis=0
)

# CREATE SUBMISSION

submission = sample.copy()

submission["A_small"] = 0.25
submission["A_big"] = 0.25
submission["B_small"] = 0.25
submission["B_big"] = 0.25

class_names = target_encoder.classes_

for i in range(min(len(final_probs), len(submission))):

    probs = final_probs[i]

    prob_map = dict(
        zip(class_names, probs)
    )

    submission.loc[i, "A_small"] = (
        prob_map.get("A_small", 0.25)
    )

    submission.loc[i, "A_big"] = (
        prob_map.get("A_big", 0.25)
    )

    submission.loc[i, "B_small"] = (
        prob_map.get("B_small", 0.25)
    )

    submission.loc[i, "B_big"] = (
        prob_map.get("B_big", 0.25)
    )

# NORMALIZE PROBABILITIES

prob_cols = [

    "A_small",
    "A_big",
    "B_small",
    "B_big"
]

submission[prob_cols] = (

    submission[prob_cols]
    .clip(0.0001, 0.9999)
)

submission[prob_cols] = (

    submission[prob_cols]
    .div(

        submission[prob_cols]
        .sum(axis=1),

        axis=0
    )
)

# VALIDATION

print("\nSubmission Shape:")
print(submission.shape)

print("\nMissing Values:")
print(submission.isnull().sum())

print("\nProbability Row Sums:")
print(
    submission[prob_cols]
    .sum(axis=1)
    .head()
)

# SAVE SUBMISSION

submission.to_csv(
    "submission.csv",
    index=False
)

print("\nsubmission.csv created successfully\n")

print(submission.head())

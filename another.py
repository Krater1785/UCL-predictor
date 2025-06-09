# === TRAINING SCRIPT ===

import pandas as pd
import json
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load data
features_df = pd.read_csv("Team_Match_Features_2.csv", parse_dates=["Match_Date"])
with open("train.json") as f:
    train_data = json.load(f)
mapping_df = pd.read_csv("auto_team_name_mapping.csv")

team_map = dict(zip(mapping_df["train_team"], mapping_df["Mapped_Team_Name"]))
features_df["Team_Name"] = features_df["Team_Name"].str.upper()

t5_teams = set([
    "Manchester City", "Manchester United", "Chelsea", "Arsenal", "Liverpool", "Tottenham Hotspur", "Newcastle United",
    "Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Valencia", "Villarreal", "Athletic Bilbao",
    "Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen", "Schalke 04", "Wolfsburg", "Eintracht Frankfurt",
    "Juventus", "Inter Milan", "Milan", "Napoli", "Roma", "Atalanta", "Lazio",
    "Paris Saint-Germain", "Monaco", "Lille", "Lyon", "Marseille"
])

base_prestige_map = {
    "REAL MADRID": 6, "BARCELONA": 4, "BAYERN MUNICH": 4, "JUVENTUS": 3,
    "MANCHESTER CITY": 2, "LIVERPOOL": 3, "CHELSEA": 3, "PARIS SAINT-GERMAIN": 2,
    "ATLETICO MADRID": 2, "AC MILAN": 2, "INTER MILAN": 2, "MANCHESTER UNITED": 2,
    "ARSENAL": 2, "BORUSSIA DORTMUND": 2, "PORTO": 1, "BENFICA": 1, "ROMA": 2,
    "NAPOLI": 2, "SEVILLA": 1, "TOTTENHAM HOTSPUR": 2, "MONACO": 1, "LYON": 1,
    "VILLARREAL": 1, "CELTIC": 1, "RANGERS": 1
}

# Compute prestige history with decay
prestige_history = {}
seasons_sorted = sorted(train_data.keys())

for season in seasons_sorted:
    season_start = int(season[:4])
    final_teams = []
    if "final" in train_data[season]:
        final_teams = [team_map.get(train_data[season]["final"][0]["team_1"], train_data[season]["final"][0]["team_1"].upper()),
                       team_map.get(train_data[season]["final"][0]["team_2"], train_data[season]["final"][0]["team_2"].upper())]
    prestige_history[season] = {}
    all_teams = set(base_prestige_map.keys()).union(final_teams)
    for team in all_teams:
        prestige = base_prestige_map.get(team, 1)
        appeared_last_year = False
        appeared_2_years_ago = False
        for i, bonus in zip([1, 2], [2, 1]):
            prev_season = f"{season_start - i}-{str(season_start - i + 1)[2:]}"
            if prev_season in prestige_history and team in prestige_history[prev_season].get("final_teams", []):
                if i == 1:
                    appeared_last_year = True
                elif i == 2:
                    appeared_2_years_ago = True
        if appeared_last_year:
            prestige += 2
        elif appeared_2_years_ago:
            prestige += 1
        prestige_history[season][team] = prestige
    prestige_history[season]["final_teams"] = final_teams

# Helper

def map_team(name):
    return team_map.get(name, name.upper())

samples = []

for season, rounds in train_data.items():
    for round_name, matches in rounds.items():
        for match in matches:
            team_1 = map_team(match["team_1"])
            team_2 = map_team(match["team_2"])
            date = datetime.strptime(match["date"], "%d/%m/%Y")
            winner = map_team(match["winner"])

            row1 = features_df[(features_df["Team_Name"] == team_1) & (features_df["Match_Date"] < date)]
            row2 = features_df[(features_df["Team_Name"] == team_2) & (features_df["Match_Date"] < date)]
            if row1.empty or row2.empty:
                continue

            row1 = row1.sort_values("Match_Date").iloc[-1]
            row2 = row2.sort_values("Match_Date").iloc[-1]

            p1 = prestige_history.get(season, {}).get(team_1, base_prestige_map.get(team_1, 1))
            p2 = prestige_history.get(season, {}).get(team_2, base_prestige_map.get(team_2, 1))
            scale1 = 0.5
            scale2 = 0.5
            if team_1 in t5_teams:
                scale1 = 1.0
            if team_2 in t5_teams:
                scale2 = 1.0
            features = {
                "team1_goals_scored": (float(row1["Team_Goals_Scored"]))*scale1,
                "team1_goals_conceded": (float(row1["Team_Goals_Conceded"]))*scale1,
                "team1_points": (float(row1["Team_Points"]))*scale1,
                # "team1_form": (float((row1["Recent_Form"])*(team_1 in t5_teams)))*scale1,
                "team1_is_top5": int(team_1 in t5_teams),
                "team1_prestige": int(p1),
                "team2_goals_scored": (float(row2["Team_Goals_Scored"]))*scale2,
                "team2_goals_conceded": (float(row2["Team_Goals_Conceded"]))*scale2,
                "team2_points": (float(row2["Team_Points"]))*scale2,
                # "team2_form": (float(row2["Recent_Form"]))*scale2,
                "team2_is_top5": int(team_2 in t5_teams),
                "team2_prestige": int(p2),
                "goal_diff": float(row1["Team_Goals_Scored"] - row2["Team_Goals_Scored"]),
                "form_diff": float(row1["Recent_Form"] - row2["Recent_Form"]),
                "prestige_diff": int(p1 - p2),
                "label": int(winner == team_1)
            }
            samples.append(features)

# Train
train_df = pd.DataFrame(samples)
X = train_df.drop("label", axis=1)
y = train_df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb = XGBClassifier(eval_metric="logloss")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

ensemble_model = VotingClassifier(estimators=[
    ('xgb', xgb),
    ('rf', rf),
    ('gb', gb)
], voting='soft')

ensemble_model.fit(X_train, y_train)
pickle.dump(ensemble_model, open("xgb_model_2.pkl", "wb"))
print(f"Training Accuracy: {accuracy_score(y_test, ensemble_model.predict(X_test)):.2%}")

import pandas as pd
import json
from datetime import datetime
import pickle

# Load model and data
with open("xgb_model_2.pkl", "rb") as f:
    model = pickle.load(f)

features_df = pd.read_csv("Team_Match_Features_2.csv", parse_dates=["Match_Date"])
mapping_df = pd.read_csv("auto_team_name_mapping.csv")
with open("test_matchups.json") as f:
    test_data = json.load(f)

team_map = dict(zip(mapping_df["train_team"], mapping_df["Mapped_Team_Name"]))
features_df["Team_Name"] = features_df["Team_Name"].str.upper()

t5_teams = set([
    "Manchester City", "Manchester United", "Chelsea", "Arsenal", "Liverpool", "Tottenham Hotspur", "Newcastle United",
    "Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Valencia", "Villarreal", "Athletic Bilbao",
    "Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen", "Schalke 04", "Wolfsburg", "Eintracht Frankfurt",
    "Juventus", "Inter Milan", "Milan", "Napoli", "Roma", "Atalanta", "Lazio",
    "Paris Saint-Germain", "Monaco", "Lille", "Lyon", "Marseille"
])
base_prestige_map = {
    "REAL MADRID": 7, "BARCELONA": 2, "BAYERN MUNICH": 4, "JUVENTUS": 2,
    "MANCHESTER CITY": 3, "LIVERPOOL": 3, "CHELSEA": 2, "PARIS SAINT-GERMAIN": 3,
    "ATLETICO MADRID": 2, "AC MILAN": 2, "INTER MILAN": 2, "MANCHESTER UNITED": 2,
    "ARSENAL": 2, "BORUSSIA DORTMUND": 2, "PORTO": 1, "BENFICA": 1, "ROMA": 2,
    "NAPOLI": 2, "SEVILLA": 1, "TOTTENHAM HOTSPUR": 2, "MONACO": 1, "LYON": 1,
    "VILLARREAL": 1, "CELTIC": 1, "RANGERS": 1
}

def map_team(name):
    return team_map.get(name, name.upper())
features_df.fillna(0, inplace=True)
def get_latest_features(team_name, date):
    df = features_df[(features_df["Team_Name"] == team_name.upper()) & (features_df["Match_Date"] < date)]
    if df.empty:
        return None
    row = df.sort_values("Match_Date").iloc[-1]
    if team_name not in t5_teams:
        scale = 0.5
        row["Team_Goals_Scored"] *= scale
        row["Team_Goals_Conceded"] *= scale
        row["Team_Points"] *= scale
        row["Recent_Form"] *= scale
    return row

def get_prestige_scores(train_data, base_prestige_map, form_weight=0.4):
    """
    Returns a dictionary mapping each team to its updated prestige score
    that blends base prestige and recent form.
    """
    from collections import defaultdict
    form_scores = defaultdict(lambda: [0, 0])  # [wins, matches]
    for match in train_data:
        winner = match["winner"]
        loser = match["team_1"] if match["team_2"] == winner else match["team_2"]
        form_scores[winner][0] += 1
        form_scores[winner][1] += 1
        form_scores[loser][1] += 1

    prestige_scores = {}
    for team in base_prestige_map:
        base = base_prestige_map[team]
        wins, total = form_scores[team]
        form = wins / total if total > 0 else 0.5  # default to neutral form
        prestige_scores[team] = (1 - form_weight) * base + form_weight * form

    return prestige_scores
# Assuming train_data and base_prestige_map are already defined
prestige_scores = get_prestige_scores(train_data, base_prestige_map)


def predict_match(team1, team2, date):
    t1 = map_team(team1)
    t2 = map_team(team2)
    row1 = get_latest_features(t1, date)
    row2 = get_latest_features(t2, date)
    if row1 is None or row2 is None:
        print(f"No data for {team1} or {team2} before {date}")
        return team1
    # Check for NaNs
    if row1.isnull().any():
        print(f"NaN in features for {team1} before {date}:")
        print(row1[row1.isnull()])
    if row2.isnull().any():
        print(f"NaN in features for {team2} before {date}:")
        print(row2[row2.isnull()])
    p1 = base_prestige_map.get(t1, 1)
    p2 = base_prestige_map.get(t2, 1)
    features = pd.DataFrame([{
        "team1_goals_scored": float(row1["Team_Goals_Scored"]),
        "team1_goals_conceded": float(row1["Team_Goals_Conceded"]),
        "team1_points": float(row1["Team_Points"]),
        # "team1_form": float((row1["Recent_Form"])*(t1 in t5_teams)),
        "team1_is_top5": int(t1 in t5_teams),
        "team1_prestige": p1,
        "team2_goals_scored": float(row2["Team_Goals_Scored"]),
        "team2_goals_conceded": float(row2["Team_Goals_Conceded"]),
        "team2_points": float(row2["Team_Points"]),
        # "team2_form": float((row1["Recent_Form"])*(t2 in t5_teams)),
        "team2_is_top5": int(t2 in t5_teams),
        "team2_prestige": p2,
        "goal_diff": float(row1["Team_Goals_Scored"] - row2["Team_Goals_Scored"]),
        "form_diff": float(row1["Recent_Form"] - row2["Recent_Form"]),
        "prestige_diff": int(p1 - p2),
    }])
    if features.isnull().any(axis=None):
        print(f"NaN in feature vector for {team1} vs {team2} on {date}")
        print(features)
        return team1
    pred = model.predict(features)[0]
    p1 = prestige_scores.get(team_1, 0.5)
    p2 = prestige_scores.get(team_2, 0.5)
    return team_1 if p1 >= p2 else team_2


def resolve_placeholder(name, r16_results, qf_results=None, sf_results=None):
    if name.startswith("Winner of QF"):
        idx = int(name.replace("Winner of QF", "")) - 1
        if qf_results is not None and 0 <= idx < len(qf_results):
            return qf_results[idx]["winner"]
        raise ValueError(f"Could not resolve placeholder '{name}' in qf_results")
    elif name.startswith("Winner of SF"):
        idx = int(name.replace("Winner of SF", "")) - 1
        if sf_results is not None and 0 <= idx < len(sf_results):
            return sf_results[idx]["winner"]
        raise ValueError(f"Could not resolve placeholder '{name}' in sf_results")
    elif name.startswith("Winner of "):
        matchup = name[len("Winner of "):]
        for match in r16_results:
            if f"{match['team_1']} vs {match['team_2']}" == matchup:
                return match["winner"]
        raise ValueError(f"Could not resolve placeholder '{name}' in r16_results")
    else:
        return name

submission_rows = []
season_id = 0

for season, rounds in test_data.items():
    # --- Round of 16 ---
    r16_matches = rounds["round_of_16_matchups"]
    r16_results = []
    for match in r16_matches:
        t1, t2 = match["team_1"], match["team_2"]
        print(f"{t1} and {t1 in t5_teams}")
        date = datetime.strptime(match["date"], "%Y-%m-%d")
        winner = predict_match(t1, t2, prestige_scores, match["date"])
        r16_results.append({"team_1": t1, "team_2": t2, "winner": winner})

    # --- Quarterfinals ---
    qf_matches = rounds["quarter_finals_matchups"]
    qf_results = []
    for match in qf_matches:
        t1 = resolve_placeholder(match["team_1"], r16_results)
        t2 = resolve_placeholder(match["team_2"], r16_results)
        date = datetime.strptime(match["date"], "%Y-%m-%d")
        winner = predict_match(t1, t2, prestige_scores, match["date"])
        qf_results.append({"team_1": t1, "team_2": t2, "winner": winner})

    # --- Semifinals ---
    sf_matches = rounds["semi_finals_matchups"]
    sf_results = []
    for match in sf_matches:
        t1 = resolve_placeholder(match["team_1"], r16_results, qf_results=qf_results)
        t2 = resolve_placeholder(match["team_2"], r16_results, qf_results=qf_results)
        date = datetime.strptime(match["date"], "%Y-%m-%d")
        winner = predict_match(t1, t2, prestige_scores, match["date"])
        sf_results.append({"team_1": t1, "team_2": t2, "winner": winner})

    # --- Final ---
    final_match = rounds["final_matchup"]
    t1 = resolve_placeholder(final_match["team_1"], r16_results, qf_results=qf_results, sf_results=sf_results)
    t2 = resolve_placeholder(final_match["team_2"], r16_results, qf_results=qf_results, sf_results=sf_results)
    date = datetime.strptime(final_match["date"], "%Y-%m-%d")
    winner = predict_match(t1, t2, prestige_scores, match["date"])
    final_results = [{"team_1": t1, "team_2": t2, "winner": winner}]

    # --- Collect all rounds for output ---
    round_results = {
        "round_of_16": r16_results,
        "quarter_finals": qf_results,
        "semi_finals": sf_results,
        "final": final_results
    }
    submission_rows.append({
        "id": season_id,
        "season": season,
        "predictions": json.dumps(round_results)
    })
    season_id += 1

# Save to CSV in sample_submission format
pd.DataFrame(submission_rows).to_csv("sample_submission_like_predictions_3.csv", index=False)
print("sample_submission_like_predictions.csv generated.")
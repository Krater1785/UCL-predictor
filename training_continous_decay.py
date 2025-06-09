import pandas as pd
import json
from datetime import datetime, timedelta
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import math

# Configuration
DECAY_RATE = 0.03  # 3% monthly decay
BASE_SEASON_START_MONTH = 8  # August

# Load data
features_df = pd.read_csv("Team_Match_Features.csv", parse_dates=["Match_Date"])
with open("train.json") as f:
    train_data = json.load(f)
mapping_df = pd.read_csv("auto_team_name_mapping.csv")

# Team mappings and constants
team_map = dict(zip(mapping_df["train_team"], mapping_df["Mapped_Team_Name"]))
features_df["Team_Name"] = features_df["Team_Name"].str.upper()

# Updated base prestige map (more granular and up-to-date)
base_prestige_map = {
    "MANCHESTER CITY": 8, "LIVERPOOL": 8, "REAL MADRID": 9, "BAYERN MUNICH": 8,
    "BARCELONA": 7, "CHELSEA": 7, "PARIS SAINT-GERMAIN": 7, "TOTTENHAM HOTSPUR": 6,
    "ARSENAL": 6, "MANCHESTER UNITED": 6, "ATLETICO MADRID": 7, "JUVENTUS": 7,
    "INTER MILAN": 6, "AC MILAN": 6, "NAPOLI": 6, "BORUSSIA DORTMUND": 6,
    "RB LEIPZIG": 5, "SEVILLA": 5, "VILLARREAL": 5, "PORTO": 4, "BENFICA": 4,
    "LYON": 4, "MONACO": 4, "LAZIO": 5, "ROMA": 5, "ATALANTA": 5, "BASEL": 2,
    "SHAKHTAR DONETSK": 3, "DINAMO ZAGREB": 2, "CLUB BRUGGE": 2, "SALZBURG": 3
}

# Updated top teams list
t5_teams = set([
    "MANCHESTER CITY", "LIVERPOOL", "REAL MADRID", "BAYERN MUNICH",
    "BARCELONA", "CHELSEA", "PARIS SAINT-GERMAIN", "TOTTENHAM HOTSPUR",
    "ARSENAL", "MANCHESTER UNITED", "ATLETICO MADRID", "JUVENTUS",
    "INTER MILAN", "AC MILAN", "NAPOLI", "BORUSSIA DORTMUND"
])

# Enhanced PrestigeCalculator with more achievements
class PrestigeCalculator:
    def __init__(self):
        self.team_history = {}
        
    def add_achievement(self, team, date, bonus, achievement_type):
        if team not in self.team_history:
            self.team_history[team] = []
            
        # Different bonuses for different achievements
        if achievement_type == "ucl_final":
            bonus = 4.0
        elif achievement_type == "ucl_semi":
            bonus = 2.5
        elif achievement_type == "ucl_qf":
            bonus = 1.5
        elif achievement_type == "league_title":
            bonus = 2.0
            
        self.team_history[team].append({
            "date": date,
            "bonus": bonus,
            "type": achievement_type
        })
    
    def calculate_prestige(self, team, current_date):
        base = base_prestige_map.get(team, 1)
        if team not in self.team_history:
            return base
        
        total_bonus = 0
        for achievement in self.team_history[team]:
            months_passed = ((current_date - achievement["date"]).days) / 30.44
            # Slower decay for more significant achievements
            decay_rate = DECAY_RATE * (0.7 if achievement["type"] == "ucl_final" else 1.0)
            decayed_bonus = achievement["bonus"] * math.exp(-decay_rate * months_passed)
            total_bonus += decayed_bonus
        
        return min(base + total_bonus, 10)  # Cap at 10

# Enhanced feature engineering
def create_features(row1, row2, team1, team2, date):
    p1 = prestige_calc.calculate_prestige(team1, date)
    p2 = prestige_calc.calculate_prestige(team2, date)
    
    # Additional features
    team1_avg_goals = row1["Team_Goals_Scored"] / row1["Matches_Played"]
    team2_avg_goals = row2["Team_Goals_Scored"] / row2["Matches_Played"]
    
    return {
        "team1_goals_scored": float(row1["Team_Goals_Scored"]),
        "team1_goals_conceded": float(row1["Team_Goals_Conceded"]),
        "team1_points": float(row1["Team_Points"]),
        "team1_form": float(row1["Recent_Form"]),
        "team1_is_top5": int(team1 in t5_teams),
        "team1_prestige": p1,
        "team1_avg_goals": team1_avg_goals,
        "team2_goals_scored": float(row2["Team_Goals_Scored"]),
        "team2_goals_conceded": float(row2["Team_Goals_Conceded"]),
        "team2_points": float(row2["Team_Points"]),
        "team2_form": float(row2["Recent_Form"]),
        "team2_is_top5": int(team2 in t5_teams),
        "team2_prestige": p2,
        "team2_avg_goals": team2_avg_goals,
        "goal_diff": float(row1["Team_Goals_Scored"] - row2["Team_Goals_Scored"]),
        "form_diff": float(row1["Recent_Form"] - row2["Recent_Form"]),
        "prestige_diff": float(p1 - p2),
        "prestige_ratio": float(p1 / (p2 + 0.1)),  # Avoid division by zero
        "label": int(winner == team1)
    }

# Enhanced model with class weights
ensemble_model = VotingClassifier(estimators=[
    ('xgb', XGBClassifier(
        eval_metric="logloss", 
        n_estimators=200,
        scale_pos_weight=1.5,  # Favor favorites slightly more
        max_depth=6
    )),
    ('rf', RandomForestClassifier(
        n_estimators=150, 
        random_state=42,
        class_weight="balanced_subsample"
    )),
    ('gb', GradientBoostingClassifier(
        n_estimators=150, 
        random_state=42,
        max_depth=5
    ))
], voting='soft', weights=[1.5, 1, 1])  # Give XGBoost more weight

# Modified predict_match function
def predict_match(team1, team2, date):
    # ... (previous code)
    
    pred_prob = model.predict_proba(features)[0]
    # Higher threshold for favorites
    if p1 > p2 * 1.5:  # If team1 is significantly stronger
        return team1 if pred_prob[1] >= 0.4 else team2
    else:
        return team1 if pred_prob[1] >= 0.5 else team2

# Create training samples
samples = []
for season, rounds in train_data.items():
    for round_name, matches in rounds.items():
        for match in matches:
            team_1 = team_map.get(match["team_1"], match["team_1"].upper())
            team_2 = team_map.get(match["team_2"], match["team_2"].upper())
            date = datetime.strptime(match["date"], "%d/%m/%Y")
            winner = team_map.get(match["winner"], match["winner"].upper())

            # Get recent features
            def get_team_features(team):
                team_rows = features_df[
                    (features_df["Team_Name"] == team) & 
                    (features_df["Match_Date"] < date)
                ]
                if team_rows.empty:
                    return None
                return team_rows.sort_values("Match_Date").iloc[-1]
            
            row1 = get_team_features(team_1)
            row2 = get_team_features(team_2)
            if row1 is None or row2 is None:
                continue

            # Calculate decayed prestige
            p1 = prestige_calc.calculate_prestige(team_1, date)
            p2 = prestige_calc.calculate_prestige(team_2, date)

            features = {
                "team1_goals_scored": float(row1["Team_Goals_Scored"]),
                "team1_goals_conceded": float(row1["Team_Goals_Conceded"]),
                "team1_points": float(row1["Team_Points"]),
                "team1_form": float(row1["Recent_Form"]),
                "team1_is_top5": int(team_1 in t5_teams),
                "team1_prestige": p1,
                "team2_goals_scored": float(row2["Team_Goals_Scored"]),
                "team2_goals_conceded": float(row2["Team_Goals_Conceded"]),
                "team2_points": float(row2["Team_Points"]),
                "team2_form": float(row2["Recent_Form"]),
                "team2_is_top5": int(team_2 in t5_teams),
                "team2_prestige": p2,
                "goal_diff": float(row1["Team_Goals_Scored"] - row2["Team_Goals_Scored"]),
                "form_diff": float(row1["Recent_Form"] - row2["Recent_Form"]),
                "prestige_diff": float(p1 - p2),
                "label": int(winner == team_1)
            }
            samples.append(features)

# Train model
train_df = pd.DataFrame(samples)
X = train_df.drop("label", axis=1)
y = train_df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ensemble_model = VotingClassifier(estimators=[
    ('xgb', XGBClassifier(eval_metric="logloss", n_estimators=150)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
], voting='soft')

ensemble_model.fit(X_train, y_train)
pickle.dump({
    'model': ensemble_model,
    'prestige_calc': prestige_calc,
    'base_prestige_map': base_prestige_map
}, open("model_with_decay.pkl", "wb"))

print(f"✅ Training Accuracy: {accuracy_score(y_test, ensemble_model.predict(X_test)):.2%}")

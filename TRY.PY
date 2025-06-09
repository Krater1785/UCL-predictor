import pandas as pd
import json
from datetime import datetime
import pickle
import numpy as np
from sklearn.ensemble import VotingClassifier

# Load model and data
with open("xgb_model_2.pkl", "rb") as f:
    model = pickle.load(f)

features_df = pd.read_csv("Team_Match_Features_2.csv", parse_dates=["Match_Date"])
mapping_df = pd.read_csv("auto_team_name_mapping.csv")
with open("test_matchups.json") as f:
    test_data = json.load(f)

# Team mapping and preprocessing
team_map = dict(zip(mapping_df["train_team"], mapping_df["Mapped_Team_Name"]))
features_df["Team_Name"] = features_df["Team_Name"].str.upper()

# Enhanced team categorizations
T5_LEAGUES = {
    "PREMIER_LEAGUE": {
        "MANCHESTER CITY", "MANCHESTER UNITED", "CHELSEA", "ARSENAL", "LIVERPOOL",
        "TOTTENHAM HOTSPUR", "NEWCASTLE UNITED"
    },
    "LA_LIGA": {
        "REAL MADRID", "BARCELONA", "ATLETICO MADRID", "SEVILLA", "VALENCIA",
        "VILLARREAL", "ATHLETIC BILBAO"
    },
    "BUNDESLIGA": {
        "BAYERN MUNICH", "BORUSSIA DORTMUND", "RB LEIPZIG", "BAYER LEVERKUSEN",
        "WOLFSBURG", "EINTRACHT FRANKFURT"
    },
    "SERIE_A": {
        "JUVENTUS", "INTER MILAN", "MILAN", "NAPOLI", "ROMA", "ATALANTA", "LAZIO"
    },
    "LIGUE_1": {
        "PARIS SAINT-GERMAIN", "MONACO", "LYON", "MARSEILLE", "LILLE"
    }
}

# Historical prestige factors (0-10 scale)
PRESTIGE_RANKINGS = {
    "REAL MADRID": 10, "BARCELONA": 9, "BAYERN MUNICH": 9, 
    "MANCHESTER UNITED": 8, "LIVERPOOL": 8, "JUVENTUS": 8,
    "MILAN": 8, "INTER MILAN": 7, "CHELSEA": 7, "ARSENAL": 6,
    "MANCHESTER CITY": 9, "PARIS SAINT-GERMAIN": 7, "ATLETICO MADRID": 7,
    "TOTTENHAM HOTSPUR": 5, "BORUSSIA DORTMUND": 6, "PORTO": 5, "BENFICA": 5,
    "AJAX": 6, "PSV EINDHOVEN": 4, "CELTIC": 4, "RANGERS": 4
}

def map_team(name):
    """Normalize team names using mapping table"""
    return team_map.get(name, name.upper())

def is_t5_team(team_name):
    """Check if team is in any top 5 league"""
    team_upper = team_name.upper()
    for league in T5_LEAGUES.values():
        if team_upper in league:
            return True
    return False

def get_prestige_score(team_name):
    """Get prestige score (0-10) for a team"""
    return PRESTIGE_RANKINGS.get(team_name.upper(), 3)  # Default=3 for other teams

def get_team_weight(team_name):
    """
    Calculate composite weight based on:
    - League strength
    - Historical prestige
    - Recent Champions League performance
    """
    team_upper = team_name.upper()
    weight = 1.0
    
    # League weighting
    if team_upper in T5_LEAGUES["PREMIER_LEAGUE"]:
        weight *= 1.4  # Premier League premium
    elif team_upper in T5_LEAGUES["LA_LIGA"]:
        weight *= 1.35  # La Liga slightly less
    elif is_t5_team(team_name):
        weight *= 1.25  # Other T5 leagues
    
    # Prestige boost (adds 0-100% based on historical strength)
    prestige = get_prestige_score(team_name)
    weight *= (1 + prestige/10)  
    
    # Recent CL performance bonus (if applicable)
    if team_upper in ["MANCHESTER CITY", "CHELSEA", "LIVERPOOL"]:
        weight *= 1.1  # Recent winners/contenders
    
    return min(weight, 2.5)  # Cap at 2.5x to prevent extreme values

def get_latest_features(team_name, date):
    """Get the most recent pre-match features with applied weights"""
    team_upper = team_name.upper()
    df = features_df[
        (features_df["Team_Name"] == team_upper) & 
        (features_df["Match_Date"] < date)
    ]
    
    if df.empty:
        return None
    
    row = df.sort_values("Match_Date").iloc[-1].copy()
    weight = get_team_weight(team_name)
    
    # Apply weighted adjustments
    row["Team_Goals_Scored"] *= weight
    row["Team_Points"] *= weight
    row["Recent_Form"] *= weight
    row["Team_Goals_Conceded"] *= (1/weight)  # Better teams concede less
    
    return row

def predict_match(team1, team2, date):
    """Predict match outcome using model with all features"""
    t1 = map_team(team1)
    t2 = map_team(team2)
    row1 = get_latest_features(t1, date)
    row2 = get_latest_features(t2, date)
    
    # Fallback if no data available
    if row1 is None or row2 is None:
        return team1 if get_team_weight(t1) >= get_team_weight(t2) else team2
    
    # Create feature vector matching training data exactly
    features = pd.DataFrame([{
        "team1_goals_scored": float(row1["Team_Goals_Scored"]),
        "team1_goals_conceded": float(row1["Team_Goals_Conceded"]),
        "team1_points": float(row1["Team_Points"]),
        "team1_is_top5": int(is_t5_team(t1)),
        "team1_prestige": get_prestige_score(t1),
        "team2_goals_scored": float(row2["Team_Goals_Scored"]),
        "team2_goals_conceded": float(row2["Team_Goals_Conceded"]),
        "team2_points": float(row2["Team_Points"]),
        "team2_is_top5": int(is_t5_team(t2)),
        "team2_prestige": get_prestige_score(t2),
        "goal_diff": float(row1["Team_Goals_Scored"] - row2["Team_Goals_Scored"]),
        "form_diff": float(row1["Recent_Form"] - row2["Recent_Form"]),
        "prestige_diff": get_prestige_score(t1) - get_prestige_score(t2)
    }])
    
    try:
        pred = model.predict(features)[0]
        return team1 if pred == 1 else team2
    except Exception as e:
        print(f"Prediction error for {team1} vs {team2}: {str(e)}")
        return team1 if get_team_weight(t1) >= get_team_weight(t2) else team2

def resolve_placeholder(name, r16_results, qf_results=None, sf_results=None):
    """Resolve 'Winner of X' placeholders in tournament brackets"""
    if name.startswith("Winner of QF"):
        idx = int(name.replace("Winner of QF", "")) - 1
        if qf_results and 0 <= idx < len(qf_results):
            return qf_results[idx]["winner"]
    elif name.startswith("Winner of SF"):
        idx = int(name.replace("Winner of SF", "")) - 1
        if sf_results and 0 <= idx < len(sf_results):
            return sf_results[idx]["winner"]
    elif name.startswith("Winner of "):
        matchup = name[len("Winner of "):]
        for match in r16_results:
            if f"{match['team_1']} vs {match['team_2']}" == matchup:
                return match["winner"]
    return name

def generate_predictions(test_data):
    """Generate predictions for all tournament stages"""
    submission_rows = []
    season_id = 0
    
    for season, rounds in test_data.items():
        print(f"\nProcessing {season}...")
        
        # Round of 16
        r16_results = []
        for match in rounds["round_of_16_matchups"]:
            winner = predict_match(
                match["team_1"], 
                match["team_2"], 
                datetime.strptime(match["date"], "%Y-%m-%d")
            )
            r16_results.append({
                "team_1": match["team_1"],
                "team_2": match["team_2"],
                "winner": winner
            })
        
        # Quarterfinals
        qf_results = []
        for match in rounds["quarter_finals_matchups"]:
            t1 = resolve_placeholder(match["team_1"], r16_results)
            t2 = resolve_placeholder(match["team_2"], r16_results)
            winner = predict_match(
                t1, t2, 
                datetime.strptime(match["date"], "%Y-%m-%d")
            )
            qf_results.append({
                "team_1": t1,
                "team_2": t2,
                "winner": winner
            })
        
        # Semifinals
        sf_results = []
        for match in rounds["semi_finals_matchups"]:
            t1 = resolve_placeholder(match["team_1"], r16_results, qf_results=qf_results)
            t2 = resolve_placeholder(match["team_2"], r16_results, qf_results=qf_results)
            winner = predict_match(
                t1, t2,
                datetime.strptime(match["date"], "%Y-%m-%d")
            )
            sf_results.append({
                "team_1": t1,
                "team_2": t2,
                "winner": winner
            })
        
        # Final
        final_match = rounds["final_matchup"]
        t1 = resolve_placeholder(
            final_match["team_1"], r16_results, 
            qf_results=qf_results, sf_results=sf_results
        )
        t2 = resolve_placeholder(
            final_match["team_2"], r16_results,
            qf_results=qf_results, sf_results=sf_results
        )
        winner = predict_match(
            t1, t2,
            datetime.strptime(final_match["date"], "%Y-%m-%d")
        )
        
        submission_rows.append({
            "id": season_id,
            "season": season,
            "predictions": json.dumps({
                "round_of_16": r16_results,
                "quarter_finals": qf_results,
                "semi_finals": sf_results,
                "final": [{
                    "team_1": t1,
                    "team_2": t2,
                    "winner": winner
                }]
            })
        })
        season_id += 1
    
    return submission_rows

# Main execution
if __name__ == "__main__":
    features_df.fillna(0, inplace=True)
    
    print("Generating predictions with:")
    print(f"- {len(PRESTIGE_RANKINGS)} prestige-ranked teams")
    print(f"- Model: {type(model).__name__}")
    
    submissions = generate_predictions(test_data)
    pd.DataFrame(submissions).to_csv("final_predictions_full.csv", index=False)
    
    print("\nPrediction complete! Results saved to final_predictions_full.csv")
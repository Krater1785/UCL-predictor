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
features_df = pd.read_csv("Team_Match_Features_Final.csv", parse_dates=["Match_Date"])
with open("train.json") as f:
    train_data = json.load(f)
mapping_df = pd.read_csv("auto_team_name_mapping.csv")

team_map = dict(zip(mapping_df["train_team"], mapping_df["Mapped_Team_Name"]))
features_df["Team_Name"] = features_df["Team_Name"].str.upper()

t5_teams = set([
    "MANCHESTER CITY", "MANCHESTER UTD", "CHELSEA", "ARSENAL", "LIVERPOOL", "TOTTENHAM", "NEWCASTLE",
    "REAL MADRID", "BARCELONA", "ATLETICO MADRID", "SEVILLA", "VALENCIA", "VILLARREAL", "ATHLETIC BILBAO",
    "BAYERN MUNICH", "BORUSSIA DORTMUND", "RB LEIPZIG", "BAYER LEVERKUSEN", "FC SCHALKE 04", "VFL WOLFSBURG", "EINTRACHT FRANKFURT",
    "JUVENTUS", "INTER", "AC MILAN", "NAPOLI", "ROMA", "ATALANTA", "LAZIO",
    "PARIS SAINT GERMAIN", "MONACO", "LILLE", "LYON", "MARSEILLE"
])


base_prestige_map = {
    "REAL MADRID": 10, "BARCELONA": 9, "BAYERN MUNICH": 9, "JUVENTUS": 8,
    "MANCHESTER CITY": 6, "LIVERPOOL": 8, "CHELSEA": 7, "PARIS SAINT GERMAIN": 6,
    "ATLETICO MADRID": 6, "AC MILAN": 7, "INTER": 7, "MANCHESTER UTD": 7,
    "ARSENAL": 5, "BORUSSIA DORTMUND": 5, "PORTO": 4, "BENFICA": 3, "ROMA": 3,
    "NAPOLI": 5, "SEVILLA": 1, "TOTTENHAM": 2, "MONACO": 1, "LYON": 1,
    "VILLARREAL": 1, "CELTIC": 1, "RANGERS": 1,"RB LEIPZIG":0
}
uefa_coeff = {
    '2004-05': {
        'REAL MADRID': 127.166,
        'JUVENTUS': 117.326,
        'LIVERPOOL': 65.769,
        'BAYER LEVERKUSEN': 57.166,
        'PSV': 60.145,
        'MONACO': 52.748,
        'BAYERN MUNICH': 99.166,
        'ARSENAL': 87.915,
        'BARCELONA': 97.166,
        'CHELSEA': 67.915,
        'MANCHESTER UTD': 99.915,
        'AC MILAN': 118.191,
        'WERDER BREMEN': 37.166,
        'LYON': 71.748,
        'FC PORTO': 98.748,
        'INTER': 92.191
    },
    '2005-06': {
        'CHELSEA': 84.006,
        'BARCELONA': 117.006,
        'REAL MADRID': 124.006,
        'ARSENAL': 90.006,
        'WERDER BREMEN': 56.006,
        'JUVENTUS': 101.191,
        'BAYERN MUNICH': 96.006,
        'AC MILAN': 123.191,
        'PSV': 68.145,
        'LYON': 89.748,
        'AJAX': 77.145,
        'INTER': 104.191,
        'BENFICA': 49.533,
        'LIVERPOOL': 80.769,
        'RANGERS': 67.476,
        'VILLARREAL': 49.326
    },
    '2006-07': {
        'CHELSEA': 93.950,
        'FC PORTO': 87.533,
        'CELTIC': 61.023,
        'AC MILAN': 112.191,
        'PSV': 81.145,
        'ARSENAL': 91.950,
        'LILLE': 39.757,
        'MANCHESTER UTD': 85.950,
        'ROMA': 52.191,
        'LYON': 99.748,
        'REAL MADRID': 118.950,
        'BAYERN MUNICH': 97.950,
        'INTER': 99.191,
        'VALENCIA': 89.326,
        'BARCELONA': 129.950,
        'LIVERPOOL': 105.769
    },
    '2007-08': {
        'CELTIC': 64.064,
        'BARCELONA': 117.064,
        'LYON': 99.748,
        'MANCHESTER UTD': 106.064,
        'FC SCHALKE 04': 56.078,
        'FC PORTO': 87.533,
        'LIVERPOOL': 118.769,
        'INTER': 93.191,
        'ROMA': 66.191,
        'REAL MADRID': 118.064,
        'ARSENAL': 91.064,
        'AC MILAN': 120.191,
        'OLYMPIACOS PIRAEUS': 51.415,
        'CHELSEA': 120.064,
        'FENERBAHCE': 44.656,
        'SEVILLA': 70.326
    },
    '2008-09': {
        'CHELSEA': 124.853,
        'JUVENTUS': 81.582,
        'VILLARREAL': 61.326,
        'PANATHINAIKOS': 48.632,
        'SPORTING CP': 56.176,
        'BAYERN MUNICH': 91.853,
        'ATLETICO MADRID': 39.853,
        'FC PORTO': 92.176,
        'LYON': 99.748,
        'BARCELONA': 117.853,
        'REAL MADRID': 118.853,
        'LIVERPOOL': 118.769,
        'ARSENAL': 91.853,
        'ROMA': 66.191,
        'INTER': 93.191,
        'MANCHESTER UTD': 106.853
    },
    '2009-10': {
        'VFB STUTTGART': 41.339,
        'BARCELONA': 118.853,
        'OLYMPIACOS PIRAEUS': 41.633,
        'BORDEAUX': 54.748,
        'INTER': 87.191,
        'CHELSEA': 124.853,
        'BAYERN MUNICH': 91.853,
        'FIORENTINA': 42.582,
        'CSKA MOSCOW': 56.525,
        'SEVILLA': 70.326,
        'LYON': 99.748,
        'REAL MADRID': 118.853,
        'FC PORTO': 92.176,
        'ARSENAL': 91.853,
        'AC MILAN': 99.191,
        'MANCHESTER UTD': 106.853
    },
    '2010-11': {
        'ROMA': 72.191,
        'SHAKHTAR DONETSK': 63.910,
        'TOTTENHAM': 38.371,
        'AC MILAN': 99.191,
        'VALENCIA': 79.326,
        'FC SCHALKE 04': 66.078,
        'INTER': 99.191,
        'BAYERN MUNICH': 91.853,
        'LYON': 99.748,
        'REAL MADRID': 112.853,
        'ARSENAL': 91.853,
        'BARCELONA': 118.853,
        'CHELSEA': 124.853,
        'MARSEILLE': 54.748,
        'FC COPENHAGEN': 15.860,
        'MANCHESTER UTD': 106.853
    },
    '2011-12': {
        'NAPOLI': 20.996,
        'CHELSEA': 93.882,
        'CSKA MOSCOW': 79.941,
        'REAL MADRID': 121.882,
        'MARSEILLE': 61.748,
        'BAYERN MUNICH': 98.853,
        'FC BASEL 1893': 54.980,
        'LYON': 95.748,
        'BAYER LEVERKUSEN': 47.296,
        'BARCELONA': 126.882,
        'ZENIT SAINT PETERSBURG': 73.941,
        'BENFICA': 61.522,
        'APOEL': 12.333,
        'AC MILAN': 99.191,
        'ARSENAL': 86.882,
        'INTER': 96.191
    },
    '2012-13': {
        'GALATASARAY': 46.400,
        'FC SCHALKE 04': 95.078,
        'CELTIC': 34.728,
        'JUVENTUS': 63.996,
        'ARSENAL': 83.882,
        'BAYERN MUNICH': 103.853,
        'SHAKHTAR DONETSK': 63.910,
        'BORUSSIA DORTMUND': 61.922,
        'AC MILAN': 89.191,
        'BARCELONA': 135.882,
        'PARIS SAINT GERMAIN': 45.835,
        'VALENCIA': 89.326,
        'FC PORTO': 98.176,
        'MALAGA': 16.835,
        'REAL MADRID': 121.882,
        'MANCHESTER UTD': 137.853
    },
    '2013-14': {
        'MANCHESTER CITY': 63.592,
        'BARCELONA': 157.882,
        'OLYMPIACOS PIRAEUS': 61.720,
        'MANCHESTER UTD': 137.853,
        'ZENIT SAINT PETERSBURG': 99.941,
        'BORUSSIA DORTMUND': 61.922,
        'ARSENAL': 113.882,
        'BAYERN MUNICH': 154.883,
        'AC MILAN': 89.191,
        'ATLETICO MADRID': 61.542,
        'PARIS SAINT GERMAIN': 72.835,
        'BAYER LEVERKUSEN': 61.296,
        'REAL MADRID': 136.882,
        'CHELSEA': 140.882,
        'GALATASARAY': 55.400,
        'FC SCHALKE 04': 95.078
    },
    '2014-15': {
        'PARIS SAINT GERMAIN': 98.300,
        'CHELSEA': 157.078,
        'MANCHESTER CITY': 85.592,
        'BARCELONA': 157.885,
        'BAYER LEVERKUSEN': 87.295,
        'ATLETICO MADRID': 119.542,
        'JUVENTUS': 95.996,
        'BORUSSIA DORTMUND': 72.922,
        'FC SCHALKE 04': 95.078,
        'REAL MADRID': 171.885,
        'SHAKHTAR DONETSK': 86.910,
        'BAYERN MUNICH': 154.883,
        'ARSENAL': 113.882,
        'MONACO': 36.300,
        'FC BASEL 1893': 66.875,
        'FC PORTO': 111.176
    },
    '2015-16': {
        'PARIS SAINT GERMAIN': 112.283,
        'CHELSEA': 152.078,
        'BENFICA': 100.276,
        'ZENIT SAINT PETERSBURG': 99.106,
        'GENT': 13.440,
        'VFL WOLFSBURG': 31.035,
        'REAL MADRID': 176.885,
        'ROMA': 54.191,
        'DYNAMO KYIV': 67.976,
        'MANCHESTER CITY': 99.592,
        'ATLETICO MADRID': 115.542,
        'BAYERN MUNICH': 167.883,
        'ARSENAL': 110.882,
        'JUVENTUS': 107.996,
        'PSV': 57.945,
        'BARCELONA': 159.885
    },
    '2016-17': {
        'MANCHESTER CITY': 100.592,
        'MONACO': 62.333,
        'REAL MADRID': 176.885,
        'NAPOLI': 78.087,
        'BAYERN MUNICH': 151.883,
        'ARSENAL': 110.882,
        'BENFICA': 111.276,
        'BORUSSIA DORTMUND': 99.922,
        'PARIS SAINT GERMAIN': 126.283,
        'BARCELONA': 151.885,
        'JUVENTUS': 140.996,
        'FC PORTO': 111.176,
        'BAYER LEVERKUSEN': 80.035,
        'LEICESTER': 15.256,
        'SEVILLA': 112.542,
        'ATLETICO MADRID': 142.542
    },
    '2017-18': {
        'MANCHESTER UTD': 89.000,
        'FC BASEL 1893': 74.000,
        'PARIS SAINT GERMAIN': 134.000,
        'REAL MADRID': 162.000,
        'LIVERPOOL': 62.000,
        'FC PORTO': 98.000,
        'JUVENTUS': 120.000,
        'TOTTENHAM': 67.000,
        'MANCHESTER CITY': 100.000,
        'SHAKHTAR DONETSK': 87.000,
        'CHELSEA': 106.000,
        'BARCELONA': 132.000,
        'BAYERN MUNICH': 135.000,
        'BESIKTAS': 57.000,
        'ROMA': 53.000,
        'SEVILLA': 109.000
    },
    '2018-19': {
        'MANCHESTER UTD': 82.000,
        'PARIS SAINT GERMAIN': 103.000,
        'FC PORTO': 93.000,
        'ROMA': 64.000,
        'TOTTENHAM': 67.000,
        'BORUSSIA DORTMUND': 89.000,
        'AJAX': 67.500,
        'REAL MADRID': 155.000,
        'JUVENTUS': 120.000,
        'ATLETICO MADRID': 146.000,
        'FC SCHALKE 04': 62.000,
        'MANCHESTER CITY': 100.000,
        'LYON': 61.500,
        'BARCELONA': 132.000,
        'BAYERN MUNICH': 135.000,
        'LIVERPOOL': 91.000
    },
    '2019-20': {
        'ATALANTA': 14.945,
        'ATLETICO MADRID': 127.000,
        'BARCELONA': 124.000,
        'BAYERN MUNICH': 136.000,
        'CHELSEA': 87.000,
        'JUVENTUS': 117.000,
        'RB LEIPZIG': 49.000,
        'LIVERPOOL': 99.000,
        'LYON': 61.000,
        'MANCHESTER CITY': 112.000,
        'NAPOLI': 71.000,
        'PARIS SAINT GERMAIN': 113.000,
        'REAL MADRID': 134.000,
        'VALENCIA': 37.000,
        'BORUSSIA DORTMUND': 85.000,
        'TOTTENHAM': 85.000
    },
    '2020-21': {
        'ATALANTA': 33.000,
        'ATLETICO MADRID': 136.000,
        'BARCELONA': 123.000,
        'BAYERN MUNICH': 136.000,
        'CHELSEA': 91.000,
        'JUVENTUS': 117.000,
        'LAZIO': 41.000,
        'RB LEIPZIG': 53.000,
        'LIVERPOOL': 120.000,
        'MANCHESTER CITY': 125.000,
        'BORUSSIA MONCHENGLADBACH': 26.000,
        'PARIS SAINT GERMAIN': 113.000,
        'FC PORTO': 87.000,
        'REAL MADRID': 134.000,
        'SEVILLA': 102.000,
        'BORUSSIA DORTMUND': 85.000
    },
    '2021-22': {
        'ATLETICO MADRID': 115.000,
        'BAYERN MUNICH': 134.000,
        'BENFICA': 58.000,
        'CHELSEA': 123.000,
        'INTER': 67.000,
        'JUVENTUS': 120.000,
        'LIVERPOOL': 119.000,
        'MANCHESTER CITY': 125.000,
        'MANCHESTER UTD': 113.000,
        'PARIS SAINT GERMAIN': 112.000,
        'REAL MADRID': 127.000,
        'RED BULL SALZBURG': 59.000,
        'SPORTING CP': 45.500,
        'VILLARREAL': 78.000,
        'AJAX': 82.500,
        'LILLE': 32.000
    },
    '2022-23': {
        'BAYERN MUNICH': 136.000,
        'BENFICA': 61.000,
        'CHELSEA': 126.000,
        'CLUB BRUGGE': 38.000,
        'BORUSSIA DORTMUND': 78.000,
        'EINTRACHT FRANKFURT': 61.000,
        'INTER': 67.000,
        'LIVERPOOL': 123.000,
        'MANCHESTER CITY': 134.000,
        'AC MILAN': 38.000,
        'NAPOLI': 81.000,
        'PARIS SAINT GERMAIN': 112.000,
        'FC PORTO': 80.000,
        'RB LEIPZIG': 72.000,
        'REAL MADRID': 121.000,
        'TOTTENHAM': 83.000
    },
    '2023-24': {
        'FC PORTO': 81.000,
        'ARSENAL': 72.000,
        'NAPOLI': 81.000,
        'BARCELONA': 98.000,
        'PARIS SAINT GERMAIN': 112.000,
        'REAL SOCIEDAD': 52.000,
        'INTER': 89.000,
        'ATLETICO MADRID': 104.000,
        'PSV': 52.000,
        'BORUSSIA DORTMUND': 89.000,
        'LAZIO': 49.000,
        'BAYERN MUNICH': 136.000,
        'FC COPENHAGEN': 28.000,
        'MANCHESTER CITY': 145.000,
        'RB LEIPZIG': 98.000,
        'REAL MADRID': 121.000
    },
    '2024-25': {
        'PARIS SAINT GERMAIN': 113.000,
        'LIVERPOOL': 123.000,
        'CLUB BRUGGE': 39.000,
        'ASTON VILLA': 20.000,
        'REAL MADRID': 128.000,
        'ATLETICO MADRID': 104.000,
        'PSV': 52.000,
        'ARSENAL': 72.000,
        'BENFICA': 61.000,
        'BARCELONA': 98.000,
        'BORUSSIA DORTMUND': 89.000,
        'LILLE': 36.000,
        'BAYERN MUNICH': 136.000,
        'BAYER LEVERKUSEN': 80.000,
        'FEYENOORD': 51.000,
        'INTER': 89.000
    }
}




# Compute prestige history with decay
prestige_history = {}
seasons_sorted = sorted(train_data.keys())

# for season in seasons_sorted:
#     season_start = int(season[:4])
#     final_teams = []
#     if "final" in train_data[season]:
#         final_teams = [team_map.get(train_data[season]["final"][0]["team_1"], train_data[season]["final"][0]["team_1"].upper()),
#                        team_map.get(train_data[season]["final"][0]["team_2"], train_data[season]["final"][0]["team_2"].upper())]
#     prestige_history[season] = {}
#     all_teams = set(base_prestige_map.keys()).union(final_teams)
#     for team in all_teams:
#         prestige = base_prestige_map.get(team, 1)
#         appeared_last_year = False
#         appeared_2_years_ago = False
#         for i, bonus in zip([1, 2], [2, 1]):
#             prev_season = f"{season_start - i}-{str(season_start - i + 1)[2:]}"
#             if prev_season in prestige_history and team in prestige_history[prev_season].get("final_teams", []):
#                 if i == 1:
#                     appeared_last_year = True
#                 elif i == 2:
#                     appeared_2_years_ago = True
#         if appeared_last_year:
#             prestige += 2
#         elif appeared_2_years_ago:
#             prestige += 1
#         prestige_history[season][team] = prestige
#     prestige_history[season]["final_teams"] = final_teams
finals = {
    "2002-03": ["AC MILAN", "JUVENTUS"],
    "2003-04": ["PORTO", "MONACO"],
    "2004-05": ["LIVERPOOL", "AC MILAN"],
    "2005-06": ["BARCELONA", "ARSENAL"],
    "2006-07": ["AC MILAN", "LIVERPOOL"],
    "2007-08": ["MANCHESTER UTD", "CHELSEA"],
    "2008-09": ["BARCELONA", "MANCHESTER UTD"],
    "2009-10": ["INTER", "BAYERN MUNICH"],
    "2010-11": ["BARCELONA", "MANCHESTER UTD"],
    "2011-12": ["CHELSEA", "BAYERN MUNICH"],
    "2012-13": ["BAYERN MUNICH", "BORUSSIA DORTMUND"],
    "2013-14": ["REAL MADRID", "ATLÉTICO MADRID"],
    "2014-15": ["BARCELONA", "JUVENTUS"],
    "2015-16": ["REAL MADRID", "ATLÉTICO MADRID"],
    "2016-17": ["REAL MADRID", "JUVENTUS"],
    "2017-18": ["REAL MADRID", "LIVERPOOL"],
    "2018-19": ["LIVERPOOL", "TOTTENHAM"],
    "2019-20": ["BAYERN MUNICH", "PARIS SAINT GERMAIN"],
    "2020-21": ["CHELSEA", "MANCHESTER CITY"],
    "2021-22": ["REAL MADRID", "LIVERPOOL"],
    "2022-23": ["MANCHESTER CITY", "INTER"],
    "2023-24": ["REAL MADRID", "BORUSSIA DORTMUND"],
    "2024-25": ["PARIS SAINT GERMAIN", "INTER"]
}




# Helper

def map_team(name):
    return team_map.get(name, name.upper())

samples = []

for season, rounds in train_data.items():
    index=list(finals).index(season)
    prev=list(finals)[index-1]
    prev2=list(finals)[index-2]
    # print(prev)
    # print(finals[prev])
    for round_name, matches in rounds.items():
        for match in matches:
            team_1 = map_team(match["team_1"])
            # print(f"{team_1} and {team_1 in t5_teams}")
            team_2 = map_team(match["team_2"])
            date = datetime.strptime(match["date"], "%d/%m/%Y")
            winner = map_team(match["winner"])

            row1 = features_df[(features_df["Team_Name"] == team_1) & (features_df["Match_Date"] < date)]
            row2 = features_df[(features_df["Team_Name"] == team_2) & (features_df["Match_Date"] < date)]
            if row1.empty or row2.empty:
                continue

            row1 = row1.sort_values("Match_Date").iloc[-1]
            row2 = row2.sort_values("Match_Date").iloc[-1]

            p1 = base_prestige_map.get(team_1, 0)
            if(team_1 in finals[prev]):
              p1+=5
            elif(team_1 in finals[prev2]):
              p1+=3
            p2 = base_prestige_map.get(team_2, 0)
            if(team_2 in finals[prev]):
              p2+=5
            elif(team_2 in finals[prev2]):
              p2+=3
            scale1 = 0.5
            scale2 = 0.5
            if team_1 in t5_teams:
                scale1 = 1.0
            if team_2 in t5_teams:
                scale2 = 1.0
            # if(team_1 in finals[prev]):
            #   print(f"{team_1} and {prev}")
            # (Pseudo-code)
            # h2h1 = features_df[(features_df['Team_Name'] == team_1) & (features_df['Opponent_Name'] == team_2)]
            # h2h_wins_1 = h2h1[h2h1['Team_Points'] == 3].shape[0]
            # h2h2 = features_df[(features_df['Team_Name'] == team_2) & (features_df['Opponent_Name'] == team_1)]
            # h2h_wins_2 = h2h2[h2h2['Team_Points'] == 3].shape[0]
            features = {
                # "team1_goals_scored": (float(row1["Team_Goals_Scored"]))*scale1,
                # "team1_goals_conceded": (float(row1["Team_Goals_Conceded"]))*scale1,
                # "team1_points": (float(row1["Team_Points"]))*scale1,
                # "team1_form": (float((row1["Recent_Form"])*(team_1 in t5_teams)))*scale1,
                "team1_is_top5": int((team_1 in t5_teams)),
                "u1":float(uefa_coeff[season][team_1]),
                "team1_prestige": int(p1),
                # "prev11":int(team_1 in finals[prev]),
                # "prev12":int(team_1 in finals[prev2]),
                # "team2_goals_scored": (float(row2["Team_Goals_Scored"]))*scale2,
                # "team2_goals_conceded": (float(row2["Team_Goals_Conceded"]))*scale2,
                # "team2_points": (float(row2["Team_Points"]))*scale2,
                # "team2_form": (float(row2["Recent_Form"]))*scale2,
                "team2_is_top5": int((team_2 in t5_teams)),
                "u2":float(uefa_coeff[season][team_2]),
                "team2_prestige": int(p2),
                # "prev21":int(team_2 in finals[prev]),
                # "prev22":int(team_2 in finals[prev2]),
                "t5 diff": int(team_1 in t5_teams) - int(team_2 in t5_teams),
                # "goal_diff": float((row1["Team_Goals_Scored"]-row1["Team_Goals_Conceded"])*scale1 - (row2["Team_Goals_Scored"]-row2["Team_Goals_Conceded"])*scale2),
                # "form_diff": float(row1["Recent_Form"]*scale1 - row2["Recent_Form"]*scale2),
                "prestige_diff": int((p1 - p2)),
                "team1_ucl_form": float(row1.get("ucl_form", 0.0)),
                "team2_ucl_form": float(row2.get("ucl_form", 0.0)),
                "ucl_form_diff": float(row1.get("ucl_form", 0.0) - row2.get("ucl_form", 0.0)),
                "uefa_coeff_diff":float(uefa_coeff[season][team_1] - uefa_coeff[season][team_2]),
                # "team1_h2h_wins": h2h_wins_1,
                # "team2_h2h_wins": h2h_wins_2,
                # "h2h_wins_diff": h2h_wins_1 - h2h_wins_2,
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
xgb.fit(X_train, y_train)
rf.fit(X_train, y_train)
gb.fit(X_train, y_train)
# ensemble_model.get_booster().save_model("xgb_model.json")
pickle.dump(ensemble_model, open("en_model_3.pkl", "wb"))
pickle.dump(xgb, open("xgb_model_3.pkl", "wb"))
pickle.dump(rf, open("rf_model_3.pkl", "wb"))
pickle.dump(gb, open("gb_model_3.pkl", "wb"))
print(f" Training Accuracy: {accuracy_score(y_test, ensemble_model.predict(X_test)):.2%}")
print(f" XGB Accuracy: {accuracy_score(y_test, xgb.predict(X_test)):.2%}")
print(f" RF Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.2%}")
print(f" GB Accuracy: {accuracy_score(y_test, gb.predict(X_test)):.2%}")
import pandas as pd
import json
from datetime import datetime
import pickle

# Load model and data
with open("en_model_3.pkl", "rb") as f:
    model = pickle.load(f)

features_df = pd.read_csv("Team_Match_Features_Final.csv", parse_dates=["Match_Date"])
mapping_df = pd.read_csv("auto_team_name_mapping.csv")
with open("test_matchups.json") as f:
    test_data = json.load(f)

team_map = dict(zip(mapping_df["train_team"], mapping_df["Mapped_Team_Name"]))
features_df["Team_Name"] = features_df["Team_Name"].str.upper()

t5_teams = set([
    "MANCHESTER CITY", "MANCHESTER UTD", "CHELSEA", "ARSENAL", "LIVERPOOL", "TOTTENHAM", "NEWCASTLE UNITED",
    "REAL MADRID", "BARCELONA", "ATLETICO MADRID", "SEVILLA", "VALENCIA", "VILLARREAL", "ATHLETIC BILBAO",
    "BAYERN MUNICH", "BORUSSIA DORTMUND", "RB LEIPZIG", "BAYER LEVERKUSEN", "FC SCHALKE 04", "VLF WOLFSBURG", "EINTRACHT FRANKFURT",
    "JUVENTUS", "INTER", "MILAN", "NAPOLI", "ROMA", "ATALANTA", "LAZIO",
    "PARIS SAINT GERMAIN", "MONACO", "LILLE", "LYON", "MARSEILLE"
])
base_prestige_map = {
    "REAL MADRID": 10, "BARCELONA": 6, "BAYERN MUNICH": 7, "JUVENTUS": 4,
    "MANCHESTER CITY": 7, "LIVERPOOL": 7, "CHELSEA": 7, "PARIS SAINT GERMAIN": 6,
    "ATLETICO MADRID": 6, "AC MILAN": 5, "INTER": 7, "MANCHESTER UTD": 5,
    "ARSENAL": 5, "BORUSSIA DORTMUND": 5, "PORTO": 2, "BENFICA": 2, "ROMA": 3,
    "NAPOLI": 5, "SEVILLA": 1, "TOTTENHAM": 2, "MONACO": 1, "LYON": 1,
    "VILLARREAL": 1, "CELTIC": 1, "RANGERS": 1,"RB LEIPZIG":0,"REAL SOCIEDAD":0,"CLUB BRUGGE":0
}

def map_team(name):
    return team_map.get(name, name.upper())
features_df.fillna(0, inplace=True)
def get_latest_features(team_name, date):
    df = features_df[(features_df["Team_Name"] == team_name.upper()) & (features_df["Match_Date"] < date)]
    if df.empty:
        return None
    row = df.sort_values("Match_Date").iloc[-1]
    # if team_name not in t5_teams:
    #     scale = 0.7
    #     row["Team_Goals_Scored"] *= scale
    #     row["Team_Goals_Conceded"] *= scale
    #     row["Team_Points"] *= scale
    #     row["Recent_Form"] *= scale
    return row

def predict_match(team1, team2, date,prev,prev2):
    t1 = map_team(team1)
    t2 = map_team(team2)
    season=str(date.year-1)+"-"+str(date.year)[2:]
    # print(f"{t1} and {t1 in t5_teams}")
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
    p1 = base_prestige_map.get(t1, 0)
    if(t1 in finals[prev]):
      p1+=5
    elif(t1 in finals[prev2]):
      p1+=3
    p2 = base_prestige_map.get(t2, 0)
    if(t2 in finals[prev]):
      p2+=5
    elif(t2 in finals[prev2]):
      p2+=3
    s1=0.5
    s2=0.5
    if(t1 in t5_teams):
      s1=1.0
    if(t2 in t5_teams):
      s2=1.0
    # if(t1 in finals[prev]):
    #   print(f"{t1} and {prev}")
    # h2h1 = features_df[(features_df['Team_Name'] == team_1) & (features_df['Opponent_Name'] == team_2)]
    # h2h_wins_1 = h2h1[h2h1['Team_Points'] == 3].shape[0]
    # h2h2 = features_df[(features_df['Team_Name'] == team_2) & (features_df['Opponent_Name'] == team_1)]
    # h2h_wins_2 = h2h2[h2h2['Team_Points'] == 3].shape[0]
    features = pd.DataFrame([{
        # "team1_goals_scored": float(row1["Team_Goals_Scored"])*s1,
        # "team1_goals_conceded": float(row1["Team_Goals_Conceded"])*s1,
        # "team1_points": float(row1["Team_Points"])*s1,
        # "team1_form": float((row1["Recent_Form"]))*s1,
        "team1_is_top5": int((t1 in t5_teams)),
        "u1":float(uefa_coeff[season][t1]),
        "team1_prestige": p1,
        # "prev11":int(t1 in finals[prev]),
        # "prev12":int(t1 in finals[prev2]),
        # "team2_goals_scored": float(row2["Team_Goals_Scored"])*s2,
        # "team2_goals_conceded": float(row2["Team_Goals_Conceded"])*s2,
        # "team2_points": float(row2["Team_Points"])*s2,
        # "team2_form": float((row2["Recent_Form"]))*s2,
        "team2_is_top5": int((t2 in t5_teams)),
        "u2":float(uefa_coeff[season][t2]),
        "team2_prestige": p2,
        # "prev21":int(t2 in finals[prev]),
        # "prev22":int(t2 in finals[prev2]),
        "t5 diff": int(t1 in t5_teams) - int(t2 in t5_teams),
        # "goal_diff": float((row1["Team_Goals_Scored"]-row1["Team_Goals_Conceded"])*s1 - (row2["Team_Goals_Scored"]-row2["Team_Goals_Conceded"])*s2),
        # "form_diff": float(row1["Recent_Form"]*s1- row2["Recent_Form"]*s2),
        "prestige_diff": int((p1 - p2)),
        "team1_ucl_form": float(row1.get("ucl_form", 0.0)),
        "team2_ucl_form": float(row2.get("ucl_form", 0.0)),
        "ucl_form_diff": float(row1.get("ucl_form", 0.0) - row2.get("ucl_form", 0.0)),
        "uefa_coeff_diff":float(uefa_coeff[season][t1] - uefa_coeff[season][t2]),
        # "team1_h2h_wins": h2h_wins_1,
        # "team2_h2h_wins": h2h_wins_2,
        # "h2h_wins_diff": h2h_wins_1 - h2h_wins_2,

    }])
    if features.isnull().any(axis=None):
        print(f"NaN in feature vector for {team1} vs {team2} on {date}")
        print(features)
        return team1
    pred = model.predict(features)[0]
    return team1 if pred == 1 else team2


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
    index=list(finals).index(season)
    prev=list(finals)[index-1]
    prev2=list(finals)[index-2]
    # --- Round of 16 ---
    r16_matches = rounds["round_of_16_matchups"]
    r16_results = []
    for match in r16_matches:
        t1, t2 = match["team_1"], match["team_2"]
        # print(f"{t1} and {t1 in t5_teams}")
        date = datetime.strptime(match["date"], "%Y-%m-%d")
        winner = predict_match(t1, t2, date,prev,prev2)
        r16_results.append({"team_1": t1, "team_2": t2, "winner": winner})

    # --- Quarterfinals ---
    qf_matches = rounds["quarter_finals_matchups"]
    qf_results = []
    for match in qf_matches:
        t1 = resolve_placeholder(match["team_1"], r16_results)
        t2 = resolve_placeholder(match["team_2"], r16_results)
        date = datetime.strptime(match["date"], "%Y-%m-%d")
        winner = predict_match(t1, t2, date,prev,prev2)
        qf_results.append({"team_1": t1, "team_2": t2, "winner": winner})

    # --- Semifinals ---
    sf_matches = rounds["semi_finals_matchups"]
    sf_results = []
    for match in sf_matches:
        t1 = resolve_placeholder(match["team_1"], r16_results, qf_results=qf_results)
        t2 = resolve_placeholder(match["team_2"], r16_results, qf_results=qf_results)
        date = datetime.strptime(match["date"], "%Y-%m-%d")
        winner = predict_match(t1, t2, date,prev,prev2)
        sf_results.append({"team_1": t1, "team_2": t2, "winner": winner})

    # --- Final ---
    final_match = rounds["final_matchup"]
    t1 = resolve_placeholder(final_match["team_1"], r16_results, qf_results=qf_results, sf_results=sf_results)
    t2 = resolve_placeholder(final_match["team_2"], r16_results, qf_results=qf_results, sf_results=sf_results)
    date = datetime.strptime(final_match["date"], "%Y-%m-%d")
    winner = predict_match(t1, t2, date,prev,prev2)
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
pd.DataFrame(submission_rows).to_csv("sample_submission_like_predictions_4.csv", index=False)
print("sample_submission_like_predictions.csv generated.")
 # type: ignore
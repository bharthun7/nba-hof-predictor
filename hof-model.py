import pandas as pd

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# get datasets for eligible and ineligible players from saved csv files
eligible = pd.read_csv("eligible_player_data.csv")
ineligible = pd.read_csv("ineligible_player_data.csv")

# minor corrections needed for players with incorrect HOF status
hof_additions = {
    "Dick Barnett",
    "Chauncey Billups",
    "Chris Bosh",
    "Carl Braun",
    "Kobe Bryant",
    "Vince Carter",
    "Chuck Cooper",
    "Michael Cooper",
    "Bob Dandridge",
    "Walter Davis",
    "Vlade Divac",
    "Tim Duncan",
    "Kevin Garnett",
    "Pau Gasol",
    "Manu Ginobili",
    "Tim Hardaway",
    "Lou Hudson",
    "Neil Johnston",
    "Bobby Jones",
    "Toni Kukoc",
    "Sidney Moncrief",
    "Dirk Nowitzki",
    "Tony Parker",
    "Paul Pierce",
    "Jack Sikma",
    "Dwayne Wade",
    "Ben Wallace",
    "Chris Webber",
    "Paul Westphal",
}
hof_removals = {"Reggie Hanson", "Nate Johnston", "Red Dehnert", "John Thompson"}
eligible.loc[eligible["full_name"].isin(hof_additions), "Hall of Fame Inductee"] = 1
eligible.loc[eligible["full_name"].isin(hof_removals), "Hall of Fame Inductee"] = 0

# fix Paul Millsap having two 2nd Team All-Rookie awards on nba.com
ineligible.loc[
    ineligible["full_name"] == "Paul Millsap", "2nd Team All-Rookie Team"
] = 1

# reorder columns in a more logical ordering than alphabetical order
stat_order = [
    "GP",
    "GS",
    "MIN",
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PTS",
    "MPG",
    "FGMPG",
    "FGAPG",
    "FG3MPG",
    "FG3APG",
    "FTMPG",
    "FTAPG",
    "ORPG",
    "DRPG",
    "RPG",
    "APG",
    "SPG",
    "BPG",
    "TPG",
    "FPG",
    "PPG",
]
stat_order += ["PF_" + i for i in stat_order]
eligible = pd.concat(
    [eligible.iloc[:, :5], eligible.loc[:, stat_order], eligible.iloc[:, 79:]], axis=1
)
ineligible = pd.concat(
    [ineligible.iloc[:, :5], ineligible.loc[:, stat_order], ineligible.iloc[:, 79:]],
    axis=1,
)

# drop all personal columns besides full name and any obsolete/irrelevant awards
drop_columns = [
    "id",
    "first_name",
    "last_name",
    "is_active",
    "IBM Award",
    "J. Walter Kennedy Citizenship",
    "NBA All-Star Selection",
    "NBA Comeback Player of the Year",
    "NBA Sporting News Most Valuable Player of the Year",
    "NBA Sporting News Rookie of the Year",
    "NBA Sportsmanship",
    "Olympic Appearance",
]
eligible = eligible.drop(drop_columns, axis=1)
ineligible = ineligible.drop(
    drop_columns
    + [
        "1st Team NBA Cup All-Tournament Team",
        "NBA Clutch Player of the Year",
        "NBA Cup Most Valuable Player",
        "NBA Defensive Player of the Month",
        "Hall of Fame Inductee",
    ],
    axis=1,
)
# move HOF to the end for easier model construction
eligible.insert(
    len(eligible.columns) - 1,
    "Hall of Fame Inductee",
    eligible.pop("Hall of Fame Inductee"),
)

# separate eligible players into train and test splits
train, test = train_test_split(eligible)


# combine this into function for easy model comparison
def train_run(pipe: Pipeline):
    pipe.fit(train.iloc[:, 1:97], train["Hall of Fame Inductee"])
    print(pipe.score(test.iloc[:, 1:97], test["Hall of Fame Inductee"]))
    with pd.option_context("display.max_rows", None):
        ineligible["HOF Probability"] = pipe.predict_proba(ineligible.iloc[:, 1:97])[
            :, 1
        ]
        print(
            ineligible.sort_values("HOF Probability", ascending=False).iloc[:50, :][
                ["full_name", "HOF Probability"]
            ]
        )


scalers = [("std", StandardScaler()), ("rbt", RobustScaler())]
models = [
    ("lr", LogisticRegression()),
    ("sgd", SGDClassifier(loss="log_loss")),
    ("rf", RandomForestClassifier()),
    ("gb", GradientBoostingClassifier()),
]
# create model pipeline with different preprocessing steps and model
pipe = Pipeline([])
mo = 0
for i in range(2):
    if i:
        pipe.steps.append(("pf", PolynomialFeatures()))
    for j in range(3):
        if j:
            pipe.steps.append(scalers[j - 1])
        for k in range(2):
            for model in models:
                if k:
                    pipe.steps.append(("rfe", RFE(model[1], n_features_to_select=30)))
                else:
                    pipe.steps.append(model)
                mo += 1
                if mo > 29 and not k:
                    print(f"MODEL {mo}: {pipe.steps}")
                    print("/" * 100)
                    train_run(pipe)
                pipe.steps.pop()
                print()
                print()
                print()
        if j:
            pipe.steps.pop()
    if i:
        pipe.steps.pop()

# train the model on eligible players and verify accuracy
# pipe.fit(train.iloc[:, 1:97], train["Hall of Fame Inductee"])
# print(pipe.score(test.iloc[:, 1:97], test["Hall of Fame Inductee"]))

# Display importance of features because I'm curious
# importance = pd.DataFrame(
#    {
#       "Feature": eligible.columns[1:97].to_list(),
#        "Coefficient": pipe.steps[1][1].feature_importances_,
#   }
# ).sort_values("Coefficient")

# with pd.option_context("display.max_rows", None):
# print(importance)
# use proba to get the probability of classifiction for inelg playere
# ineligible["HOF Probability"] = pipe.predict_proba(ineligible.iloc[:, 1:97])[
#    :, 1
# ].round(4)
# print the top 50
# print(
#    ineligible.sort_values("HOF Probability", ascending=False).iloc[:50, :][
#        ["full_name", "HOF Probability"]
#    ]
# )

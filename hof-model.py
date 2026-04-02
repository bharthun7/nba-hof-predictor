import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest

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
    "Michael Cooper",
    "Bob Dandridge",
    "Walter Davis",
    "Tim Duncan",
    "Kevin Garnett",
    "Pau Gasol",
    "Manu Ginobili",
    "Tim Hardaway",
    "Lou Hudson",
    "Neil Johnston",
    "Sidney Moncrief",
    "Dirk Nowitzki",
    "Tony Parker",
    "Paul Pierce",
    "Jack Sikma",
    "Dwyane Wade",
    "Ben Wallace",
    "Chris Webber",
    "Paul Westphal",
}
hof_removals = {
    "Reggie Hanson",
    "Nate Johnston",
    "Red Dehnert",
    "John Thompson",
    "Bill Bradley",
    "Al Cervi",
    "Bob Houbregs",
    "Buddy Jeannette",
    "Sarunas Marciulionis",
    "Drazen Petrovic",
    "Dino Radja",
    "Arvydas Sabonis",
}

eligible.loc[eligible["id"] == 77193, "Hall of Fame Inductee"] = 1
eligible.loc[eligible["full_name"].isin(hof_additions), "Hall of Fame Inductee"] = 1
eligible.loc[eligible["full_name"].isin(hof_removals), "Hall of Fame Inductee"] = 0

# fix Paul Millsap having two 2nd Team All-Rookie awards on nba.com
ineligible.loc[
    ineligible["full_name"] == "Paul Millsap", "2nd Team All-Rookie Team"
] = 1

# dictionary used to add in ABA awards for ABA HOFers
aba_awards = {
    "Rick Barry": {
        "1st Team All-NBA": 4,
        "NBA All-Star": 4,
    },
    "Zelmo Beaty": {
        "NBA Champion": 1,
        "NBA Finals Most Valuable Player": 1,
        "2nd Team All-NBA": 2,
        "NBA All-Star": 3,
    },
    "Billy Cunningham": {
        "NBA Most Valuable Player": 1,
        "1st Team All-NBA": 1,
        "NBA All-Star": 1,
    },
    "Lou Dampier": {
        "NBA Champion": 1,
        "2nd Team All-NBA": 4,
        "1st Team All-Rookie Team": 1,
        "NBA All-Star": 7,
    },
    "Mel Daniels": {
        "NBA Champion": 3,
        "NBA Most Valuable Player": 2,
        "NBA Rookie of the Year": 1,
        "NBA All-Star Most Valuable Player": 1,
        "1st Team All-NBA": 4,
        "2nd Team All-NBA": 1,
        "1st Team All-Rookie Team": 1,
        "NBA All-Star": 7,
    },
    "Julius Erving": {
        "NBA Champion": 2,
        "NBA Most Valuable Player": 3,
        "NBA Finals Most Valuable Player": 2,
        "1st Team All-NBA": 4,
        "2nd Team All-NBA": 1,
        "1st Team All-Defensive Team": 1,
        "1st Team All-Rookie Team": 1,
        "NBA All-Star": 5,
    },
    "George Gervin": {
        "2nd Team All-NBA": 2,
        "1st Team All-Rookie Team": 1,
        "NBA All-Star": 3,
    },
    "Artis Gilmore": {
        "NBA Champion": 1,
        "NBA Most Valuable Player": 1,
        "NBA Rookie of the Year": 1,
        "NBA Finals Most Valuable Player": 1,
        "NBA All-Star Most Valuable Player": 1,
        "1st Team All-NBA": 5,
        "1st Team All-Defensive Team": 4,
        "1st Team All-Rookie Team": 1,
        "NBA All-Star": 5,
    },
    "Cliff Hagan": {
        "NBA All-Star": 1,
    },
    "Connie Hawkins": {
        "NBA Champion": 1,
        "NBA Most Valuable Player": 1,
        "NBA Finals Most Valuable Player": 1,
        "1st Team All-NBA": 2,
        "NBA All-Star": 2,
    },
    "Spencer Haywood": {
        "NBA Most Valuable Player": 1,
        "NBA Rookie of the Year": 1,
        "NBA All-Star Most Valuable Player": 1,
        "1st Team All-NBA": 1,
        "1st Team All-Rookie Team": 1,
        "NBA All-Star": 1,
    },
    "Dan Issel": {
        "NBA Champion": 1,
        "NBA Rookie of the Year": 1,
        "NBA All-Star Most Valuable Player": 1,
        "1st Team All-NBA": 1,
        "2nd Team All-NBA": 4,
        "1st Team All-Rookie Team": 1,
        "NBA All-Star": 6,
    },
    "Gus Johnson": {
        "NBA Champion": 1,
    },
    "Moses Malone": {
        "1st Team All-Rookie Team": 1,
        "NBA All-Star": 1,
    },
    "George McGinnis": {
        "NBA Champion": 2,
        "NBA Most Valuable Player": 1,
        "NBA Finals Most Valuable Player": 1,
        "1st Team All-NBA": 2,
        "2nd Team All-NBA": 1,
        "1st Team All-Rookie Team": 1,
        "NBA All-Star": 3,
    },
    "Charlie Scott": {
        "NBA Rookie of the Year": 1,
        "1st Team All-NBA": 1,
        "2nd Team All-NBA": 1,
        "1st Team All-Rookie Team": 1,
        "NBA All-Star": 2,
    },
    "David Thompson": {
        "NBA Rookie of the Year": 1,
        "NBA All-Star Most Valuable Player": 1,
        "2nd Team All-NBA": 1,
        "1st Team All-Rookie Team": 1,
        "NBA All-Star": 1,
    },
}

# Bobby Jones has to be looked up by ID because of dupe names
bobby = {
    "2nd Team All-NBA": 1,
    "1st Team All-Defensive Team": 2,
    "1st Team All-Rookie Team": 1,
    "NBA All-Star": 1,
}

# actually adding in the awards
for player, awards in aba_awards.items():
    for award, num in awards.items():
        eligible.loc[eligible["full_name"] == player, award] += num
for award, num in bobby.items():
    eligible.loc[eligible["id"] == 77193, award] += num

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


def player_selection(prompt):
    player = input(f"{prompt}: ")
    while player not in ineligible["full_name"].to_list():
        split_names = ineligible["full_name"].str.lower().str.split(" ", expand=True)
        names = player.lower().split(" ")
        first_matches = ineligible[split_names[0] == names[0]]["full_name"].to_list()
        last_matches = []
        if len(names) > 1:
            last_matches = ineligible[split_names[1] == names[1]]["full_name"].to_list()
        if len(first_matches + last_matches) > 0:
            print()
            print("Not a valid player. Maybe you meant: ")
            print("\n".join(first_matches + last_matches))
            print()
            player = input("Please try again: ")
        else:
            player = input("Not a valid player. Please try again: ")
    return player


# separate eligible players into train and test splits
train, test = train_test_split(eligible)

n_features = 20
print("NBA HOF Prediction Model")
print("/" * 200)
while True:
    print("Please make a selection: ")
    print()
    print(f"1. Change number of selected features (currently {n_features}).")
    print("2. Edit an ineligible player's stats.")
    print("3. Run the model.")
    print("4. Quit.")
    print()
    choice = int(input("Selection: "))
    while choice not in range(1, 5):
        choice = int(input("Invalid selection. Please try again: "))
    if choice == 1:
        print()
        new_nf = int(input("Features to select: "))
        while new_nf <= 0:
            new_nf = int(input("Features must be > 0. Please try again: "))
        print()
        if new_nf > 96:
            n_features = 96
            print("Max features is 96. Number of features set to 96.")
        else:
            n_features = new_nf
            print(f"Number of features set to {n_features}.")
        print()
    elif choice == 2:
        print()
        player = player_selection("Player to modify")
        print()
        print("Modifiable Stats:")
        for num, stat in enumerate(ineligible.columns.to_list()[1:97]):
            print(f"{num+1}. {stat}")
        print()
        stat_choice = int(input("Stat to modify: "))
        while stat_choice < 1 or stat_choice > 96:
            stat_choice = int(input("Invalid choice. Please try again: "))
        stat_column = ineligible.columns.to_list()[stat_choice]
        print()
        new_val = int(
            input(
                f"New value for {player} {stat_column} (currently {ineligible[ineligible['full_name']==player][stat_column].iloc[0]}): "
            )
        )
        while new_val < 0:
            new_val = int(input("Value cannot be negative. Please try again: "))
        print()
        print(f"{player} {stat_column} set to {new_val}.")
        print()
        ineligible.loc[ineligible["full_name"] == player, stat_column] = new_val
    elif choice == 3:
        print()
        pipe = Pipeline(
            [
                ("std", StandardScaler()),
                ("kb", SelectKBest(k=n_features)),
                ("lr", LogisticRegression(C=0.01, solver="liblinear")),
            ]
        )
        pipe.fit(train.iloc[:, 1:-1], train["Hall of Fame Inductee"])
        ineligible["HOF Probability"] = pipe.predict_proba(ineligible.iloc[:, 1:97])[
            :, 1
        ]
        print("Finished fitting model. ", end="")
        while True:
            print("What would you like to do?")
            print()
            print("1. View model accuracy.")
            print("2. View feature coefficients.")
            print("3. View highest-probability players.")
            print("4. Lookup a player's probability.")
            print("5. Exit back to main menu.")
            print()
            choice = int(input("Selection: "))
            while choice not in range(1, 6):
                choice = int(input("Invalid selection. Please try again: "))
            if choice == 1:
                print()
                print(
                    f"Model Accuracy: {pipe.score(test.iloc[:, 1:-1], test["Hall of Fame Inductee"])*100:.2f}%"
                )
                print()
            elif choice == 2:
                importance = pd.DataFrame(
                    {
                        "Feature": eligible.columns[1:-1][
                            pipe.steps[1][1].get_support()
                        ].to_list(),
                        "Coefficient": pipe.steps[2][1].coef_[0],
                    }
                ).sort_values("Coefficient", key=abs, ascending=False)
                print()
                print(importance.to_string(index=False))
                print()
            elif choice == 3:
                print()
                n_players = int(input("Number of players to display: "))
                while n_players <= 0:
                    n_players = int(input("Players must be > 0. Please try again: "))
                print()
                if n_players > 997:
                    n_players = 997
                    print("Max players is 997. ", end="")
                print(
                    f"Displaying top {n_players} players with highest HOF-probability:"
                )
                print()
                print(
                    ineligible.sort_values("HOF Probability", ascending=False)
                    .iloc[:n_players, :][["full_name", "HOF Probability"]]
                    .to_string(
                        header=["Player", "HOF Probability"],
                        index=False,
                        float_format=lambda x: f"{x*100:.2f}%",
                    )
                )
                print()
            elif choice == 4:
                print()
                player = player_selection("Player to look up")
                print()
                print(
                    f"{player} has a {ineligible[ineligible['full_name']==player]["HOF Probability"].iloc[0]*100:.2f}% chance of making the HOF."
                )
                print()
            else:
                break
    else:
        quit()
    print("/" * 200)

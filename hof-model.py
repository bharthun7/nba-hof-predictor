import pandas as pd

#get datasets for eligible and ineligible players from saved csv files
eligible=pd.read_csv("eligible_player_data.csv")
ineligible=pd.read_csv("ineligible_player_data.csv")

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

eligible.loc[eligible["full_name"].isin(hof_additions),"Hall of Fame Inductee"]=1
eligible.loc[eligible["full_name"].isin(hof_removals),"Hall of Fame Inductee"]=0
print(eligible[eligible["full_name"]=="Red Dehnert"].iloc[0].to_dict())
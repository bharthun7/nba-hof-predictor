import pandas as pd
from time import sleep
from datetime import date

from nba_api.stats.static import players
from nba_api.stats.endpoints import playerawards, playercareerstats

# get our dataframes of inactive and active player
inactives = pd.DataFrame(players.get_inactive_players())
actives = pd.DataFrame(players.get_active_players())
# this list is used to keep track of players who are inactive, but not long enough to
# be HOF-eligible (4 years according to their website)
inactive_ineligibles = []

# we'll use the current data to find the current season
season = date.today().year
if date.today().month > 6:
    # this is needed for later months (i.e. October 2025 is part of 2025-26 season)
    season += 1


def rename_avgs(col: str) -> str:
    """
    This function renames statistical average columns to align with conventional
    abbreviations for average stats (PPG, RPG, etc)

    :param col: the column title to be named
    :type col: str
    :return: the renamed column title
    :rtype: str
    """

    if col == "PF":
        # fouls per game is the only one to use its 2nd character
        rename = "F"
    elif col == "OREB" or col == "DREB":
        # offensive and defensive rebounds must be distinguished from one another
        rename = col[:2]
    elif col[0] == "F":
        # these are all the shooting stats (FGM, FG3A, etc); they use the whole name
        rename = col
    else:
        # all other stats retain just their first letter
        rename = col[0]
    # PG is added to reflect the per-game nature of these columns
    return rename + "PG"


def clean_avgs(df: pd.DataFrame) -> pd.Series:
    """
    Renames and slices the DataFrame of a player's average stats to be concatenated
    with the rest of their stats and awards

    :param df: The DataFrame of a player's career average stats
    :type df: pd.DataFrame
    :return: A series of just the values for each stat for concatenation
    :rtype: Series[Any]
    """

    # the rename_avgs function from before is used as the mapper in Pandas's rename
    df.rename(columns=rename_avgs, inplace=True)
    # the 0 row index is used to get the values while the column indices are used to
    # remove items that are irrelevant (year, team) or redundant (shooting splits)
    return pd.concat(
        [df.iloc[0, 5:8], df.iloc[0, 9:11], df.iloc[0, 12:14], df.iloc[0, 15:]]
    )


def get_career_stats(row: pd.Series) -> pd.Series:
    """
    Uses nba_api to get the career totals and averages for each player for both the
    regular season and playoffs. Processes averages with aformentioned helper function
    and concatenates Series into a Series that is attached to main DataFrame

    :param row: The Series representing a player from the inactive or active Dataframes
    :type row: pd.Series
    :return: A formatted Series of the player's career totals and averages
    :rtype: Series[Any]
    """

    # calls to get the player's career totals and averages, sleeping to respect the
    # NBA's rate limiting
    sleep(0.5)
    totals = playercareerstats.PlayerCareerStats(row["id"]).get_data_frames()
    sleep(0.5)
    avgs = playercareerstats.PlayerCareerStats(
        row["id"], per_mode36="PerGame"
    ).get_data_frames()

    # for inactive players, check their last season to determine eligibility
    if (
        row["is_active"] == False
        and season - (int(totals[0].iloc[-1]["SEASON_ID"][:4]) + 1) <= 4
    ):
        inactive_ineligibles.append[row["full_name"]]

    # if a player has never played a playoff game, only return their regular season
    # totals and averages (index 1 in both lists)
    if len(totals[3]) == 0:
        return pd.concat([totals[1].iloc[0, 3:], clean_avgs(avgs[1])])
    # otherwise, add in the playoff DataFrames at index 3; iloc and slicing are used
    # near identical to clean_avgs, but with shooting splits and games played included
    return pd.concat(
        [
            totals[1].iloc[0, 3:],
            clean_avgs(avgs[1]),
            totals[3].iloc[0, 3:].add_prefix("PF_"),
            clean_avgs(avgs[3]).add_prefix("PF_"),
        ]
    )


def get_awards(row: pd.Series) -> pd.Series:
    """
    Uses nba_api to get a list of awards a player has won and converts that into a
    Series of number of wins for each award to be attached to the main DataFrame

    :param row: The Series representing a player from the inactive or active DataFrames
    :type row: pd.Series
    :return: A series of the number of times a player has won each award
    :rtype: Series[Any]
    """
    
    # call to get list of player's awards, sleeping to respect rate-limiting
    sleep(0.5)
    awards = playerawards.PlayerAwards(row["id"]).get_data_frames()[0]

    # dictionary used to map All-NBA team numbers to distinguish between each team
    team_nums = {"1": "1st", "2": "2nd", "3": "3rd"}
    awards.loc[
        awards["ALL_NBA_TEAM_NUMBER"].fillna("").str.isnumeric(), "DESCRIPTION"
    ] = (
        awards["ALL_NBA_TEAM_NUMBER"].map(team_nums) + " Team " + awards["DESCRIPTION"]
    )

    # Hall of Fame Inductee is a listed award, so HOF status will be numeric for now
    return awards.groupby("DESCRIPTION").size()

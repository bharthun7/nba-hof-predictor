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

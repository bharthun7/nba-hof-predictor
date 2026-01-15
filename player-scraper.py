import pandas as pd
from time import sleep
from datetime import date
from io import StringIO

from nba_api.stats.static import players
from nba_api.stats.endpoints import playerawards, playercareerstats

from selenium import webdriver
from selenium.webdriver.common.by import By

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


def insert_missing(stats: pd.Series) -> pd.Series:
    """
    Modifies a player's career statline scraped from Basketball Reference if it's
    missing shooting percentage columns due to having 0 attempts

    :param stats: The Series representing the career statline for a player
    :type stats: pd.Series
    :return: The modified Series with any missing shooting percentage columns inserted
    :rtype: Series[Any]
    """

    # if 0 FGs were attempted, FG% will be missing along with 2P%, 3P%, and EFG%
    if "FG%" not in stats:
        stats = pd.concat(
            [
                stats[:10],
                pd.Series({"FG%": 0.0}),
                stats[10:12],
                pd.Series({"3P%": 0.0}),
                stats[12:14],
                pd.Series({"2P%": 0.0, "eFG%": 0.0}),
                stats[14:],
            ]
        )
    # otherwise, check 3P% and 2P% since player may have only attempted one kind
    elif "3P%" not in stats:
        stats = pd.concat([stats[:13], pd.Series({"3P%": 0.0}), stats[13:]])
    elif "2P%" not in stats:
        stats = pd.concat([stats[:16], pd.Series({"2P%": 0.0}), stats[16:]])

    # FT attempts is independent of FG attempts but follows same logic
    if "FT%" not in stats:
        stats = pd.concat([stats[:20], pd.Series({"FT%": 0.0}), stats[20:]])

    return stats


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

    # for debugging purposes
    print(row["full_name"])
    # calls to get the player's career totals and averages, sleeping to respect the
    # NBA's rate limiting
    sleep(0.5)
    try:
        totals = playercareerstats.PlayerCareerStats(row["id"]).get_data_frames()
    except KeyError:
        # a KeyError will occur if the player's page on nba.com is empty. In this case,
        # Selenium is needed to manually scrape Basketball Reference to get their
        # career stats
        driver = webdriver.Firefox()

        name_parts = row["full_name"].lower().replace("-", "").split(" ")
        # BR uses first 5 of last name, first 2 of first name for the url, no hyphens
        driver.get(
            f"https://www.basketball-reference.com/players/{name_parts[1][0]}/{name_parts[1][:5]}{name_parts[0][:2]}01.html"
        )

        # use IDs to get tables for career totals and averages, playoffs not needed
        totals_table = driver.find_element(By.ID, "totals_stats")
        avgs_table = driver.find_element(By.ID, "per_game_stats")
        # then convert said tables to Series for processing
        totals = pd.read_html(StringIO(totals_table.get_attribute("outerHTML")))[
            0
        ].iloc[1]
        avgs = pd.read_html(StringIO(avgs_table.get_attribute("outerHTML")))[0].iloc[1]
        driver.quit()

        # used for converting BR column names to NBA.com column names
        br_rename = {
            "G": "GP",
            "MP": "MIN",
            "FG": "FGM",
            "FG%": "FG_PCT",
            "3P": "FG3M",
            "3P%": "FG3_PCT",
            "3PA": "FG3A",
            "FT": "FTM",
            "FT%": "FT_PCT",
            "ORB": "OREB",
            "DRB": "DREB",
            "TRB": "REB",
        }
        # from here, insert_missing will be used to ensure all required columns are
        # present, and columns will be renamed to align with the rest of the DataFrame
        totals = insert_missing(totals).rename(br_rename)
        avgs = insert_missing(avgs).rename(br_rename).rename(rename_avgs)

        # concatenate the two together like usual, ignoring irrelevant columns
        return pd.concat(
            [
                totals[5:14],
                totals[18:-2],
                avgs[7:10],
                avgs[11:13],
                avgs[18:20],
                avgs[21:-1],
            ]
        )
    sleep(0.5)
    avgs = playercareerstats.PlayerCareerStats(
        row["id"], per_mode36="PerGame"
    ).get_data_frames()

    # for inactive players, check their last season to determine eligibility
    if (
        row["is_active"] == False
        and season - (int(totals[0].iloc[-1]["SEASON_ID"][:4]) + 1) <= 4
    ):
        inactive_ineligibles.append(row["full_name"])

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


# for each function, apply will create a DataFrame that can be concatenated row-wise
print("Begin scraping stats for inactive players...")
inactives = pd.concat([inactives, inactives.apply(get_career_stats, axis=1)], axis=1)
print("Finished scraping stats for inactive players, begin scraping awards...")
inactives = pd.concat([inactives, inactives.apply(get_awards, axis=1)], axis=1).fillna(
    0
)
print("Finished scraping awards, begin removing IIs and saving to csv file...")

# use the inactive_ineligible list to remove any of those players from the inactive df
inactive_ineligibles_df = inactives[inactives["full_name"].isin(inactive_ineligibles)]
# that df is then saved to a csv for easy access/to prevent repeated scraping
inactives.drop(inactive_ineligibles_df.index).to_csv("eligible_player_data.csv")

# now we'll repeat that whole process for active players
print("Finished saving inactives df, begin scraping stats for active players...")
actives = pd.concat([actives, actives.apply(get_career_stats, axis=1)], axis=1)
print("Finished scraping stats for active players, begin scraping awards...")
actives = pd.concat([actives, actives.apply(get_awards, axis=1)], axis=1).fillna(0)
print("Finished scraping awards, begin adding IIs and saving to csv file...")

pd.concat([actives, inactive_ineligibles_df]).to_csv("ineligible_player_data.csv")
print("Finished scraping!")

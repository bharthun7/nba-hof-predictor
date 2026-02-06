import pandas as pd
from time import sleep
from datetime import date
from io import StringIO
import requests
import pickle

from nba_api.stats.static import players
from nba_api.stats.endpoints import playerawards, playercareerstats

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options

options = Options()
options.add_argument("--headless")

# get our dataframes of inactive and active player
inactives = pd.DataFrame(players.get_inactive_players())
actives = pd.DataFrame(players.get_active_players())
# this set is used to keep track of players who are inactive, but not long enough to
# be HOF-eligible (4 years according to their website)
inactive_ineligibles = set()
# this one is used for G-League players who never played in the NBA but have a page
never_in_nba = set()

# minor corrections for players whose names are just wrong, which messes things up
inactives.loc[inactives["full_name"] == "Cui Cui", "full_name"] = "Cui Yongxi"
inactives.loc[inactives["full_name"] == "Ike Fontaine", "full_name"] = "Isaac Fontaine"
inactives.loc[inactives["full_name"] == "Ruben Garces", "full_name"] = "Rubén Garcés"
inactives.loc[inactives["full_name"] == "Vincent Hunter", "full_name"] = "Vince Hunter"
inactives.loc[inactives["full_name"] == "Ibrahim Kutluay", "full_name"] = "Ibo Kutluay"
inactives.loc[inactives["full_name"] == "Nicolas Laprovittola", "full_name"] = (
    "Nicolás Laprovíttola"
)
inactives.loc[inactives["full_name"] == "Karim Mane", "full_name"] = "Karim Mané"
inactives.loc[inactives["full_name"] == "Boniface Ndong", "full_name"] = (
    "Boniface N'Dong"
)
inactives.loc[inactives["full_name"] == "Zach Norvell Jr.", "full_name"] = (
    "Zach Norvell"
)
inactives.loc[inactives["full_name"] == "JJ O'Brien", "full_name"] = "J.J. O'Brien"
inactives.loc[inactives["full_name"] == "Maozinha Pereira", "full_name"] = (
    "Mãozinha Pereira"
)
inactives.loc[inactives["full_name"] == "Filip Petrusev", "full_name"] = (
    "Filip Petrušev"
)
inactives.loc[inactives["full_name"] == "Aleksandar Radojevic", "full_name"] = (
    "Aleksandar Radojević"
)
inactives.loc[inactives["full_name"] == "Trevon Scott", "full_name"] = "Tre Scott"
inactives.loc[inactives["full_name"] == "DJ Stephens", "full_name"] = "D.J. Stephens"
inactives.loc[inactives["full_name"] == "Slavko Vranes", "full_name"] = "Slavko Vraneš"
inactives.loc[inactives["full_name"] == "MJ Walker", "full_name"] = "M.J. Walker"
inactives.loc[inactives["full_name"] == "Matt Williams Jr.", "full_name"] = (
    "Matt Williams"
)

# we'll use the current data to find the current season
season = date.today().year
if date.today().month > 6:
    # this is needed for later months (i.e. October 2025 is part of 2025-26 season)
    season += 1

# for preventing timeouts
custom_headers = {
    "Host": "stats.nba.com",
    "Connection": "keep-alive",
    "Cache-Control": "max-age=0",
    "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:147.0) Gecko/20100101 Firefox/147.0",
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com",
}

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


def get_totals(row: pd.Series) -> pd.Series:
    """
    Uses nba_api to get the career totals for each player for both the regular
    season and playoffs. Scrapes Basketball Reference when needed and concatenates
    Series into a Series that is attached to main DataFrame

    :param row: The Series representing a player from the inactive or active Dataframes
    :type row: pd.Series
    :return: A formatted Series of the player's career totals
    :rtype: Series[Any]
    """

    # calls to get the player's career totals, sleeping to respect the
    # NBA's rate limiting
    sleep(10)
    try:
        totals = playercareerstats.PlayerCareerStats(
            row["id"], headers=custom_headers
        ).get_data_frames()
    except KeyError:
        # this and all similar print statements are for my debugging
        print(f"{row["full_name"]} scraped on BR")
        # a KeyError will occur if the player's page on nba.com is empty. In this case,
        # Selenium is needed to manually scrape Basketball Reference to get their
        # career stats
        driver = webdriver.Firefox(options=options)
        driver.install_addon("ublock_origin-1.68.0.xpi")

        # BR organizes players by first letter of last name, so find player in their
        # corresponding page
        driver.get(
            f"https://www.basketball-reference.com/players/{row['last_name'].lower()[0]}/"
        )
        player_link = driver.find_element(By.LINK_TEXT, row["full_name"]).get_attribute(
            "href"
        )
        driver.get(player_link)  # type: ignore

        # use ID to get table for career totals, playoffs not needed yet
        totals_table = driver.find_element(By.ID, "totals_stats")
        # then convert said table to a Series for processing
        totals = pd.read_html(StringIO(totals_table.get_attribute("outerHTML")))[0]

        # get last season for inactive-ineligible check, albeit slightly more complex
        last_season = (
            int(
                totals[totals.fillna("")["Season"].str.contains(r"\d{4}-\d{2}")].iloc[
                    -1
                ]["Season"][:4]
            )
            + 1
        )
        # the check for the difference not being 0 is used in place of is_active column
        if season - last_season != 0 and season - last_season <= 4:
            print("\tAlso inactive-ineligible")
            inactive_ineligibles.add(row["full_name"])

        # extract the career row by regex, as it position can vary if they player has
        # played for multiple teams in their career
        totals = totals[totals.fillna("")["Season"].str.contains(r"^\d Yrs?$")].iloc[0]

        # if a player has played in the playoffs, he'll have a playoff table as well
        pf_totals_table = driver.find_elements(By.ID, "totals_stats_post")
        has_pf = False
        if len(pf_totals_table) == 1:
            # if the table exists, a Series for playoff totals can be made
            pf_totals = pd.read_html(
                StringIO(pf_totals_table[0].get_attribute("outerHTML"))
            )[0]
            pf_totals = pf_totals[
                pf_totals.fillna("")["Season"].str.contains(r"^\d Yrs?$")
            ].iloc[0]
            # this is now set to true, so playoff Series can be concatenated later
            has_pf = True
        driver.quit()

        # from here, insert_missing will be used to ensure all required columns are
        # present, and columns will be renamed to align with the rest of the DataFrame
        totals = insert_missing(totals).rename(br_rename)

        # if the player has played in the playoffs, process their playoff Series also
        if has_pf:
            print("\tAlso played in playoffs")
            pf_totals = insert_missing(pf_totals).rename(br_rename).add_prefix("PF_")  # type: ignore
            # similar concatenation, just with playoffs included
            return pd.concat(
                [
                    totals[5:14],
                    totals[18:-2],
                    pf_totals[5:14],
                    pf_totals[18:-2],
                ]
            )

        # concatenate to ignore irrelevant columns and return that Series
        return pd.concat([totals[5:14], totals[18:-2]])
    except requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError:
        print(f"{row['full_name']} caused a timeout")
        quit()

    # if the player has no seasons, skip over them; they'll be removed later
    if len(totals[0]) == 0:
        print(f"{row['full_name']} never played in the NBA")
        never_in_nba.add(row["full_name"])
        return pd.Series()

    # for inactive players, check their last season to determine eligibility
    if (
        row["is_active"] == False
        and season - (int(totals[0].iloc[-1]["SEASON_ID"][:4]) + 1) <= 4
    ):
        print(f"{row['full_name']} is inactive-ineligible")
        inactive_ineligibles.add(row["full_name"])

    # if a player has never played a playoff game, only return their regular season
    # totals (index 1 in the list)
    if len(totals[3]) == 0:
        return totals[1].iloc[0, 3:]

    # otherwise, add in the playoff DataFrame at index 3; iloc and slicing are used
    # near identical to clean_avgs, but with shooting splits and games played included
    return pd.concat([totals[1].iloc[0, 3:], totals[3].iloc[0, 3:].add_prefix("PF_")])


def get_avgs(row: pd.Series) -> pd.Series:
    """
    Uses nba_api to get the career averages for each player for both the regular
    season and playoffs. Scrapes Basketball Reference when needed and concatenates
    Series into a Series that is attached to main DataFrame

    :param row: The Series representing a player from the inactive or active Dataframes
    :type row: pd.Series
    :return: A formatted Series of the player's career averages
    :rtype: Series[Any]
    """

    # calls to get the player's career averages, sleeping to respect the
    # NBA's rate limiting
    sleep(10)
    try:
        avgs = playercareerstats.PlayerCareerStats(
            row["id"], per_mode36="PerGame", headers=custom_headers
        ).get_data_frames()
    except KeyError:
        # this and all similar print statements are for my debugging
        print(f"{row["full_name"]} scraped on BR")
        # a KeyError will occur if the player's page on nba.com is empty. In this case,
        # Selenium is needed to manually scrape Basketball Reference to get their
        # career stats
        driver = webdriver.Firefox(options=options)
        driver.install_addon("ublock_origin-1.68.0.xpi")

        # BR organizes players by first letter of last name, so find player in their
        # corresponding page
        driver.get(
            f"https://www.basketball-reference.com/players/{row['last_name'].lower()[0]}/"
        )
        player_link = driver.find_element(By.LINK_TEXT, row["full_name"]).get_attribute(
            "href"
        )
        driver.get(player_link)  # type: ignore

        # use IDs to get table for career averages, playoffs not needed yet
        avgs_table = driver.find_element(By.ID, "per_game_stats")
        # then convert said table to Series for processing
        avgs = pd.read_html(StringIO(avgs_table.get_attribute("outerHTML")))[0]

        # extract the career row by regex, as it position can vary if they player has
        # played for multiple teams in their career
        avgs = avgs[avgs.fillna("")["Season"].str.contains(r"^\d Yrs?$")].iloc[0]

        # if a player has played in the playoffs, he'll have a playoff table as well
        pf_avgs_table = driver.find_elements(By.ID, "per_game_stats_post")
        has_pf = False
        if len(pf_avgs_table) == 1:
            # if the table exists, a Series for playoff averages can be made
            pf_avgs = pd.read_html(
                StringIO(pf_avgs_table[0].get_attribute("outerHTML"))
            )[0]
            pf_avgs = pf_avgs[
                pf_avgs.fillna("")["Season"].str.contains(r"^\d Yrs?$")
            ].iloc[0]
            # this is now set to true, so playoff Series can be concatenated later
            has_pf = True
        driver.quit()

        # from here, insert_missing will be used to ensure all required columns are
        # present, and columns will be renamed to align with the rest of the DataFrame
        avgs = insert_missing(avgs).rename(br_rename).rename(rename_avgs)

        # if the player has played in the playoffs, process their playoff Series also
        if has_pf:
            print("\tAlso played in playoffs")
            pf_avgs = insert_missing(pf_avgs).rename(br_rename).rename(rename_avgs).add_prefix("PF_")  # type: ignore
            # similar concatenation, just with playoffs included
            return pd.concat(
                [
                    avgs[7:10],
                    avgs[11:13],
                    avgs[18:20],
                    avgs[21:-1],
                    pf_avgs[7:10],
                    pf_avgs[11:13],
                    pf_avgs[18:20],
                    pf_avgs[21:-1],
                ]
            )

        # concatenate to ignore irrelevant columns and return the Series
        return pd.concat(
            [
                avgs[7:10],
                avgs[11:13],
                avgs[18:20],
                avgs[21:-1],
            ]
        )
    except requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError:
        print(f"{row['full_name']} caused a timeout")
        quit()

    # if the player has no seasons, skip over them; they'll be removed later
    if len(avgs[0]) == 0:
        print(f"{row['full_name']} never played in the NBA")
        return pd.Series()

    # if a player has never played a playoff game, only return their regular season
    # averages (index 1 in the list)
    if len(avgs[3]) == 0:
        return clean_avgs(avgs[1])

    # otherwise, add in the playoff DataFrames at index 3
    return pd.concat(
        [
            clean_avgs(avgs[1]),
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
    sleep(10)
    awards = playerawards.PlayerAwards(
        row["id"], headers=custom_headers
    ).get_data_frames()[0]

    # dictionary used to map All-NBA team numbers to distinguish between each team
    team_nums = {"1": "1st", "2": "2nd", "3": "3rd"}
    awards.loc[
        awards["ALL_NBA_TEAM_NUMBER"].fillna("").str.isnumeric(), "DESCRIPTION"
    ] = (
        awards["ALL_NBA_TEAM_NUMBER"].map(team_nums) + " Team " + awards["DESCRIPTION"]
    )

    if "Hall of Fame Inductee" in awards["DESCRIPTION"].values:
        print(f"{row['full_name']} is a Hall of Famer")

    # Hall of Fame Inductee is a listed award, so HOF status will be numeric for now
    return awards.groupby("DESCRIPTION").size()


def inactive_totals():
    """
    Gets total stats for inactive players df and saves it to a csv file to create a
    checkpoint, along with preserving inactive_ineligibles and never_in_nba sets
    """

    global inactives
    # for each function, apply will create a DataFrame that can be concatenated on
    print("Begin scraping totals for inactive players...")
    inactives = pd.concat([inactives, inactives.apply(get_totals, axis=1)], axis=1)

    # adding an intermediate save to csv file as a fail-safe so I wouldn't have to
    # repeat the entire stats process again in the event of internet going out, etc
    inactives.to_csv("eligible_player_data.csv", index=False)

    # inactive_ineligible and never_in_nba sets are saved to files for the same reason
    with open("inactive_ineligibles.pkl", "wb") as file:
        pickle.dump(inactive_ineligibles, file)
    with open("never_in_nba.pkl", "wb") as file:
        pickle.dump(never_in_nba, file)


def inactive_avgs():
    """
    Restores inactive df from previous checkpoint and adds on average stats before
    saving for another checkpoint
    """

    print("Finished scraping totals for inactive players, begin scraping averages")
    # inactives in read in after being saved at the previous checkpoint
    inactives = pd.read_csv("eligible_player_data.csv")
    inactives = pd.concat([inactives, inactives.apply(get_avgs, axis=1)], axis=1)
    inactives.to_csv("eligible_player_data.csv", index=False)


def inactive_awards():
    """
    Restores inactive df from previous checkpoint, adds on awards, removes inactive-
    ineligible and never_in_nba players, and saves it to the final csv
    """

    print("Finished scraping stats for inactive players, begin scraping awards...")
    inactives = pd.read_csv("eligible_player_data.csv")
    inactives = pd.concat(
        [inactives, inactives.apply(get_awards, axis=1)], axis=1
    ).fillna(0)
    print("Finished scraping awards, begin removing players and saving to csv file...")

    # restore inactive_ineligibles and never_in_nba from their pickle files
    with open("inactive_ineligibles.pkl", "rb") as file:
        inactive_ineligibles = pickle.load(file)
    with open("never_in_nba.pkl", "rb") as file:
        never_in_nba = pickle.load(file)

    # use the ii set to remove any of those players from the inactive df
    inactive_ineligibles_df = inactives[
        inactives["full_name"].isin(inactive_ineligibles)
    ]
    # this df is saved to a file for adding later onto active df
    inactive_ineligibles_df.to_csv("inactive_ineligibles.csv", index=False)
    # the never_in_nba set is used in a similar manner for removal
    never_in_nba_df = inactives[inactives["full_name"].isin(never_in_nba)]
    # that df is then saved to a csv for easy access/to prevent repeated scraping
    inactives.drop(inactive_ineligibles_df.index).drop(never_in_nba_df.index).to_csv(
        "eligible_player_data.csv", index=False
    )


def active_totals():
    """
    Gets total stats for active players df and saves it to a csv file to create a
    checkpoint
    """

    global actives
    print("Finished saving inactives df, begin scraping totals for active players...")
    actives = pd.concat([actives, actives.apply(get_totals, axis=1)], axis=1)
    actives.to_csv("ineligible_player_data.csv", index=False)


def active_avgs():
    """
    Restores active df from previous checkpoint and adds on average stats before
    saving for another checkpoint
    """

    print("Finished scraping totals for active players, begin scraping averages")
    actives = pd.read_csv("ineligible_player_data.csv")
    actives = pd.concat([actives, actives.apply(get_avgs, axis=1)], axis=1)
    actives.to_csv("ineligible_player_data.csv", index=False)


def active_awards():
    """
    Restores active df from previous checkpoint, adds on awards, adds back inactive-
    ineligible players, and saves it to the final csv
    """

    print("Finished scraping stats for active players, begin scraping awards...")
    actives = pd.read_csv("ineligible_player_data.csv")
    actives = pd.concat([actives, actives.apply(get_awards, axis=1)], axis=1).fillna(0)
    print("Finished scraping awards, begin adding IIs and saving to csv file...")

    # restore inactive_ineligible df to be added onto active df
    inactive_ineligibles_df = pd.read_csv("inactive_ineligibles.csv")

    pd.concat([actives, inactive_ineligibles_df]).to_csv(
        "ineligible_player_data.csv", index=False
    )
    print("Finished scraping!")


# When arranged into functions like this, it's much easier to comment out a previous
# checkpoint
inactive_totals()
inactive_avgs()
inactive_awards()
active_totals()
active_avgs()
active_awards()

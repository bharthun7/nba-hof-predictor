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
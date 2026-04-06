# NBA Hall of Fame Predictor

## Table of Contents

[Background](#background)

[Data Acquisition](#data-acquisition)

[Analyzing the Data](#analyzing-the-data)

[Creating the Model](#creating-the-model)

[How to Run the Model Yourself](#how-to-run-the-model-yourself)

[Planned Upcoming Features](#planned-upcoming-features)

## Background

If LeBron James retired today, would he make the Hall of Fame? That's probably a stupid question; a better one would be asking if he's the GOAT, but that's not the point of this project. With his résumé, no one would be surprised if The King ended his career and was a first-ballot HOFer.

But what about someone whose career isn't quite as close to being done? What about, say, Nikola Jokić? Sure, he probably has a few more years than LeBron left, but the answer is still almost certainly the same: another first-ballot HOF nod. With three MVPs, a championship, and some of the greatest numbers the league has ever seen, it's hard to argue otherwise.

OK, well, what about if Shai Gilgeous-Alexander shocked the world and retired today? This one should at least make you stop and think for a second. He did have one of the greatest seasons ever in 2024-25, both numbers and accolade-wise, but his career is still fairly young, and his career numbers on their own don't exactly scream "Hall of Fame" (as of April 2026).

Alright, final one: Derrick Rose. Like SGA, he has an MVP and an impressive peak, but no championship, and a career cut painfully short by injuries. Bringing up his candidacy for the Hall is when you start to have some real painful [discussions](https://www.reddit.com/r/NBATalk/comments/1k0l4oi/is_derrick_rose_a_hall_of_famer/) about peak vs. longevity.

That was ultimately what led me to make this project. [The Basketball Hall of Fame](https://www.hoophall.com/) is notorious for having a rather [vague set of entry criteria](https://www.nytimes.com/athletic/5394827/2024/04/06/basketball-hall-of-fame-2024-michael-cooper/), and I wanted to see if there was a way to objectively evaluate current NBA players' (and those who haven't been retired long enough to be eligible) chances of making the Hall of Fame. Before you ask, yes, I am aware that it is the Basketball Hall of Fame, not the NBA Hall of Fame, and players are sometimes inducted for collegiate or overseas careers. For the purposes of this project, I'll only consider a player's chance of making the Hall of Fame in the context of their NBA career.

To achieve this goal, I decided to create a machine learning model that, when trained on the career stats and accolades of all HOF-eligible players and whether or not they made the Hall of Fame, would perform a binary classification on all ineligible players based on their current stats and accolades. I could then use the probability of a positive classification as their probability of making the Hall of Fame. Here is how it went.

## Data Acquisition

I collected most of my data using the [nba_api](https://github.com/swar/nba_api) package, which is an awesome package that, when combined with [pandas](https://pandas.pydata.org/), makes it super convenient to access virtually any data from the history of the NBA. For my purposes, I only needed to scratch the surface and collect the following: a DataFrame of all active and inactive players, the career stats for each player, and the career awards for each player.

For most players, collecting the data was fairly simple; I would use the playercareerstats endpoint to extract DataFrames of their career total stats and average stats, and after some light processing, return them as a Series to be concatenated onto the main DataFrame. This involved checking if the player had played in the playoffs and including those stats as well if so.

Additionally, for inactive players, I would check their most recent season to determine if they were eligible for the Hall of Fame (defined as being retired for four full seasons). Any ineligible players were stored in a set to be switched from the inactive to active DataFrames later. However, I soon ran into several issues:

#### 1. Empty nba.com Pages

nba_api pulls its data from [nba.com](https://www.nba.com/players), which, normally, is perfectly fine, since it is the NBA's official website. However, for about 100 players, their stats page on nba.com is completely blank. It occurred in inactive players from the late 90s to the present, with no apparent pattern outside of most having only played a single season and having rather unremarkable careers. These players caused the endpoint to raise an error, so I needed an alternative way to get their stats.

I turned to [Basketball Reference](https://www.basketball-reference.com/), a reputable site for basketball data. All of the missing nba.com players had a valid page on Basketball Reference, so if the error signifying that a player's page on nba.com was empty was caught, I could scrape BR to get their stats instead. Since the stat tables on BR are dynamically loaded, I used Selenium to scrape the website and get the data I needed, which led to even more problems.

#### 2. Mismatched Names

Basketball Reference organizes players by name, so when I looked up a player's page by their name, some issues arose if their name on nba.com did not match their name on BR, which was mostly due to accented characters. So, for any BR-scraped players with mismatched names, I had to manually change their names before scraping to align with BR. Luckily, nba_api uses a player's unique ID number as a parameter for its endpoints, so this did not affect getting their awards later.

#### 3. Column Differences

The columns for some stats have different names on nba.com vs BR, which is fixed by a simple dictionary mapping rename. This was not the only difference, though. While nba.com includes all basic stat columns, regardless of whether they can apply to the player, BR only includes columns that can apply to the player. For example, for players who played before stats like turnovers and blocks were tracked, there is no column on BR. Or, if the player has zero career 3PT attempts, there is no column for 3P%. This caused issues when slicing the row to return the relevant columns, as the indices would not match, leading to incorrectly returned columns.

To remedy this, I had to create a separate function that would scan the row of a player's career stats scraped from BR, and if a stat was missing, insert a zero value into the correct ordered position.

#### 4. Players Who've Never Played

There were an even smaller number of players who, despite being included in the DataFrame of all active players, had never actually played in the NBA. From what I could tell, all of these players were recent draft picks who were on two-way contracts and had only played in the G-League thus far. The strange part was that they didn't raise an error when calling the endpoint with their ID, as the BR-scraped players did; it just returned empty DataFrames, which would naturally raise errors when trying to process.

So, I had to add a separate check that if the DataFrame of a player's stats was empty, a Series of zeros would be returned as a placeholder, and they would be added to a set marked for removal from the DataFrame at the end, much like the inactive-ineligible players.

#### 5. Timeouts

Calling each endpoint took 1-2 seconds, so for 5000 players with three calls each (total stats, average stats, and awards), I estimated that the scraping would take around six hours in total. Unfortunately, I quickly ran into the issue where, due to rate, throttling, my IP would get temporarily banned from nba.com after a couple of hundred requests, preventing me from making any future requests to the site.

After lots of trial and error, the only way I was able to get around this was to use custom headers in my requests and institute a timeout of 10 seconds between each call. Suddenly, I was staring at multiple days' worth of runtime, with constant internet access the entire time. Given I needed to go to class and couldn't just leave my computer at home to run the program, I broke it up into six sections, one for each endpoint for both active and inactive, that could be run in chunks.

In each section, after the data was collected, I would save it to a .csv file, so that when I was ready to run the next section, I could load in the .csv and comment out the previous section to pick up where I left off. Eventually, I acquired all the data I needed and was ready to start analyzing it and making the model.

## Analyzing the Data

Before creating any ML models, I had to do what any good data scientist would and verify the accuracy of my data. I checked that there were no negative or NA values, the inactive-ineligible players ended up in the correct DataFrame, the G-League players didn't end up anywhere, and pulled random rows to check that random players' stats were correct. None of these turned up any issues, but I discovered a few once I started looking at individual columns.

I sorted each DataFrame by each column to see if there were any illogical values, and to verify that the leaderboards for those stats were accurate to real life. One possible illogical value in my mind was a value other than zero or one for any rookie-based award, as a player is only a rookie once. And sure enough, for some odd reason, Paul Millsap's page on nba.com shows him as being a 2x member of the All-Rookie 2nd Team, so I had to manually correct that.

I also looked into the number of players with a positive value (indicating having won the award) for each award, and discovered that the number of players listed as being in the Hall of Fame (represented as a zero or one) was lower than expected. Every player inducted after 2018 had not had their page on nba.com correctly updated to reflect this, and there were also some players who, despite not being Hall of Famers, were listed as inductees on their pages, and I could not figure out why. These incorrect HOF statuses required some more manual corrections, but after that, the data was accurate as far as I could tell, so it was time to start building the model.

## Creating the Model

I performed some standard steps on the DataFrames before fitting a model, like reordering the columns from their previous alphabetical ordering to a more logical one with HOF status at the end. I also split the training data (eligible players) into a train/test split and removed the HOF status column from the ineligible dataset since that's what I aimed to predict.

From there, I set up the basic framework for the model: a classifier would be fitted on the training data before predict_proba was used to obtain Hall of Fame probabilities for eligible players. Through the course of testing and tuning, I found the best combination to be a StandardScaler, SelectKBest that allowed the user to restrict the number of features used to predict, and a LogisticRegression classifier with some hyperparameters tweaked. I also made the following modifications to the DataFrames to improve the model:

#### 1. Dropped Columns

Before even fitting the model and restricting prediction columns, I dropped some columns from the DataFrames altogether. These included some personal columns that were no longer necessary, like ID and whether the player was active, awards that are obsolete that no ineligible player has ever won, like the Comeback Player of the Year Award, and awards that don't/shouldn't factor into consideration, like the Sportsmanship Award. There was also an additional "All-Star Selection" column that one singular player had a value of one in for some reason, so I removed that, too.

#### 2. ABA Stats

One interesting thing I noticed when analyzing the data was the NBA's treatment of stats from other leagues. Stats and accolades from the NBA's predecessor leagues, like the [BAA](https://en.wikipedia.org/wiki/Basketball_Association_of_America), were included, but stats and accolades from the [ABA](https://en.wikipedia.org/wiki/American_Basketball_Association), the rival league in the 60s-70s that eventually folded in the NBA (bringing with it the 3-pointer), were not included. However, this was neither unintentional nor an error; the NBA has maintained that the ABA is a separate league, and thus, does not include its stats in its records. The Hall of Fame, on the other hand, does consider ABA Stats when evaluating a player's candidacy.

Not having ABA stats would inherently lead to a worse model, as while some players, like Julius Erving, are worthy of the Hall of Fame based on their NBA careers alone, others, like Mel Daniels, are included strictly due to their ABA careers. Luckily, our good friend Basketball Reference aggregates both NBA and ABA stats, so I could use the same scraping procedure as before to get a player's combined stats from both leagues.

So, in the scraper, if the current player was one from the ABA, I could manually raise the error, indicating the player needed to be scraped from BR to achieve the same effect. However, BR does not provide a list of all ABA players, so I had no way of knowing which players I needed to trigger a BR scrape for. I settled on only players from the ABA who were in the Hall of Fame, my rationale being that the model would benefit greatly from having players classified as being in the Hall of Fame to have their most accurate, higher stat totals, but players not in the Hall of Fame having lower stat totals wouldn't hurt the model as much.

#### 3. ABA Awards

Just like ABA stats, ABA accolades are not included in NBA records, but are factored into Hall of Fame consideration. And while BR is wonderful for stats, its award listing for each player is not as comprehensive as the NBA's website, so I needed to find a different way to obtain ABA awards. At the end of the day, I settled on manually adding them to the DataFrame, aided mostly by [this](https://en.wikipedia.org/wiki/List_of_American_Basketball_Association_awards_and_honors) source.

I also chose to treat each award as its NBA equivalent, since if they were different columns, it wouldn't affect the players predicting on, as they would all just have zeros for each award. The awards also line up pretty much perfectly with their NBA equivalents, as the ABA Playoff MVP was always given to an ABA Finals winner anyway, and the level of competition in both leagues was similar.

#### 4. Additional Hall of Fame Removals

In addition to the players erroneously marked as being in the Hall of Fame, whose HOF status I removed in the DataFrame, there were a number of other players who were legitimately in the Hall of Fame, but whose HOF status I chose to remove for other reasons.

Many of them are players who played internationally, like Arvydas Sabonis, who, like the ABA players, are in the Hall due to their careers in other leagues, but unlike the ABA players, I had no way of reliably recovering their stats, and the level of competition was much different. There were also some miscellaneous players, many from the early days of the NBA, for whom I could not discern a reason for their induction into the Hall of Fame.

With all the players removed, the underlying theme was that if I were fitting a model to predict a player's chance of making the Hall of Fame based solely on their NBA career, I should not classify players as Hall of Famers whose NBA career is not the reason they are a Hall of Famer.

## How to Run the Model Yourself

The `hof_model.py` file can be run from the command line, and is an interactive program that allows the user to edit a player's stats, change the number of features predicted on, run the model, and view various parts of its output. To run it, you will need Python installed along with the following libraries:

- [pandas](https://pypi.org/project/pandas/)
- [scikit-learn](https://pypi.org/project/scikit-learn/)

In addition, you'll also need copies of the `eligible_player_data.csv` and `ineligible_player_data.csv` files. These files are provided in the repo for convenience, but may not be 100% current, as I only run the scraper to recreate them periodically.

If you'd like to create the files yourself, you can run the `player_scraper.py` file, which requires Python along with the following libraries:

- [requests](https://pypi.org/project/requests/)
- [pandas](https://pypi.org/project/pandas/)
- [nba_api](https://pypi.org/project/nba_api/)
- [selenium](https://pypi.org/project/selenium/)

To run the scraper, scroll to the bottom of the file, where you'll see the six "checkpoint" functions. Running the file will complete the checkpoints in order, which is necessary for properly collecting the data, but if you need to run the program in multiple sessions, you can comment out any checkpoints you've already completed, which is indicated by command line output when you run the program.

Some other things to note about the scraper are that there are other print statements currently commented out that provide some additional info when special case players are scraped, which can also help gauge how far along you are. The .csv and .pkl files are used for intermediate saving between checkpoints so that the program can be run in multiple sessions if need be.

For the Selenium scraping, a Firefox driver is used with [uBlock Origin](https://ublockorigin.com/) installed. A copy of the file is provided in the repo, but if you don't want to install it on the driver or want to use a different driver, you can modify that in the scraper. Each of the inactive checkpoints takes approximately 13 hours, and each of the active ones takes approximately 3, although these times are based only on my computer.

## Planned Upcoming Features

#### Scraper

- Obtain win shares and/or other advanced stats

- Measure appearances on season stat leaderboards as a way to quantify a player's peak

#### Modeler

- Print incorrectly classified players from testing data

- Create a new player from scratch for predicting on

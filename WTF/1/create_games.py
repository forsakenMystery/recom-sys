import pandas as pd
import numpy as np

# Read the dataset into a data table using Pandas
games_table_df = pd.read_csv("games.csv")
users_table_df = pd.read_csv("users.csv")
games_rating_df = pd.read_csv("games_raw_Full.csv")

games_rating_df['game_name'] = games_rating_df['game_name'].map(games_table_df.set_index('game_name')['game_id'])
games_rating_df['user_id'] = games_rating_df['user_id'].map(users_table_df.set_index('user_id_steam')['user_id'])

# Create a csv file of the data for easy viewing
games_rating_df.to_csv("games_ratings_data_set.csv", index=False, na_rep="")

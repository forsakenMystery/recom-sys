import pandas as pd
import numpy as np

# Read the dataset into a data table using Pandas
df = pd.read_csv("games_raw_Full.csv")

# Convert the running list of user ratings into a matrix using the 'pivot table' function
unique_games = df['game_name'].unique()
unique_users = df['user_id'].unique()

games_df = pd.DataFrame(unique_games, columns=['game_name'])
games_df.index.names = ['game_id']
games_df.index += 1

games_ratings_df = pd.DataFrame(unique_users, columns=['user_id_steam'])
games_ratings_df.index.names = ['user_id']
games_ratings_df.index += 1

# Create a csv file of the data for easy viewing
games_df.to_csv("games.csv", na_rep="")
games_ratings_df.to_csv("users.csv", na_rep="")

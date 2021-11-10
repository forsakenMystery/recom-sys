import pickle
import pandas as pd

# Load prediction rules from data files
means = pickle.load(open("game_means.dat", "rb"))

# Load game titles
games_df = pd.read_csv('games.csv', index_col='game_id')

# Just use the average game ratings directly as the user's predicted ratings
user_ratings = means

print("Movies we will recommend:")

games_df['rating'] = user_ratings
games_df = games_df.sort_values(by=['rating'], ascending=False)

print(games_df[['game_name', 'rating']].head(5))
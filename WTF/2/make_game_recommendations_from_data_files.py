import pickle
import pandas as pd

# Load prediction rules from data files
U = pickle.load(open("game_user_features.dat", "rb"))
M = pickle.load(open("game_product_features.dat", "rb"))
predicted_ratings = pickle.load(open("game_predicted_ratings.dat", "rb"))

# Load game titles
games_df = pd.read_csv('games.csv', index_col='game_id')

user_df = pd.read_csv('users.csv', index_col='user_id')

print("Enter a user_id to get recommendations (Between 1 and 220):")
user_id_to_search = int(input())

user_id_steam = user_df.loc[user_id_to_search,'user_id_steam']
print("Games we will recommend to user %s:" % user_id_steam)

user_ratings = predicted_ratings[user_id_to_search - 1]
games_df['rating'] = user_ratings
games_df = games_df.sort_values(by=['rating'], ascending=False)

print(games_df[['game_name', 'rating']].head(5))

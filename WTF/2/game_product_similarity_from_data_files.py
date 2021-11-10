import pickle
import pandas as pd
import numpy as np

# Load prediction rules from data files
M = pickle.load(open("game_product_features.dat", "rb"))

# Swap the rows and columns of product_features just so it's easier to work with
M = np.transpose(M)

# Load game titles
games_df = pd.read_csv('games.csv', index_col='game_id')

# Choose a game to find similar games to. Let's find games similar to game #5:
game_id = 5

# Get game #1's name and genre
game_information = games_df.loc[game_id]

print("We are finding games similar to this game:")
print("Game title: {}".format(game_information.game_name))

# Get the features for game #1 we found via matrix factorization
current_game_features = M[game_id - 1]

print("The attributes for this game are:")
print(current_game_features)

# The main logic for finding similar games:

# 1. Subtract the current game's features from every other game's features
difference = M - current_game_features

# 2. Take the absolute value of that difference (so all numbers are positive)
absolute_difference = np.abs(difference)

# 3. Each game has several features. Sum those features to get a total 'difference score' for each game
total_difference = np.sum(absolute_difference, axis=1)

# 4. Create a new column in the game list with the difference score for each game
games_df['difference_score'] = total_difference

# 5. Sort the game list by difference score, from least different to most different
sorted_game_list = games_df.sort_values('difference_score')

# 6. Print the result, showing the 5 most similar games to game_id #1
print("The five most similar games are:")
print(sorted_game_list[['game_name', 'difference_score']][0:5])

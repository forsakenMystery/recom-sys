import pandas as pd
import os

path = os.path.join(os.path.dirname(__file__), "MovieLens/ml-25m")
movies = pd.read_csv(os.path.join(path, 'movies.csv'))
movies.head()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("steam-200k.csv")
df.columns = ["user_id", "game", "activity", "hours", "unknown"]

df_users = df[["user_id","game"]][df["activity"]=="play"].groupby("user_id", axis=0).count().reset_index()
df_users["game"][df_users["game"] < 10].hist()
plt.title("How many different games are played by how many users?")
plt.show()
print("Top users:", len(df_users["user_id"][df_users["game"]>2].unique()))

user_games_dict = df[["user_id", "game"]][df["activity"] == "play"].groupby("user_id", axis=0).count().to_dict()
true_gamers_dict = {}
for user in user_games_dict["game"]:
    if user_games_dict["game"][user] > 2:
        true_gamers_dict[user] = user_games_dict["game"][user]


def top_gamer(x):
    if x in true_gamers_dict:
        return 1
    else:
        return 0


df["gamer"] = df["user_id"].map(lambda x: top_gamer(x))
df_gamers = df[df["gamer"] == 1][df["activity"] == "play"]
print("Top users:", len(df_gamers["user_id"].unique()))

df_recom = df_gamers[["user_id", "game", "hours"]]

vectors = {}

usersand = []

for index in df_recom.index:
    row = df_recom.loc[index, :]
    user_id = row["user_id"]
    usersand.append(user_id)
    game = row["game"]
    hours = row["hours"]
    if user_id not in vectors:
        vectors[user_id] = {}
    else:
        pass
    vectors[user_id][game] = hours

user_example = 103804924

print(vectors[103804924])


def corr_users(vectors, random_id):
    best = []
    for user in vectors:

        possible_recom = []
        matched_games = []

        vector_1 = vectors[random_id]
        vector_2 = vectors[user]

        given_vector = []
        matched_vector = []

        if user != random_id:

            for game in vector_2:

                if game in vector_1:
                    matched_games.append(game)

                else:
                    possible_recom.append(game)

            for game in matched_games:

                given_vector.insert(0, vector_1[game])
                matched_vector.insert(0, vector_2[game])

        if len(matched_games) > 4:
            # print(given_vector)
            # print(matched_vector)
            # input()
            corr = np.corrcoef(x=given_vector, y=matched_vector, rowvar=True)[0][1]

            dic = {}
            for game in possible_recom:
                dic[game] = vector_2[game]

            best.append((corr, dic))

        else:
            pass

    print("You were matched up with this number of gamers:", len(best))

    if len(best) == 0:
        print("Warning: No matches")

    else:
        print("Coincidence levels are: ")
        for i in best:
            print(str(i[0]))

        # ans = sorted(best, key=lambda x: x[0], reverse=True)
        # print(ans)
        best_positive = sorted(best, key=lambda x: x[0], reverse=True)[0]
        second_positive = sorted(best, key=lambda x: x[0], reverse=True)[1]
        second_recom = max(second_positive[1])
        first_recom = max(best_positive[1])
        recoms = (first_recom, second_recom)

        print("We recommend : ")
        print("-" + recoms[0] + "  Coincidence: " + str(best_positive[0] * 100)[:5] + "%")
        print("-" + recoms[1] + "  Coincidence: " + str(abs(second_positive[0] * 100))[:5] + "%")

correlated_users = corr_users(vectors, user_example)

print("give me a number between 0 to "+str(len(usersand)))
a = int(input())
if 0 <= a < len(usersand):
    corr_users(vectors, usersand[a])

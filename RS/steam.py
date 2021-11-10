import json
import numpy as np


def reading_dataset(path):
    file = open(path, "r")
    x_list = file.readlines()
    i = 0
    holder = []
    for x in x_list:
        x = x.replace("\'", "\"")
        # print(type(x))
        # print(x)
        print(i)
        i += 1
        y = json.loads(x)
        holder.append(y)
        if i==125:
            break
        # print(y["steam_id"])
    # ans = holder[0]["items"]
    # print(ans)
    # print(ans[0])
    train = holder[:100]
    validation = holder[100:105]
    test = holder[105:]
    return train, test, validation


def dictionary(data, extract):
    dictionary_x_y = {}
    dictionary_y_x = {}
    for i in range(len(data)):
        dictionary_x_y[i] = data[i][extract]
        dictionary_y_x[data[i][extract]] = i
    return dictionary_x_y, dictionary_y_x


def item_rating(games, games_dictionary, played, played_times, maxx, minn, avgg, data):
    item_play = {}
    for i in range(len(data)):
        ans = data[i]["items"]
        lel = []
        for j in range(len(ans)):
            sesami = ans[j]
            print(sesami)
            if sesami["item_id"] in games:
                played[sesami["item_id"]] = played[sesami["item_id"]] + sesami["playtime_forever"] + 1
                played_times[sesami["item_id"]] = played_times[sesami["item_id"]] + 1
                if maxx[sesami["item_id"]] < sesami["playtime_forever"] + 1:
                    maxx[sesami["item_id"]] = sesami["playtime_forever"] + 1
                if minn[sesami["item_id"]] > sesami["playtime_forever"] + 1:
                    minn[sesami["item_id"]] = sesami["playtime_forever"] + 1
            else:
                games.add(sesami["item_id"])
                games_dictionary[sesami["item_id"]] = sesami["item_name"]
                played[sesami["item_id"]] = sesami["playtime_forever"] + 1
                played_times[sesami["item_id"]] = 1
                maxx[sesami["item_id"]] = sesami["playtime_forever"] + 1
                minn[sesami["item_id"]] = sesami["playtime_forever"] + 1
            avgg[sesami["item_id"]] = played[sesami["item_id"]] / played_times[sesami["item_id"]]
            lel.append({sesami["item_id"]:sesami["playtime_forever"] + 1})
        item_play[data[i]["steam_id"]] = lel
    # print()
    # print()
    # print()
    # print("Wow")
    # print()
    # print()
    # print(item_play)
    # print()
    # print()
    # print()
    # print("Mow")
    # print()
    # print()
    # print()
    # print(maxx)
    # print()
    # print()
    # print()
    # print("nice")
    # print()
    # print()
    # print()
    # print(minn)
    # print()
    # print()
    # print()
    # print("hoot")
    # print()
    # print()
    # print(avgg)
    # print()
    # print()
    # print()
    # print("dammit")
    # print()
    # print()
    mine = {}
    for me in item_play:
        # print(me)
        # print(item_play[me])
        my_list = []
        for item in item_play[me]:
            # print(item)
            diction = {}
            for x in item:
                y = item[x]
                # print(x)
                # print(y)
                if avgg[x] == minn[x] and avgg[x] == 1:
                    diction[x] = 1
                elif avgg[x] == minn[x]:
                    diction[x] = 5
                else:
                    diction[x] = round((y - minn[x])*(5-1) / (avgg[x] - minn[x]) + 1)
                if y > avgg[x]:
                    diction[x] = 5
                # print(diction[x])
                # print(round((y - minn[x])*(5-1) / (avgg[x] - minn[x]) + 1))
                # input()
            my_list.append(diction)
        mine[me] = my_list
    # print(mine)
    # print()
    # print()
    return mine
    # np.zeros([len(data), ])
    # (x - min)*(b-a) / (avg - min) + a ; a = 1, b = 5, max -> avg because if you play more than avg you like it :D


def main():
    train, test, validation = reading_dataset("australian_users_items.json")
    # train, test, validation = reading_dataset("australian_user_reviews.json")
    x_y, y_x = dictionary(train, "steam_id")
    print(x_y)
    print(len(x_y))
    games = set([])
    games_dictionary = {}
    games_played = {}
    games_played_times = {}
    game_max = {}
    game_min = {}
    game_average = {}
    rating_list = item_rating(games, games_dictionary, games_played, games_played_times, game_max, game_min, game_average, train)
    print(rating_list)


if __name__ == '__main__':
    main()

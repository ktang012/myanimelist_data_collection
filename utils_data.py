import os
import json
import math
import urllib
import time
import itertools

from collections import Counter
import matplotlib.pyplot as plt

def load_raw_animes_to_memory(path):
    animes = {}
    KEYS_TO_DROP = [
        "request_cached", "request_cache_expiry", "trailer_url",
        "request_hash", "background", "airing", 
        "broadcast", "licensors", "type"
    ]
    
    for anime_name in os.listdir(path):
        anime_path = os.path.join(path, anime_name)
        
        with open(os.path.join(anime_path, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        with open(os.path.join(anime_path, "reviews.json"), "r") as f:
            reviews = json.load(f)
            reviews = [review for page in reviews for review in page]
            
        with open(os.path.join(anime_path, "stats.json"), "r") as f:
            stats = json.load(f)
        
        for key in KEYS_TO_DROP:
            metadata.pop(key, None)
            stats.pop(key, None)
        
        combined = {**metadata, **stats}
        combined["reviews"] = reviews
        
        animes[combined["mal_id"]] = combined
        
    return animes

def load_animes_to_memory(path):
    with open(os.path.join(path, "mal_data.json"), "r") as f:
        data = json.load(f)
    return data

def retrieve_image(image_url, dest):
    time.sleep(3)
    urllib.request.urlretrieve(image_url, dest)

def round_to_half(n):
    return round(n * 2) / 2.0
    
def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

### ----- visualize -----
def plot_normalized_word_counts(word_counts, most_common=50):
    words, counts = zip(*[(word[0], math.log(word[1])) for word in word_counts.most_common(most_common)])
    indexes = list(range(len(words)))
    
    fig, ax = plt.subplots()
    fig.set_size_inches(24,16)
    ax.set_xlabel("Index", fontsize='24')
    ax.set_ylabel("log(term frequency)", fontsize='24')
    ax.scatter(indexes, counts)
    for i, word in enumerate(words):
        if i % 4 == 0:
            xytext = (indexes[i], counts[i] + 2e-1)
        elif i % 3 == 0:
            xytext = (indexes[i] - 1.75, counts[i] - 4e-1)
        elif i % 2 == 0:
            xytext = (indexes[i], counts[i] + 4e-1)
        else:
            xytext=(indexes[i] - 1.75, counts[i] - 2e-1)

        ax.annotate(word, xy=(indexes[i], counts[i]), xytext=xytext,
                    arrowprops=dict(facecolor="black", arrowstyle='-'),
                    fontsize='18')
    plt.show()

# input is a dict with studio_name: [popularity]
def scatterplot_x_by_y(dict_list, xlabel, ylabel, figsize=(18,36), ptsize=100, n_in_row=15):
    num_items = len(dict_list.keys())
    nrows = math.ceil(num_items / n_in_row)
    fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=figsize)
    
    row_num = 1
    for i, (key, val) in enumerate(dict_list.items()):
        if i % n_in_row == 0:
            plt.subplot(nrows, 1, row_num)
            row_num += 1
        plt.xticks(rotation=30, fontsize=11)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.scatter([key] * len(val), val, label=key, s=ptsize)
    
    plt.show()

# boxplot of studios by score; input is dict with studio_name: [scores]
def boxplot_x_by_y(dict_list, xlabel, ylabel, figsize=(18,36), n_in_row=12):
    num_items = len(dict_list.keys())
    nrows = math.ceil(num_items / n_in_row)
    fig, ax = plt.subplots(nrows=nrows, ncols=1, figsize=figsize)
    
    row_num = 1
    for keys, values in zip(grouper(dict_list, n_in_row, "N/A"), 
                            grouper(dict_list.values(), n_in_row, 0)):
        plt.subplot(nrows, 1, row_num)
        plt.boxplot(values, notch=True, bootstrap=1000)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.xticks(range(1,len(keys) + 1), keys, rotation=30, fontsize=10)
        row_num += 1

    plt.show()

### ----- counting -----
def total_num_of_reviews(animes):
    count = 0
    for id, anime in animes.items():
        count += len(anime["reviews"])
    return count
    
def get_normalized_word_count(animes):
    words = Counter()
    
    for mal_id, anime in animes.items():
        for review in anime["reviews"]:
            doc = review["content"]
            doc = doc.split(" ")
            doc_length = len(doc)
            for word in doc:
                word = word.lower()
                if word not in words:
                    words[word] = 1/doc_length
                    
                else:
                    words[word] += 1/doc_length
    return words
    
def get_x_by_y(animes, x_field, y_field, ignore=[], is_round_to_half=False, drop_count = 0):
    try:
        x = {}
        if ignore:
            x["Other"] = []
        
        for mal_id, anime in animes.items():
            if is_round_to_half and type(anime[y_field]) is float:
                y_val = round_to_half(anime[y_field])
            else:
                y_val = anime[y_field]
            
            for item in anime[x_field]:
                name = item["name"]
                if name and name not in x and name not in ignore:
                    x[name] = [y_val]
                elif name not in ignore:
                    x[name].append(y_val)
                else:
                    x["Other"].append(y_val)
                    
        if drop_count != 0:
            x = {k: v for k, v in x.items() if len(v) > drop_count}
    
        return x
    
    except Exception as e:
        print(e)
        # x should be hashable and iterable in anime[x_field]

def count_genres(animes, other=[]):
    genres = Counter()
    genre_ids = Counter()
    if other:
        genres["Other"] = 0
        genre_ids["Other"] = 0

    for mal_id, anime in animes.items():
        for genre in anime["genres"]:
            if genre["name"] not in genres and genre["name"] not in other:
                genres[genre["name"]] = 1
                genre_ids[genre["mal_id"]] = 1
            elif genre["name"] not in other:
                genres[genre["name"]] += 1
                genre_ids[genre["mal_id"]] += 1
            else:
                genres["Other"] += 1
                genre_ids["Other"] += 1
                
    return genres, genre_ids
    
def count_studios(animes):
    studios = Counter()
    for mal_id, anime in animes.items():
        for studio in anime["studios"]:
            if studio["name"] not in studios and studio["name"]:
                studios[studio["name"]] = 1
            else:
                studios[studio["name"]] += 1
    return studios
    
def count_ratings(animes):
    ratings = Counter()
    for mal_id, anime in animes.items():
        score = round_to_half(anime["score"])
        
        if score not in ratings:
            ratings[score] = 1
        else:
            ratings[score] += 1
    return ratings
    
def count_field(animes, field_name):
    field = Counter()
    for mal_id, anime in animes.items():
        if field_name not in anime:
            print("Could not find key", field_name, "in data")
            return
        field_val = anime[field_name]
        if field_val not in field:
            field[field_val] = 1
        else:
            field[field_val] += 1
    return field
                

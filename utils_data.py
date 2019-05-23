import os
import json
import math
import urllib
import time

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


### ----- counting and visualizing -----
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

def count_genres(animes, other=None):
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
        score = round(anime["score"] * 2) / 2.0
        
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
                

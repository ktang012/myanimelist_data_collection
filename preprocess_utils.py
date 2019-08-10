import pandas as pd
import spacy
import re

from pandas.io.json import json_normalize

def read_animes_json(path="data/mal_data.json"):
    return pd.read_json("data/mal_data.json", orient="index")

def get_reviews(anime):
    reviews = json_normalize(anime["reviews"])
    reviews["review_id"] = reviews["mal_id"]
    reviews["mal_id"] = anime["mal_id"]
    reviews["title"] = anime["title"]
    reviews["reviewer"] = reviews["reviewer.username"]
    reviews["episodes"] = anime["episodes"]
    reviews.drop(labels=["reviewer.image_url", "reviewer.url", "url", "reviewer.username"],
                 axis="columns", inplace=True)
    
    reorder = ["title", "mal_id", "reviewer", "review_id", "content", "reviewer.episodes_seen", "episodes", "helpful_count"]
    for i, col in enumerate(reorder):
        copy = reviews[col]
        reviews.drop(labels=[col], axis="columns", inplace=True)
        reviews.insert(i, col, copy)
    
    return reviews
    
def get_reviews_dataframe(animes_dataframe):
    reviews = pd.concat(animes_dataframe.apply(get_reviews, axis="columns").values.tolist())
    reviews.set_index("review_id", inplace=True)
    return reviews
    
def drop_anime_columns(animes_dataframe):
    cols = [
        'aired', 'title_synonyms', 'title_japanese', 'url', 'scores',
        'opening_themes', 'ending_themes', 'producers', 'rating', 'reviews',
        'status', 'duration', 'title_english', 'image_url', 'studios',
        'episodes', 'rank'
    ]
    
    return animes_dataframe.drop(labels=cols, axis="columns").set_index("mal_id")

def round_anime_score(animes_dataframe):
    animes_dataframe["score"] = animes_dataframe["score"].round()
    return animes_dataframe

# splits genres into two fields, one with genre id and another with genre name
# each field is a tuple
def reformat_genre(row):
    genres_field = row["genres"]
    genre_ids, genre_names = zip(*[(x['mal_id'], x['name']) for x in genres_field])
    return pd.Series({"genre_names": genre_names, "genre_ids": genre_ids})  

# turn genres into dummy variables
def create_dummy_genres(row, genres):
    is_genres = row["genre_names"]
    dummy = {}
    for genre in genres:
        if genre in is_genres:
            dummy["genres.is_" + genre] = 1
        else:
            dummy["genres.is_" + genre] = 0

    return pd.Series(dummy)

def encode_genres(animes_dataframe):
    GENRES = set(['Action', 'Adventure', 'Comedy', 'Drama', 'Ecchi',
              'Fantasy','Harem', 'Josei', 'Mystery', 'Romance',
              'School', 'Sci-Fi', 'Seinen','Shoujo', 'Shounen',
              'Slice of Life', 'Supernatural'])
              
    animes_dataframe = animes_dataframe.join(animes_dataframe.apply(reformat_genre, axis=1, 
                                             result_type="expand"), how="right")
    animes_dataframe = animes_dataframe.drop("genres", axis=1)

    animes_dataframe = animes_dataframe.join(animes_dataframe.apply(create_dummy_genres, 
                                                                    args=[GENRES],
                                                                    axis=1,
                                                                    result_type="expand"),
                                             how="right")
    return animes_dataframe.drop(labels=["genre_names", "genre_ids"], axis=1)

# check if it has prequels or sequels 
def create_dummy_related(row):
    MEDIA_TYPES = set(["Prequel", "Sequel"])
    prequel = "Prequel"
    sequel = "Sequel"
    
    related_field = row["related"]
    is_prequel, is_sequel = 0, 0
    for media_type, media_info in related_field.items():
        if media_type == "Prequel":
            is_prequel = 1
        elif media_type == "Sequel":
            is_sequel = 1
    return pd.Series({"related.has_Prequel": is_prequel, "related.has_Sequel": is_sequel})

def encode_related(animes_dataframe):
    animes_dataframe = animes_dataframe.join(animes_dataframe.apply(create_dummy_related, axis=1, 
                                             result_type="expand"), how="right")
    return animes_dataframe.drop("related", axis=1)

def encode_source_material(animes_dataframe):
    source_dummies = pd.get_dummies(animes_dataframe["source"], prefix="source.is")
    source_dummies.drop(source_dummies.columns.difference(["source.is_Manga", "source.is_Light novel",
                                                   "source.is_Original", "source.is_Visual novel"]),
                        axis=1, inplace=True)
    animes_dataframe = animes_dataframe.join(source_dummies, how="right")
    return animes_dataframe.drop("source", axis=1)
    
def encode_popularity(animes_dataframe, bins=[0, 275, 575, 955, float("inf")],
                      labels = ["Very_Popular", "Popular", "Unpopular", "is_Very_Unpopular"]):
    popularity_dummies = pd.get_dummies(pd.cut(animes_dataframe["popularity"], 
                                        bins=bins, labels=labels), prefix="popularity.is")
    animes_dataframe = animes_dataframe.join(popularity_dummies, how="right")
    return animes_dataframe.drop("popularity", axis=1) 

# splits premier date into season and year
def split_premiered(row):
    prem_field = row["premiered"].split(" ")
    
    return pd.Series({"season": prem_field[0], "year": int(prem_field[1])})

def encode_premiered(animes_dataframe):
    animes_dataframe = animes_dataframe.join(animes_dataframe.apply(split_premiered, axis=1,
                                             result_type="expand"), how="right")
    animes_dataframe = pd.get_dummies(animes_dataframe, columns=["season", "year"], prefix_sep=".")
    return animes_dataframe.drop("premiered", axis=1)
    
def get_view_status_percent(animes_dataframe):
    view_status = [
        "completed", "favorites", "dropped", 
        "scored_by", "watching", "plan_to_watch", "on_hold"
    ]
    
    total = animes_dataframe["total"]
    prefix = "percentage."
    df = pd.DataFrame()
    for status in view_status:
        df[prefix + status] = animes_dataframe[status] / total
    
    return df

def set_view_status_percent(animes_dataframe):
    view_status = [
        "completed", "favorites", "dropped", "members",
        "scored_by", "watching", "plan_to_watch", "on_hold"
    ]
    df = get_view_status_percent(animes_dataframe)
    animes_dataframe = animes_dataframe.join(df, on="mal_id")
    return animes_dataframe.drop(labels=view_status, axis="columns")

# default: take sum across all embeddings
def get_synopsis_embeddings(anime, nlp, method=None):
    if not method:
        return pd.Series(nlp(anime["synopsis"]).tensor.sum(axis=0)).add_prefix("synopsis.vector_")
    else:
        pass
        
def set_synopsis_embeddings(animes_dataframe, nlp=None, method=None):
    if not nlp:
        nlp = spacy.load('en_pytt_bertbaseuncased_lg')
        
    if not method:
        df = animes_dataframe.apply(get_synopsis_embeddings, args=[nlp], 
                                    axis=1, result_type="expand")
        return animes_dataframe.join(df, on="mal_id")
        
def load_and_preprocess(set_synopsis_embeddings=False):
    animes = read_animes_json()
    reviews = get_reviews_dataframe(animes)
    
    animes = drop_anime_columns(animes)
    animes = round_anime_score(animes)
    animes = encode_genres(animes)
    animes = encode_source_material(animes)
    animes = encode_related(animes)
    animes = encode_popularity(animes)
    animes = encode_premiered(animes)
    animes = set_view_status(animes)
    
    if set_synopsis_embeddings:
        animes = set_synopsis_embeddings(animes)
    
    return animes
        
def remove_newline_and_carriage_returns(text):
    return re.sub(r"\n\n|\r\n|\r|\\n", " ", text)
    
def get_review_embeddings(review, nlp, method=None):
    content = remove_newline_and_carriage_returns(review["content"])

    if not method:
        return pd.Series(nlp(content).tensor.sum(axis=0)).add_prefix("content.vector_")
    else:
        pass
        
def set_review_embeddings(reviews_dataframe, nlp=None, method=None):
    if not nlp:
        nlp = spacy.load('en_pytt_bertbaseuncased_lg')
    
    if not method:
        df = reviews_dataframe.apply(get_review_embeddings, args=[nlp], 
                                     axis=1, result_type="expand")
        return reviews_dataframe.join(df, on="review_id")
        
    
    
    
    
    
    
    
    
    
    
    

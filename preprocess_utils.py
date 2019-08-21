import numpy as np
import pandas as pd
import spacy
import re
import os
import cupy
import torch

from pandas.io.json import json_normalize

DATA_PATH = os.path.join("data", "bert_vectors")
TOKENS_TO_REMOVE = set(["SPACE", "SYMBOL", "NUM", "X"])

def read_animes_json(path=os.path.join("data", "mal_data.json")):
    return pd.read_json(path, orient="index")

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
    
    return reviews.drop_duplicates()
    
def get_reviews_dataframe(animes_dataframe):
    reviews = pd.concat(animes_dataframe.apply(get_reviews, axis="columns").values.tolist())
    reviews.set_index("review_id", inplace=True)
    return reviews.drop_duplicates()

def get_reviews_from_anime():
    df = read_animes_json()
    return get_reviews_dataframe(df)

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
def get_synopsis_embeddings(anime, nlp, method=None, use_gpu=False):
    if not method:
        embed = nlp(anime["synopsis"]).tensor.sum(axis=0)
        
        if use_gpu:
            embed = cupy.asnumpy(embed)
            
        return pd.Series(embed).add_prefix("synopsis.vector_")
    else:
        pass
        
def set_synopsis_embeddings(animes_dataframe, nlp=None, method=None):
    is_using_gpu = spacy.prefer_gpu()
    if is_using_gpu:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        
    if not nlp:
        nlp = spacy.load('en_pytt_bertbaseuncased_lg')
        
    df = animes_dataframe.apply(get_synopsis_embeddings, args=[nlp, method, is_using_gpu], 
                                axis=1, result_type="expand")
    return animes_dataframe.join(df, on="mal_id")
        
def preprocess_animes(animes, use_synopsis_embeddings=False):
    animes = drop_anime_columns(animes)
    animes = round_anime_score(animes)
    animes = encode_genres(animes)
    animes = encode_source_material(animes)
    animes = encode_related(animes)
    animes = encode_popularity(animes)
    animes = encode_premiered(animes)
    animes = set_view_status_percent(animes)
    
    if use_synopsis_embeddings:
        animes = set_synopsis_embeddings(animes)
    
    return animes
    
def load_preprocessed_animes(path=os.path.join(DATA_PATH, "animes.hdf5")):
    return pd.read_hdf(path, "table")
        
def remove_newline_and_carriage_returns(text):
    return re.sub(r"\n\n|\r\n|\r|\\n", " ", text)
    
    
def get_review_embeddings(reviews_dataframe, nlp=None, method=None, batch_size=32):
    is_using_gpu = spacy.prefer_gpu()
    if is_using_gpu:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    
    texts = list(reviews_dataframe["content"])
    texts = [remove_newline_and_carriage_returns(text) for text in texts]
    
    # treats entire review as a sentence
    if nlp == "BERT":
        nlp = spacy.load('en_pytt_bertbaseuncased_lg')
        embeds = []
        for doc in nlp.pipe(texts, batch_size=batch_size):
            if method == "mean":
                embeds.append(cupy.asnumpy(doc._.pytt_last_hidden_state.mean(axis=0)))
            elif method == "max":
                embeds.append(cupy.asnumpy(doc._.pytt_last_hidden_state.max(axis=0)))
            else:
                embeds.append(cupy.asnumpy(doc.tensor.sum(axis=0)))
    if nlp == "CST": 
        nlp = spacy.load('en_core_web_lg')
        embeds = []
        for doc in nlp.pipe(texts, batch_size=batch_size):
            if method == "mean":
                embeds.append(cupy.asnumpy(doc.tensor.mean(axis=0)))
            elif method == "max":
                embeds.append(cupy.asnumpy(doc.tensor.max(axis=0)))
            else:
                embeds.append(cupy.asnumpy(doc.tensor.sum(axis=0)))
    else:
        nlp = spacy.load('en_core_web_lg')
        embeds = []
        for doc in nlp.pipe(texts, batch_size=batch_size):
            embeds.append(cupy.asnumpy(doc.vector))
    
    return embeds
    

def get_episodes_seen_percent(reviews_dataframe):
    episodes_seen = reviews_dataframe["reviewer.episodes_seen"] / reviews_dataframe["episodes"]
    return episodes_seen.to_frame("reviewer.percentage.episodes_seen")
    
def set_episodes_seen_percent(reviews_dataframe):
    df = get_episodes_seen_percent(reviews_dataframe)
    reviews_dataframe = reviews_dataframe.join(df, on="review_id")
    return reviews_dataframe.drop(labels=["reviewer.episodes_seen", "episodes"], axis="columns")

def preprocess_reviews(reviews, use_review_embeddings=False):
    reviews = set_episodes_seen_percent(reviews)
    
    if use_review_embeddings:
        reset_index_reviews = reviews.reset_index()
        review_col_names = list(reset_index_reviews().columns)
    
        embeds = get_review_embeddings(reviews)
        embedded_reviews = pd.DataFrame([pd.np.column_stack([reviews, embeds])])
        
        num_review_cols = len(review_col_names)
        num_total_cols = len(list(embedded_reviews.columns))
        
        rename_cols = {i: review_col_names[i] for i in range(num_review_cols)}
        reviews.rename(columns=rename_cols, inplace=True)
        
        rename_embed_cols = {i: "bert_embed." + str(i-num_review_cols) for i in range(num_review_cols,num_total_cols)}
        embedded_reviews.rename(columns=rename_embed_cols, inplace=True)
        
        embedded_reviews.set_index("review_id", inplace=True)
        embedded_reviews = embedded_reviews.infer_objects()
        
        return embedded_reviews
    
    return reviews
    
def load_preprocessed_reviews(path=os.path.join(DATA_PATH, "reviews_sum.hdf5")):
    return pd.read_hdf(path, "table")
    
def tokenize_text(text, nlp, STOP_WORDS={}, tokens_to_remove=TOKENS_TO_REMOVE):
    text = text.strip().replace("\\n", "").replace("\\r", "").replace("\\'", "'")
    text = text.lower()
    
    from spacy.tokens import Token, Doc
    from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA, DEP, LEMMA, LOWER, IS_PUNCT, IS_DIGIT, IS_SPACE, IS_STOP
    import numpy as np
    # https://gist.github.com/Jacobe2169/5086c7c4f6c56e9d3c7cfb1eb0010fe8
    def remove_tokens(doc, index_todel,list_attr=[LOWER, POS, ENT_TYPE, IS_ALPHA, DEP, LEMMA, LOWER, IS_PUNCT, IS_DIGIT, IS_SPACE, IS_STOP]):
        """
        Remove tokens from a Spacy *Doc* object without losing associated information (PartOfSpeech, Dependance, Lemma, ...)

        Parameters
        ----------
        doc : spacy.tokens.doc.Doc
            spacy representation of the text
        index_todel : list of integer 
             positions of each token you want to delete from the document
        list_attr : list, optional
            Contains the Spacy attributes you want to keep (the default is [LOWER, POS, ENT_TYPE, IS_ALPHA, DEP, LEMMA, LOWER, IS_PUNCT, IS_DIGIT, IS_SPACE, IS_STOP])
        Returns
        -------
        spacy.tokens.doc.Doc
            Filter version of doc
        """

        np_array = doc.to_array(list_attr)
        np_array_2 = np.delete(np_array, index_todel, axis = 0)
        index_todel = set(index_todel)
        doc2 = Doc(doc.vocab, words=[t.text for i, t in enumerate(doc) if i not in index_todel])
        doc2.from_array(list_attr, np_array_2)
        return doc2
    
    def get_excluded_index(token):
        if (token.is_punct or token.text in STOP_WORDS or token.pos_ in TOKENS_TO_REMOVE):
            return token.i
    
    Token.set_extension("get_excluded_index", getter=get_excluded_index, force=True)
    
    doc = nlp(text)
    tokens_to_exclude = []
    
    for token in doc:
        exclude_index = token._.get_excluded_index
        if exclude_index:
            tokens_to_exclude.append(exclude_index)

    doc = remove_tokens(doc, tokens_to_exclude)
    
    Token.remove_extension("get_excluded_index")
    
    return doc

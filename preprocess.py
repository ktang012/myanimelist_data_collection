import os
import json
import spacy
import pandas as pd

import utils_data as ud

TOKENS_TO_REMOVE = set(["SPACE", "SYMBOL", "NUM", "X"])
GENRES = set(['Action', 'Adventure', 'Comedy', 'Drama', 'Ecchi',
              'Fantasy','Harem', 'Josei', 'Mystery', 'Romance',
              'School', 'Sci-Fi', 'Seinen','Shoujo', 'Shounen',
              'Slice of Life', 'Supernatural'])

def create_stop_words(animes):
    wc = ud.get_normalized_word_count(animes)
    
    stop_words = set([word[0] for word in wc.most_common(100)])
    diff_stop_words = set(["anime", "season", "good", "not", "episode", "very", "cute",
                           "character", "characters", "main", "story", "plot", "watch", 
                           "series", "love", "art", "pretty", "up", "event"])
    stop_words.difference_update(diff_stop_words)
    return stop_words
    
def tokenize_text(text, nlp, STOP_WORDS, tokens_to_remove=TOKENS_TO_REMOVE):
    text = text.strip().replace("\\n", "")
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

# splits airing date into two fields and drops time info
def reformat_aired(row):
    aired_field = row["aired"]
    aired = aired_field["string"].split("to")
    if len(aired) == 1:
        aired.append(None)
    elif len(aired) == 0:
        aired = [None, None]
    return pd.Series({'air_start': aired[0], 'air_end': aired[1]})

# splits premier date into season and year
def split_premiered(row):
    prem_field = row["premiered"].split(" ")
    
    return pd.Series({"season": prem_field[0], "year": int(prem_field[1])})

# splits studios into two fields, one with studio id and another with studio name
# each field is a list
def reformat_studios(row):
    studios_field = row["studios"]
    studio_names, studio_ids = [], []
    for studio in studios_field:
        studio_names.append(studio['name'])
        studio_ids.append(studio['mal_id'])
    return pd.Series({'studio_names': studio_names, 'studio_ids': studio_ids})

# splits genres into two fields, one with genre id and another with genre name
# each field is a tuple
def reformat_genre(row):
    genres_field = row["genres"]
    genre_ids, genre_names = zip(*[(x['mal_id'], x['name']) for x in genres_field])
    return pd.Series({"genre_names": genre_names, "genre_ids": genre_ids})   

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
    return pd.Series({"has_Prequel": is_prequel, "has_Sequel": is_sequel})

# downloads cover image of anime
def retrieve_image(row):
    dest = os.path.join("data", "cover_image", str(row["mal_id"]) + ".jpg")
    ud.retrieve_image(row["image_url"], dest)

# turns synopsis into word vector (summed up) according to tokenize_text()
def create_synopsis_word_vectors(row, nlp, STOP_WORDS):
    word_vectors = tokenize_text(row["synopsis"], nlp, STOP_WORDS)
    return pd.Series({
        "synopsis_vector": word_vectors.vector
    })

# turn genres into dummy variables
def create_dummy_genres(row, genres):
    genres = set(genres)
    is_genres = row["genre_names"]
    dummy = {}
    for genre in genres:
        if genre in is_genres:
            dummy["is_" + genre] = 1
        else:
            dummy["is_" + genre] = 0

    return pd.Series(dummy)
    
def unroll_document_vector(row, column=None):
    vector_field = row[column]
    series = pd.Series({i: val for i, val in enumerate(vector_field)})
    return series.reindex(sorted(series.keys()))

# the dataframe from reading the json file
def preprocess_df(animes_df, nlp=None, stop_words=None, genres=GENRES):
    # need to reset index when read directly from json
    animes_df = animes_df.reset_index().drop(columns=['index', 'title_synonyms', 
                                          'title_japanese', 'url', 'scores',
                                          'opening_themes', 'ending_themes',
                                          'producers', 'rating', 'status',
                                          'duration', 'episodes',
                                          'title', 'image_url'])
    
    animes_df["title_english"] = animes_df["title_english"].astype(str) 
    
    # airing date -- splits into start and end dates
    # animes_df = animes_df.join(animes_df.apply(reformat_aired, axis=1, 
    #                                            result_type="expand"), how="right")
    # animes_df["air_start"] = pd.to_datetime(animes_df["air_start"],
    #                                         infer_datetime_format=True,
    #                                         errors="coerce")
    # animes_df["air_end"] = pd.to_datetime(animes_df["air_end"], 
    #                                       infer_datetime_format=True,
    #                                       errors="coerce")
    animes_df.drop("aired", axis=1, inplace=True)

    # premiere date -- split into season and year
    animes_df = animes_df.join(animes_df.apply(split_premiered, axis=1,
                                               result_type="expand"), how="right")
    animes_df = pd.get_dummies(animes_df, columns=["season", "year"])
    animes_df.drop("premiered", axis=1, inplace=True)

    # score -- split into quantiles
    quantiles=[.2, .4, .8, .9]
    score_labels=["is_Poor", "is_Below_Average", "is_Average",
                  "is_Above_Average", "is_Excellent"]
    ntiles = [0.] + list(animes_df["score"].quantile(quantiles)) + [10.]
    score_dummies = pd.get_dummies(pd.cut(animes_df["score"], bins=ntiles,
                                   labels=score_labels))
    animes_df = animes_df.join(score_dummies, how="right")
    animes_df.drop("score", axis=1, inplace=True)
    
    # studios of anime -- splits studio to name and id
    # animes_df = animes_df.join(animes_df.apply(reformat_studios, axis=1, 
    #                                            result_type="expand"), how="right")
    animes_df.drop("studios", axis=1, inplace=True)    
    
    # related medias -- checks if anime has a prequel or sequel
    animes_df = animes_df.join(animes_df.apply(create_dummy_related, axis=1, 
                                               result_type="expand"), how="right")
    animes_df.drop("related", axis=1, inplace=True)
    
    # source material -- creates dummy variables for given sources
    source_dummies = pd.get_dummies(animes_df["source"], prefix="is")
    source_dummies.drop(source_dummies.columns.difference(["is_Manga", "is_Light novel",
                                                   "is_Original", "is_Visual novel"]),
                        axis=1, inplace=True)
    animes_df = animes_df.join(source_dummies, how="right")
    animes_df.drop("source", axis=1, inplace=True)
         
    # convert synopsis to word vector (mean of synopsis)
    if nlp and stop_words:
        animes_df = animes_df.join(animes_df.apply(create_synopsis_word_vectors, 
                                   args=(nlp, stop_words), axis=1, result_type="expand"), 
                                   how="right")
        animes_df.drop("synopsis", axis=1, inplace=True)
    else:
        print("Not vectorizing synopsis")
    
    
    # anime genres -- splits genres to genre id and name
    animes_df = animes_df.join(animes_df.apply(reformat_genre, axis=1, 
                                               result_type="expand"), how="right")
    animes_df = animes_df.drop("genres", axis=1)
    # creates dummy variables with given list of genres
    if genres:
        animes_df = animes_df.join(animes_df.apply(create_dummy_genres, args=[genres],
                                                   axis=1, result_type="expand"),
                                   how="right")
        animes_df.drop(labels=["genre_names", "genre_ids"], axis=1, inplace=True)
    else:
        print("Not creating genre dummy variables")
                                                      
    return animes_df                                      


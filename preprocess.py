import os
import json
import spacy
import spacy

import utils_data as ud

def tokenize_text(text, nlp, tokens_to_remove=TOKENS_TO_REMOVE, STOP_WORDS=STOP_WORDS):
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
    

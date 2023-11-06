import pandas as pd
import spacy 
from better_profanity import profanity

old_df = pd.read_csv('/home/karet/Documents/IRE/Project/hatetext-socialmedia/Baseline1/data/hate_norm_with_span.csv')
stacked_sent = pd.concat([old_df['Sentence'], old_df['Normalized_Sentence']])
stacked_sent.reset_index(drop=True, inplace=True)
stacked_inten = pd.concat([old_df['Original_Intensity'], old_df['Normalized_Intensity']])
stacked_inten.reset_index(drop=True, inplace=True)
df = pd.DataFrame({'Sentence': stacked_sent, 'Intensity': stacked_inten})


# Classifiy each sentence to have profanity or not, GitHub: https://github.com/snguyenthanh/better_profanity compares using a list of profane words
# Adding a column to the dataframe called Profanity
df['Profanity'] = df['Sentence'].apply(profanity.contains_profanity)
df['Profanity'] = df['Profanity'].astype(int)


# SVO labeling, adding 3 columns with binary arrays in them indicating the presence of SVO
# For ex: "Cows eat grass"
# Subject:[1, 0, 0]
# Verb:   [0, 1, 0]
# Object: [0, 0, 1]

# Using the Roberta base model tokenizer https://spacy.io/models/en#en_core_web_trf
nlp = spacy.load("en_core_web_trf")

def subject_tag(sentence):
    doc = nlp(sentence)
    return [1 if tok.dep_ == "nsubj" else 0 for tok in doc]

def verb_tag(sentence):
    doc = nlp(sentence)
    return [1 if tok.pos_ == "VERB" else 0 for tok in doc]

def object_tag(sentence):
    doc = nlp(sentence)
    return [1 if tok.dep_ == "dobj" else 0 for tok in doc]

df['Subject'] = df['Sentence'].apply(subject_tag)
df['Verb'] = df['Sentence'].apply(verb_tag)
df['Object'] = df['Sentence'].apply(object_tag)

df.to_csv('hate_int_prof_SVO.tsv', sep='\t', index=False)

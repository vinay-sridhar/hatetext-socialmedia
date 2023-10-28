import pandas as pd
import spacy 
from better_profanity import profanity

df = pd.read_csv('/home/karet/Documents/IRE/Project/hatetext-socialmedia/Baseline1/data/hate_norm_with_span.csv')
df = df.drop(columns=['Normalized_Sentence', 'Normalized_Intensity','Span','spanbio','postags'])

# Classifiy each sentence to have profanity or not, GitHub: https://github.com/snguyenthanh/better_profanity compares using a list of profane words
# Adding a column to the dataframe called Profanity
df['Profanity'] = df['Sentence'].apply(profanity.contains_profanity)
df['Profanity'] = df['Profanity'].astype(int)


# SVO labeling, adding 3 columns with binary arrays in them indicating the presence of SVO
# For ex: "Cows eat grass"
# Subject:[1, 0, 0]
# Verb:   [0, 1, 0]
# Object: [0, 0, 1]
nlp = spacy.load("en_core_web_sm")

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

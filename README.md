# hatetext-socialmedia
Hate intensity prediction for social media text: Project for the course Information Retrieval and Extraction M23 at IIITH

## CODE
### To run 
```
# In python environment
pip install -r requirements.txt
# Then run desired python notebooks
```

### Baseline
**hate_intensity_prediction.ipynb**
> Contains the baseline implementation with *BERT + BiLSTM* architecture

### Advancements
**hate_intensity_prediction_roberta.ipynb**
> Contains experiments with *Roberta* + *BiLSTM, RCNN, BILSTM-RCNN*

**hate_intensity_xlnet.ipynb**
> Contains experiments with finetuning of XLnet
**hate_intensity_ensemble.ipynb**: Contains the ensemble model of the models we experimented with 

### Novelty
1. SVO relative positional encoding + custom Transformer

**SVO_profanity_labelling.py**
> Contains the code for creating the SVO and profanity encodings for all the sentences in the dataset

**hate_int_svo_pytorch.ipynb**
> Contains the custom Hate Language Multi-headed self attention transformer

2. SVO relative positional encoding + MLP fusion

**hate_int_pred_SVOMLP.ipynb**
> Contains the roberta model, with the SVO encoding injected directly to the embeddings

3. Exponential positional encoding 

**hate_intensity_exp_enc.ipynb**
> Contains the addition of the exponential encoding fusion layer upon the baseline BERT model 

Resources to start with:-
1) https://paperswithcode.com/paper/proactively-reducing-the-hate-intensity-of#code
   Has a dataset and code for finding hate phrases and normalizes it into non hate phrase based sentences.
   Also has baselines for hate intensity prediction, might be useful place to begin with.
   Sent request to authors for the dataset procured.

1.1) Hate Norm Gold Dataset found. Request for link on my google drive (Private dataset so not making it public)
https://drive.google.com/drive/folders/1NkoS6c2XhTwWwHccp79t5T1Hs0FddNNF?usp=sharing

2) https://ieeexplore.ieee.org/document/9679052
   Hate intensity prediction for twitter chain replies.



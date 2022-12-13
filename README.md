# Toxic Comment Classification [Algorithm Track]

### Team Members

#### Satyadivya Maddipudi - 016011775

#### Preethi Billa -

## Details

#### Dataset link: https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data

## Contributions

#### Satyadivya Maddipudi

#### Preethi Billa

## About Dataset

- The dataset contains train and test datasets each with 159571 entries. Each dataset has 8 columns i.e 8 features which are described below. The data has a comment and corresponding labels identifying its category based on toxicity category.

        Dataset columns:

        - id: This is comment id referring to the unique id of the comment.
        - comment_text: The comment made by the person in text format.
        - toxic: flag is 1 if said comment is toxic. 0 otherwise.
        - severe_toxic: flag is 1 if said comment is severely toxic. 0 otherwise.
        - obscene: flag is 1 if said comment is obscene. 0 otherwise.
        - threat: flag is 1 if said comment is a threat. 0 otherwise.
        - insult: flag is 1 if said comment is an insult. 0 otherwise.
        - identity_hate: flag is 1 if said comment is identity related hate. 0 otherwise.

## Data Visualizations

## Algorithms used

- Binary Relevance: Binary Relevance is one of the simplest approach to solving multi-label classification problem.

  - Logistic Regression

  - Naive Bais

  - ![](https://github.com/DivyaMaddipudi/255-Final/blob/main/Screenshots/1.png)
    - The results of the Binary Relevance with Logistic Regression and Naive Bayes show Logistic Regression to be performing better.

- Classifier Chain: Similar to Binary Relevance classifier chain is a problem transformation method for multi-label classification. It tries to improve performance over Binary Relevance by taking advantages of labels associations, which is ignored in Binary Relevance.

  - Logistic Regression
  - Naive Bais

  - ![](https://github.com/DivyaMaddipudi/255-Final/blob/main/Screenshots/2.png)

    - Here the resuls show that the embeddings did not perform any better than the bag of words method. The results are almost the same, we can still consider Binary Relevance with Logistic Regression as the best method, considering its simplicity.

- To improve our results we can try modeling with Deep Learning, using frameworks line CNN and LSTM.

- LSTM: LSTM(long short-term memory) is a type of recurrent neural networks(RNN). It is mostly used in problems related to sequence predictions as it can handle the order dependence in the sequence.

  - The first step for data preparation for LSTM is to tokenize the words and represent them as integers, where each unique word is represented by an integer value. We also need to specify the vocabulary size that determines the number of most frequent words to use in the modeling.
  - The results from the LSTM gave improvements over Logistic Regression model for the metrics Accuracy and logloss, but the AUC score is very similar. So, with improvement for two of the metrics, we can still consider LSTM to be performing better than the Logistic Regression Model.

  - The below two graphs show the Loss and AUC Score graph for the LSTM model.

  - ![](https://github.com/DivyaMaddipudi/255-Final/blob/main/Screenshots/3.png)

  - ![](https://github.com/DivyaMaddipudi/255-Final/blob/main/Screenshots/4.png)

- BERT: Transfer learning is a popular approach in deep learning, where a pretrained model is used as the starting point for training a new model in similar task.

  - For BERT, since it is a pretrained model, we will have to process our data according to the process used in the pretrained model. For that, tensorflow hub provides the necessary processing helper and we will simply make use of it.
  - As we see, with transfer learning using BERT, we achieved a higher performance compared to the previous models.

  - The below two graphs show the Loss and AUC Score graph for the BERT model.
  - ![](https://github.com/DivyaMaddipudi/255-Final/blob/main/Screenshots/5.png)
  - ![](https://github.com/DivyaMaddipudi/255-Final/blob/main/Screenshots/6.png)

- Finally, With transfer learning using BERT model, we see we achieve better results , with improvements in Accuracy, AUC and LogLoss.

## References

1. Hochreiter, Sepp & Schmidhuber, Jürgen. (1997). Long
   Short-term Memory. Neural computation. 9. 1735-80.
   10.1162/neco.1997.9.8.1735.
2. Devlin, Jacob & Chang, Ming-Wei & Lee, Kenton &
   Toutanova, Kristina. (2018). BERT: Pre-training of
   Deep Bidirectional Transformers for Language
   Understanding.
3. https://colah.github.io/posts/2015-08-Understanding-LS
   TMs/
4. Zaheri, Sara; Leath, Jeff; and Stroud, David (2020)
   "Toxic Comment Classification," SMU Data Science
   Review: Vol. 3: No. 1, Article 13.
5. Androcec, Darko. (2020). Machine learning methods
   for toxic comment classification: a systematic review.
   Acta Universitatis Sapientiae, Informatica. 12. 205-216.
   10.2478/ausi-2020-0012.

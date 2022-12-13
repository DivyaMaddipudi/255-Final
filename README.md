# CMPE255-FinalProject

## Team5 Toxic Comment Detection[Algorithmic Track]

- 1. Preethi Billa - 015920411
- 2. Satyadivya Maddipudi - 016011775

## Details

- 1. Dataset: https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data
- 2. _Colab notebook for Preethi's Contribution[file is included in github as well]:
     https://drive.google.com/file/d/1lrhtURqIUjAUmMcRcI4YjHoRjwLz9ZXU/view?usp=share_link_
- 3. Colab notebook for Satyadivya's Contribution[file included in github as well]:https://drive.google.com/file/d/1A7K8ROt8PI0F5HySwT-aqOoQqDj3ocX7/view?usp=share_link
- 4. Video demo: https://drive.google.com/file/d/1gx-R-TUN2kPPdQdMyJUDdcyiqtn07jGK/view?usp=share_link

## Contributors

### Satyadivya's Contribution

| Module                             | Contribution briefing                                                                                                                                                                                                                                              |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| I Data - A and B                   | Gathering the dataset from kaggle and extracting the data from a zip file and converting it into csv file to make the training process easy.                                                                                                                       |
| II Data Cleaning and Preprocessing | Has performed the preprocessing using an approach including tasks [Null value analysis, Category counts,Feature extraction , Stop words removal,Lemmatization,Special characters removal, Miscellaneous processing steps] as described in the respective section.. |
| III Data Visualisation             | Performed a visualization for Number of sentences in each comment, Absolute word count and unique word count, Percentage of unique words of total words in comment,Count of comments with low unique words,Correlation heatmap.                                    |
| IV Experiment and Analysis         | Developed LSTM,BERT,Naive Bayes                                                                                                                                                                                                                                    |
| V Comparing the models             | Compared SVM,Logistic Regression, KNN Model,LSTM,BERT,Naive Bayes                                                                                                                                                                                                  |

### Preethi's Contribution

| Module                             | Contribution briefing                                                                                                                                                                                                                                           |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| I Data - A and B                   | Gathering the dataset from kaggle and extracting the data from a zip file and converting it into csv file to make the training process easy                                                                                                                     |
| II Data Cleaning and Preprocessing | Has performed the preprocessing using an approach including tasks [Null value analysis, Category counts,Feature extraction , Stop words removal,Lemmatization,Special character removal, Miscellaneous processing steps] as described in the respective section |
| III Data Visualisation             | Performed a visualization for the number of sentences in each comment, Absolute word count and unique word count, Percentage of unique words of total words in comment,Count of comments with low unique words,Correlation heatmap.                             |
| IV Experiment and Analysis         | Developed SVM,Logistic Regression, KNN Model.                                                                                                                                                                                                                   |
| V Comparing the models             | Compared SVM,Logistic Regression, KNN Model,LSTM,BERT,Naive Bayes                                                                                                                                                                                               |

## About Dataset

- The dataset contains train and test datasets each with 159571 entries.
- Each dataset has 8 columns i.e 8 features which are described below.
- The data has a comment and corresponding labels identifying its category based on toxicity category.

### Dataset Description

- id: This is comment id referring to the unique id of the comment.
- comment_text: The comment made by the person in text format.
- toxic: flag is 1 if said comment is toxic. 0 otherwise.
- severe_toxic: flag is 1 if said comment is severely toxic. 0 otherwise.
- obscene: flag is 1 if said comment is obscene. 0 otherwise.
- threat: flag is 1 if said comment is a threat. 0 otherwise.
- insult: flag is 1 if said comment is an insult. 0 otherwise.
- identity_hate: flag is 1 if said comment is identity related hate. 0 otherwise.

## Data Preprocessing

### Null Value Analysis

We have checked that our data doesn’t have any missing values to fill in train and test data.

<img width="262" alt="image" src="https://user-images.githubusercontent.com/55958864/207703583-2132e5f4-4c3d-41c3-b6b5-7850a2c2e914.png">

### Category counts

We have checked the category counts of each category and value counts of the same in the dataset.

<img width="250" alt="image" src="https://user-images.githubusercontent.com/55958864/207703845-aae13f76-196a-49b3-b000-7127cc042a96.png">

### Stop words Analysis

Stop words removal is important in NLP as it helps in reducing the noise present in the data. stop words carry little meaning in the data such as ‘the’, ‘and’, ‘an’ etc., and does not contribute much to the model performance.

<img width="250" alt="image" src="https://user-images.githubusercontent.com/55958864/207703924-fcb9d16d-5e6d-4372-8cfe-abc3a9096b6a.png">

## Data Visualization

### Number of sentences in each comment

<img width="316" alt="image" src="https://user-images.githubusercontent.com/55958864/207704291-615ddda5-3394-4dce-8bf3-0b8005770cb4.png">

### Number of words in each comment

<img width="354" alt="image" src="https://user-images.githubusercontent.com/55958864/207704383-ff10d980-bc54-43f3-9397-70dcc71902b6.png">

### Absolute wordcount and unique words count

<img width="320" alt="image" src="https://user-images.githubusercontent.com/55958864/207704505-11846c5b-e776-4e4c-943d-e62898153892.png">

### Percentage of unique words of total words in comment

<img width="342" alt="image" src="https://user-images.githubusercontent.com/55958864/207704584-9ba90a5d-6d3d-4959-aa36-7a1b0a1562be.png">

### Count of comments with low unique words

<img width="361" alt="image" src="https://user-images.githubusercontent.com/55958864/207704637-078f2336-7181-4e92-bf25-25b93f58583e.png">

### Density vs length of texts

<img width="332" alt="image" src="https://user-images.githubusercontent.com/55958864/207704759-fb4a6057-a7d8-4697-ae34-b9874e0332e7.png">

### Sum harmful

<img width="333" alt="image" src="https://user-images.githubusercontent.com/55958864/207704884-d27201b8-a80d-4f1f-9729-21dfd45b0de1.png">

### Correlation heatmap

<img width="320" alt="image" src="https://user-images.githubusercontent.com/55958864/207705375-abc36b0b-3ea1-4dde-a7b6-cf8dfed07970.png">

### TF-IDF Plots per class

To offer the text for the modeling using TFIDF vectorization, we will first employ the bag of words format. Using a technique known as "bag of words," which extracts features from text documents (unique words that appear in every text document), it is common practice to classify texts or documents. The presence of words (or the document's word count) is then shown for each document by its features. TFIDF vectorization represents a score that illustrates how pertinent a word is to a document across all documents rather than just a simple word count.

The below graphs show the visualisations based on the class after TF-IDF Vectorization.

#### Toxic

<img width="318" alt="image" src="https://user-images.githubusercontent.com/55958864/207705921-cca34416-f655-40c5-8c6f-e04aa4a6066e.png">

#### Severe Toxic

<img width="323" alt="image" src="https://user-images.githubusercontent.com/55958864/207706067-8ed49c7d-a38b-4dff-8d09-bddb1f3b804f.png">

#### Obscene

<img width="313" alt="image" src="https://user-images.githubusercontent.com/55958864/207706203-36ebc172-9612-44a4-915b-f92d0662c83b.png">

#### Threat

<img width="331" alt="image" src="https://user-images.githubusercontent.com/55958864/207706310-095a9637-77d7-4296-8b3f-c97d48173a3d.png">

#### Identity hate

<img width="338" alt="image" src="https://user-images.githubusercontent.com/55958864/207706416-c01e8279-79c7-43a8-8975-e551070b0f3b.png">

#### Insult

<img width="334" alt="image" src="https://user-images.githubusercontent.com/55958864/207706527-a500e9a5-5af6-4c8a-8180-63f56e26edb4.png">

#### Clean

<img width="346" alt="image" src="https://user-images.githubusercontent.com/55958864/207706606-f0d720b6-12be-4580-9a0a-7a207c77acac.png">

## Data Modeling

### A. Logistic Regression

Here we have performed logistic regression on the dataset with and without repetitions. And also we have calculated the accuracies on all the six different classes before and after repetition removal. The below figure shows the results for logistic regression.

<img width="321" alt="image" src="https://user-images.githubusercontent.com/55958864/207706760-420ef25a-0570-4136-9053-a658b8784a80.png">

### B. Naive Bayes

Here the below figure shows the accuracies, AUC’s, log loss of all logistic regression and Multinomial DB. Here we can see that Classifier chains did not increase the model's performance, yet we still see that Logistic Regression outperforms Naive Bayes. We previously observed that there was no significant correlation between the labels, and chain classifiers perform better when there are associations between the labels, therefore it is possible that the performance with chain classifiers has not increased.

<img width="327" alt="image" src="https://user-images.githubusercontent.com/55958864/207707035-ed634255-be6f-412f-beba-c2d1477cc7b2.png">

### C. KNN Algorithm

We have chosen the K as 3. We have performed KNN on the dataset with and without repetitions similar to the previous models. And also we have calculated the accuracies on all the six different classes before and after repetition removal. The below figure shows the results for KNN algorithm

<img width="331" alt="image" src="https://user-images.githubusercontent.com/55958864/207707132-ddda839b-517b-4f6a-9416-7ff51349e34b.png">

### D. SVM Algorithm

We have done Standard scalar with with_mean being false and gamma parameter as auto. We have performed SVM Algorithm on the dataset with and without repetitions similar to the previous models. And also we have calculated the accuracies on all the six different classes before and after repetition removal. The below figure shows the results for the SVM algorithm.

<img width="322" alt="image" src="https://user-images.githubusercontent.com/55958864/207707229-c2d93702-edb7-4fa6-a3a2-c2caed00655e.png">

### E. LSTM

- The complete accuracy which we have achieved for LSTM is 98.9 which is better when we compare it to other algorithms which we have done previously.

- The below two graphs show the Loss and AUC Score graph for the LSTM model.Accuracy and logloss metrics from the LSTM model performed better than those from the Logistic Regression model, although the AUC score was nearly identical. So even if two of the measures have improved, we can still say that the LSTM is outperforming the Logistic Regression Model.

<img width="308" alt="image" src="https://user-images.githubusercontent.com/55958864/207707470-81fecd67-9f14-488f-9dfa-b97cf9b35ca2.png">

<img width="317" alt="image" src="https://user-images.githubusercontent.com/55958864/207707516-8f4fca2f-a9b7-4601-99f9-7129302597d4.png">

### F. Transfer learning with BERT

- We must process our data in accordance with the pretrained model's method since it is a pretrained model. Tensorflow Hub offers the required processing assist for that, so we will just utilize it. The routes to the BERT model and its text processing handler have been specified. Now we can build and train our model. In comparison to the earlier models, we got a greater performance with transfer learning utilizing BERT.

<img width="426" alt="image" src="https://user-images.githubusercontent.com/55958864/207707792-960ea7af-fdb9-4776-b781-e34c804ef25f.png">

<img width="418" alt="image" src="https://user-images.githubusercontent.com/55958864/207707913-8a33d9e2-613f-416e-91f9-e4cf39bf3d4e.png">

## Conclusion

We have covered different modeling approaches to solve the multi-label classification problem. We have performed different models ie., logistic regression, Naive bayes, Support Vector Machines, KNN, LSTM and transfer learning with BERT. Even though all the models' accuracies are greater than 90, BERT is more reliable on the unseen data based on the results seen by your modeling. By far, we saw the BERT model performing the best with the mean AUC ROC score of 0.978 on the test data.

## References

1. Hochreiter, Sepp & Schmidhuber, Jürgen. (1997). Long Short-term Memory. Neural computation. 9. 1735-80. 10.1162/neco.1997.9.8.1735.
2. Devlin, Jacob & Chang, Ming-Wei & Lee, Kenton & Toutanova, Kristina. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
3. https://colah.github.io/posts/2015-08-Understanding-LSTMs/
4. aheri, Sara; Leath, Jeff; and Stroud, David (2020) "Toxic Comment Classification," SMU Data Science Review: Vol. 3: No. 1, Article 13.
5. Androcec, Darko. (2020). Machine learning methods for toxic comment classification: a systematic review. Acta Universitatis Sapientiae, Informatica. 12. 205-216. 10.2478/ausi-2020-0012.

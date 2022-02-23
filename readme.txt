Annotating Privacy Policies in the Sharing Economy

This repository contains the replication package for RE'22 paper.

Dataset.xlsx: This Excel sheet contains all the information about our dataset including, 1) selected apps and their categories, average ratings on app stores, number of installments 2) Extracted privacy policy statements and the results of manual annotation of these statements into NFRs 3) different types of data each of these apps collect, track, and link to users identity aacording to the Apple App Store Privacy card section 4) The results of examining privacy policies of these parts and the included sections in each policy

Data_multilabel.csv: This file is the result of our manual annotation which then fed into our multi-label classification algorithm.

Script.py: This file contains the main Python methods we implemented in order to pre-process and classify the extracted policy statements into NFRs. Here is the content of this file:

loading the data file (Data_multilabel.csv)
loading the embedding model (we used the pre-trained model of GloVe)
Pre-processing the text by removing non-ascii tokens, stopwords, and lemmatization
Vectorizing each policy statement using the embedding model
Classifying statements and report the results. We used multiple classifers and report the best results in the paper (SVM).
Classifiying the statements using VSM and report the results.
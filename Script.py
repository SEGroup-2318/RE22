df = pd.read_csv("Data_multilabel.csv", encoding = "ISO-8859-1")

stopwords = nltk.corpus.stopwords.words('english')
lemma = nltk.stem.WordNetLemmatizer()
p_stemmer = nltk.stem.porter.PorterStemmer()

glove_model = KeyedVectors.load_word2vec_format('glove_path', binary=True)

def remove_ascii_words(df):
    non_ascii_words = []
    for i in range(len(df)):
        for word in df.loc[i, 'Statement'].split(' '):
            if len(word)< 3 and len(word) > 15:
                if any([ord(character) >= 128 for character in word]):
                    non_ascii_words.append(word)
                    df.loc[i, 'Statement'] = df.loc[i, 'Statement'].replace(word, '')
    return non_ascii_words

def get_good_tokens(sentence):
    replaced_punctation = list(map(lambda token: re.sub('[^A-Za-z]+', '', token), sentence))
    removed_punctation = list(filter(lambda token: token, replaced_punctation))
    return removed_punctation

def lda_get_good_tokens(df):
    df['Statement'] = df.Statement.str.lower()
    df['tokenized_Description'] = list(map(nltk.word_tokenize, df.Statement))
    df['tokenized_Description'] = list(map(get_good_tokens, df.tokenized_Description))

def remove_stopwords(df):
    df['stopwords_removed'] = list(map(lambda doc:
                                       [word for word in doc if word not in stopwords],
                                       df['tokenized_Description']))
def stem_words(df):
    df['lemmatized_text'] = list(map(lambda sentence:list(map(lemma.lemmatize, sentence)),df.stopwords_removed))
    df['stemmed_text'] = list(map(lambda sentence:
                                  list(map(p_stemmer.stem, sentence)),
                                  df.lemmatized_text))
#Make a BOW for every document
def document_to_bow(df):
    dictionary = Dictionary(documents=df.stemmed_text.values)
    dictionary.filter_extremes(no_above=0.8, no_below=3)
    df['bow'] = list(map(lambda doc: dictionary.doc2bow(doc), df.stemmed_text))
    
def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)
    
def glove_preprocessing(df):
    remove_ascii_words(df)

    df['Statement'] = df.Statement.str.lower()
    df['Statement'] = df['Statement'].replace(r'http\S+', '', regex=True).replace(r'www\S+', '', regex=True)
    df['document_sentences'] = df.Statement.str.split('.')  # split texts into individual sentences
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(map(nltk.word_tokenize, sentences)),
                                         df.document_sentences))  # tokenize sentences
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(map(get_good_tokens, sentences)),
                                         df.tokenized_sentences))  # remove unwanted characters
    df['tokenized_sentences'] = list(map(lambda sentences:
                                         list(filter(lambda lst: lst, sentences)),
                                         df.tokenized_sentences))  # remove empty lists
    df['stop']= ''
    for i in range(len(df)):
        df.stop[i] = list(map(lambda doc: [word for word in doc if word not in stopwords], df.loc[i, 'tokenized_sentences']))
    df['lemm'] = ''
    for i in range(len(df)):
         df.lemm[i] = list(map(lambda sentence:list(map(lemma.lemmatize, sentence)), df.loc[i,'stop']))
    df['stem'] = ''
    for i in range(len(df)):
        df.stem[i] = list(map(lambda sentence: list(map(p_stemmer.stem, sentence)), df.loc[i,'lemm']))


glove_not_included = []
def get_glove_features(glove_model, sentence_group):
    words = np.concatenate(sentence_group)  # words in text
    index2word_set = set(glove_model.wv.vocab.keys())  # words known to model
    
    featureVec = np.zeros(glove_model.vector_size, dtype="float32")
    nwords = 0
    for word in words:
        if word in index2word_set and len(word) > 2: 
            featureVec = np.add(featureVec, glove_model[word])
            nwords += 1.
        if word not in index2word_set:
            glove_not_included.append(word)
    if nwords > 0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec


def Classify(classifier, x_train, y_train, x_test, y_test):
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, predictions)
    cm = metrics.confusion_matrix(y_test, predictions)
    return classification_report(y_test, predictions, output_dict=True)
	
def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float( len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)
def Hamming_Loss(y_true, y_pred):
    temp=0
    for i in range(y_true.shape[0]):
        temp += np.size(y_true[i] == y_pred[i]) - np.count_nonzero(y_true[i] == y_pred[i])
    return temp/(y_true.shape[0] * y_true.shape[1])
	
nb = GaussianNB()
ada = AdaBoostClassifier(n_estimators=100, learning_rate=1)
rf = RandomForestClassifier(n_estimators=100)
knn = KNeighborsClassifier(n_neighbors=2)
svm = SVC(kernel='linear')
dt =  DecisionTreeClassifier()
logisticRegr = LogisticRegression()
Naive = naive_bayes.MultinomialNB()



#running
glove_preprocessing(df)
df['glove_features'] = list(map(lambda sen_group: get_glove_features(glove_model, sen_group), df.lemm))

# Report Glove Results
for category in categories:
    subset = []
    hamming_l = []
    hamming_s = []
    precision = []
    recall = []
    f1 = []
    
    printmd('**Processing {} comments...**'.format(category))
    for i in range(10):
        rand = random.randint(1, 100)
        train, test = train_test_split(df, random_state=rand, test_size=0.30, shuffle=True)
        x_train = np.array(list(map(np.array, train.glove_features)))
        x_test = np.array(list(map(np.array, test.glove_features)))
        y_train = train.drop(labels = ['Id','Statement'], axis=1)
        y_test = test.drop(labels = ['Id','Statement'], axis=1)
        

        pipeline = Pipeline([('clf', OneVsRestClassifier(svm, n_jobs=-1)),])
        
        # Training model on train data
        pipeline.fit(x_train, train[category])

        # calculating test accuracy
        prediction = pipeline.predict(x_test)
        
        subset.append(accuracy_score(test[category], prediction))
        hamming_l.append(hamming_loss(test[category], prediction))
        hamming_s.append(hamming_score(test[category].values, prediction))
        precision.append(skm.precision_score(test[category].values, prediction,average='macro'))
        recall.append(skm.recall_score(test[category].values, prediction,average='macro'))
        f1.append(skm.f1_score(test[category].values, prediction,average='macro'))
        
    print('Subset accuracy is {}'.format(np.average(subset)))
    print('Hamming loss: {0}'.format(np.average(hamming_l))) 
    print('Hamming score: {0}'.format(np.average(hamming_s)))
    print('precision: {0}'.format(np.average(precision)))
    print('recall: {0}'.format(np.average(recall)))
    print('f1_score: {0}'.format(np.average(f1)))
    print("\n")

#Report VSM results	
for category in categories:
    subset = []
    hamming_l = []
    hamming_s = []
    precision = []
    recall = []
    f1 = []
    
    printmd('**Processing {} comments...**'.format(category))
    for i in range(10):
        rand = random.randint(1, 100)
        train, test = train_test_split(df, random_state=rand, test_size=0.30, shuffle=True)
        train_text = train['Statement']
        test_text = test['Statement']

        vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
        vectorizer.fit(train_text)
        vectorizer.fit(test_text)

        x_train = vectorizer.transform(train_text)
        y_train = train.drop(labels = ['Id','Statement'], axis=1)

        x_test = vectorizer.transform(test_text)
        y_test = test.drop(labels = ['Id','Statement'], axis=1)

        pipeline = Pipeline([('clf', OneVsRestClassifier(svm, n_jobs=-1)),])
        
        # Training model on train data
        pipeline.fit(x_train, train[category])

        # calculating test accuracy
        prediction = pipeline.predict(x_test)
        
        subset.append(accuracy_score(test[category], prediction))
        hamming_l.append(hamming_loss(test[category], prediction))
        hamming_s.append(hamming_score(test[category].values, prediction))
        precision.append(skm.precision_score(test[category].values, prediction,average='macro'))
        recall.append(skm.recall_score(test[category].values, prediction,average='macro'))
        f1.append(skm.f1_score(test[category].values, prediction,average='macro'))
        
    print('Subset accuracy is {}'.format(np.average(subset)))
    print('Hamming loss: {0}'.format(np.average(hamming_l))) 
    print('Hamming score: {0}'.format(np.average(hamming_s)))
    print('precision: {0}'.format(np.average(precision)))
    print('recall: {0}'.format(np.average(recall)))
    print('f1_score: {0}'.format(np.average(f1)))
    print("\n")
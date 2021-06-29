from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('always')

def load_csv_file():
    input_file = "./dataset_clean.csv"
    df = pd.read_csv(input_file, header = 0)
    original_headers = list(df.columns.values)
    #print(df.head())
    return df
    #numpy_array = df.head(10).as_matrix()
    #print(numpy_array)
    #data =  np.loadtxt(input_file = f, delimiter = ',')
df = load_csv_file()

#Prepate Data Set to ML Pipeline



#Perform a Feature Selection


df["Keywords"] = df["Description"] + ' ' + df["Keywords"]
col = ['Description', 'Website', 'Keywords']
df = df[col]
df.columns = ['Description', 'Website', 'Keywords']
df['category_id'] = df['Website'].factorize()[0]
category_id_df = df[['Website', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Website']].values)
keywords = df['Keywords'].str.split(' ').apply(pd.Series, 1).stack()
print(df.head())


keywords.index = keywords.index.droplevel(-1)
keywords.name = 'Keywords'
del df['Keywords']
df = df.join(keywords)
df = df.drop_duplicates()
print(df.head())

df.to_csv('dataset_by_keywords.csv', index=True)


#Show Imbalanced Classes

df.groupby('category_id').Keywords.count().plot.bar(ylim=0)
plt.show()

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.Keywords).toarray()
labels = df.category_id
print(features.shape)


#####################   MODEL COMPARATION #######################

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
]
CV = 10
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
  model_name = model.__class__.__name__
  accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
  for fold_idx, accuracy in enumerate(accuracies):
    entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
import seaborn as sns
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()

cv_df.groupby('model_name').accuracy.mean()

model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.50, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import confusion_matrix
conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.category_id.values, yticklabels=category_id_df.category_id.values)
plt.ylabel('Actual')
plt.xlabel('Predicted LinearSVC')
plt.show()

#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(df['Website'])
#X_train_counts.shape
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#X_train_tfidf.shape

X_train, X_test, y_train, y_test = train_test_split(df['Keywords'], df['category_id'], random_state = 0)
#X_train, X_test, y_train, y_test = train_test_split(df['Keywords'], df['category_id'], random_state = 0)

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

y_pred = clf.predict(count_vect.transform(X_test))

print("## SCORE MultinomialNB", clf.score(count_vect.transform(X_test), y_test))

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10,10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.category_id.values, yticklabels=category_id_df.category_id.values)
plt.ylabel('Actual')
plt.xlabel('Predicted MultinomialNB')
plt.show()

print("\n\n\n\n\n### Predictions MultinomialNB")
print("i want to download:: ", clf.predict(count_vect.transform(["i want to download"])))
print("waze traffic::", clf.predict(count_vect.transform(["waze traffic"])))

###### BEST KEYWORD CORELATION  ######

N = 10
for Website, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(Website))
  print(N, " Most correlated unigrams:: {}".format(', '.join(unigrams[-N:])))
  print(N, " Most correlated bigrams::  {}".format(', '.join(bigrams[-N:])))



##### FEATURE SELECTION   ####

vectorizer = TfidfVectorizer(min_df= 3, stop_words="english", sublinear_tf=True, norm='l2', ngram_range=(1, 2))

X = df['Keywords']
Y = df['category_id']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25)
pipeline = Pipeline([('vect', vectorizer),
                     ('chi',  SelectKBest(chi2, k=10)),
                     ('clf', RandomForestClassifier())])


model = pipeline.fit(X_train, y_train)
ytest = np.array(y_test)
print(classification_report(ytest, model.predict(X_test)))
print(confusion_matrix(ytest, model.predict(X_test)))

# -*- coding: utf-8 -*-
"""updated_smallproject_1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12DIc9wgBtpL6d6GTuHcrr8-7bYfiF4jn
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import HashingTF, IDF
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression, NaiveBayes, RandomForestClassifier, GBTClassifier
from pyspark.sql.functions import col

from xgboost.spark import SparkXGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

spark = SparkSession.builder.appName("SpamClassifierWithPlots").getOrCreate()

sms_data = spark.read.csv('/content/spam.csv' , header=True, inferSchema=True)

sms_data.show(5)

sms_data.columns

# Drop the specified columns
sms_data = sms_data.drop('_c2', '_c3', '_c4')

sms_data.show(5)

sms_data = sms_data.withColumnRenamed('v1', 'output').withColumnRenamed('v2', 'input')

sms_data.columns

sms_data.groupBy('output').count().show()

import nltk
nltk.download('stopwords')

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.functions import split
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#spark = SparkSession.builder.appName("TextProcessing").getOrCreate()

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def cleaned_data(text):
    if text is None or text == '':
        return ''
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    text = [ps.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

sms_data = sms_data.withColumn("input", col("input").cast("string"))

clean_text_udf = udf(cleaned_data, StringType())

data = sms_data.withColumn("cleaned_input", clean_text_udf(col("input")))

data.select('input', 'cleaned_input').show(5, truncate=False)

data = data.withColumn('words', split(data.cleaned_input, ' '))

bagofwords_vectors = CountVectorizer(inputCol='words', outputCol='features', vocabSize=10000, minDF=5.0)
vectorized_data = bagofwords_vectors.fit(data).transform(data)

vectorized_data.select('input', 'cleaned_input', 'features').show(5, truncate=False)

from pyspark.sql.functions import when
vectorized_data = vectorized_data.withColumn("label", when(col("output") == "spam", 1).otherwise(0))

response_var = vectorized_data.select("features").head()[0].toArray()

print(response_var )

from pyspark.sql.functions import col
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.sql import functions as F
from sklearn.neighbors import KNeighborsClassifier
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, auc
from sklearn.model_selection import train_test_split
import pandas as pd
train_data, test_data = vectorized_data.randomSplit([0.8, 0.2], seed=42)

#Multinomial model
multinomial_nb = NaiveBayes(featuresCol='features', labelCol='label', modelType="multinomial")
nb_model = multinomial_nb.fit(train_data)

test_predict = nb_model.transform(test_data)

y_pred = test_predict.select("prediction", "label").rdd.map(lambda row: (row[0], row[1])).collect()
y_pred_labels, y_true = zip(*y_pred)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()

acc = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn)
specificity = tn / (tn + fp)

print("Classification Report")
print(f"Accuracy: {acc:.3f}")
print(f"Sensitivity (Recall): {recall:.3f}")
print(f"Specificity: {specificity:.3f}")

# Calculate probabilities for ROC curve
y_probabilites = test_predict.select("probability").rdd.map(lambda r: r[0][1]).collect()
fpr, tpr, thresholds = roc_curve(y_true, y_probabilites)
roc_auc_data = roc_auc_score(y_true, y_probabilites)

print(f"AUC: {roc_auc_data:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_data:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve for SMS Spam Classificatin')
plt.legend()
plt.show()

model_rf = RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=100, maxDepth=10, seed=42)
model_rf = model_rf.fit(train_data)

test_predicts = model_rf.transform(test_data)

y_pred = test_predicts.select("prediction", "label").rdd.map(lambda r: (r[0], r[1])).collect()
y_pred_labels, y_true = zip(*y_pred)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()

accu = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn)
specificity = tn / (tn + fp)

print("Classification Report")
print(f"Accuracy: {accu:.4f}")
print(f"Sensitivity (Recall): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")


y_probabilites = test_predicts.select("rawPrediction").rdd.map(lambda r: r[0][1] / (r[0][0] + r[0][1])).collect()
roc_auc_data = roc_auc_score(y_true, y_probabilites)
print(f"AUC: {roc_auc_data:.4f}")

fpr, tpr, thresholds = roc_curve(y_true, y_probabilites)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_data:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve for SMS Spam Classification t')
plt.legend()
plt.show()

model_svc = LinearSVC(featuresCol='features', labelCol='label', maxIter=100, regParam=0.1)
model_svc = model_svc.fit(train_data)

test_predicts = model_svc.transform(test_data)

y_pred = test_predicts.select("prediction", "label").rdd.map(lambda r: (r[0], r[1])).collect()
y_pred_labels, y_true = zip(*y_pred)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()

acc = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn)
specificity = tn / (tn + fp)

print("Classification Report")
print(f"Accuracy: {acc:.4f}")
print(f"Sensitivity (Recall): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")

y_raw_data = test_predicts.select("rawPrediction").rdd.map(lambda r: r[0][1]).collect()
roc_auc_data = roc_auc_score(y_true, y_raw_data)
print(f"AUC: {roc_auc_data:.4f}")

fpr, tpr, thresholds = roc_curve(y_true, y_raw_data)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_data:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve for SMS Spam Classification ')
plt.legend()
plt.show()

vectorized_data_pd = vectorized_data.select("features", "output").toPandas()
X = vectorized_data_pd['features'].apply(lambda x: x.toArray()).to_list()
X = pd.DataFrame(X)
y = vectorized_data_pd['output'].apply(lambda x: 1 if x == "spam" else 0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# Predict
y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]

# Evaluation
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
acc = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn)
specificity = tn / (tn + fp)
roc_auc = roc_auc_score(y_test, y_proba)

print("Classification Report for XGBoost")
print(f"Accuracy: {acc:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"AUC: {roc_auc:.4f}")
print("\nDetailed Report:\n", classification_report(y_test, y_pred))

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'XGBoost ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for XGBoost Spam Classification')
plt.legend()
plt.grid(True)
plt.show()

















vectorized_data_pd = vectorized_data.select("features", "output").toPandas()
X = vectorized_data_pd['features'].apply(lambda x: x.toArray()).to_list()
X = pd.DataFrame(X)
y = vectorized_data_pd['output'].apply(lambda x: 1 if x == "spam" else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Calculate metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)  # Recall or True Positive Rate
specificity = tn / (tn + fp)

# Print metrics
print("Classification Report")
print(f"Accuracy: {accuracy:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve for SMS Spam Classification using KNN')
plt.legend()
plt.show()

model_lr = LogisticRegression(featuresCol='features', labelCol='label', maxIter=100, regParam=0.1)
model_lr = model_lr.fit(train_data)

test_predicts = model_lr.transform(test_data)

y_pred = test_predicts.select("prediction", "label").rdd.map(lambda r: (r[0], r[1])).collect()
y_pred_labels, y_true = zip(*y_pred)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels).ravel()

accu = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn)
specificity = tn / (tn + fp)

print("Classification Report")
print(f"Accuracy: {accu:.4f}")
print(f"Sensitivity (Recall): {recall:.4f}")
print(f"Specificity: {specificity:.4f}")

y_raw_data = test_predicts.select("rawPrediction").rdd.map(lambda r: r[0][1]).collect()
roc_auc_data = roc_auc_score(y_true, y_raw_data)
print(f"AUC: {roc_auc_data:.4f}")

fpr, tpr, thresholds = roc_curve(y_true, y_raw_data)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_data:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve for SMS Spam Classification')
plt.legend()
plt.show()


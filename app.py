from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np

# import HashingVectorizer from local dir
# app.py 상단에 추가
from vectorizer import vect

app = Flask(__name__)

# Preparing the Classifier
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'spam_classifier_model.pkl'), 'rb'))
vect = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'vectorizer.pkl'), 'rb'))

# db setting
db = os.path.join(cur_dir, 'emails.sqlite')

def classify(document):
  # 스팸/햄을 분류
    label = {0: 'Ham', 1: 'Spam'}
    X = vect.transform([document])  # 벡터화된 텍스트
    y = clf.predict(X)[0]  # 예측 결과
    proba = np.max(clf.predict_proba(X))  # 예측 확률
    return label[y], proba

def train(document, y):
  X = vect.transform([document])
  clf.partial_fit(X, [y])

def sqlite_entry(path, document, y):
  conn = sqlite3.connect(path)
  c = conn.cursor()
  c.execute("INSERT INTO email_db (email_content, label, date) VALUES (?, ?, DATETIME('now'))", (document, y))

  conn.commit()
  conn.close()

# Flask
class EmailForm(Form):
  emailcontent = TextAreaField('',
                              [validators.DataRequired(),
                               validators.length(min=15)])

@app.route('/')
def index():
  form = EmailForm(request.form)
  return render_template('emailform.html', form=form)

@app.route('/results', methods=['POST'])
def results():
  form = EmailForm(request.form)
  if request.method == 'POST' and form.validate():
    email_content = request.form['emailcontent']
    y, proba = classify(email_content)
    return render_template('results.html', 
                           content=email_content,
                           prediction = y,
                           probability=round(proba*100, 2))
  return render_template('reviewerform.html', form=form)

@app.route('/thanks', methods=['POST'])
def feedback():
  feedback = request.form['feedback_button']
  email_content = request.form['emailcontent']
  prediction = request.form['prediction']
  inv_label = {'Ham': 0, 'Spam': 1}
  y = inv_label[prediction]

  if feedback == 'Incorrect':
    y = int(not(y))

  train(email_content, y)
  sqlite_entry(db, email_content, y)
  return render_template('thanks.html')


if __name__ == '__main__':
  app.run(debug=True)
from sklearn.feature_extraction.text import CountVectorizer

import re
import os
import pickle

cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(
                os.path.join(cur_dir,
                'pkl_objects',
                'vectorizer.pkl'), 'rb'))

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)

    # 그냥 텍스트만 사용하도록 설정
    text = re.sub('[\W_]+', ' ', text)

    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

# CountVectorizer로 텍스트 데이터를 벡터화
vect = CountVectorizer(decode_error='ignore',
                       stop_words=None,  # 이미 토크나이저에서 stopwords를 처리하므로 None으로 설정
                       tokenizer=tokenizer)

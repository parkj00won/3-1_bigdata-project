from datasets import load_dataset
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import joblib
import os

# 1. 데이터 불러오기
print("데이터 불러오기 시작")
dataset = load_dataset("dair-ai/emotion", "split")

train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

print("train 데이터 크기:", train_df.shape)
print("test 데이터 크기:", test_df.shape)

# 2. 라벨 이름 정의
label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# 3. 입력값과 정답 분리
X_train = train_df["text"]
y_train = train_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]

# 4. 텍스트를 숫자로 변환
print("TF-IDF 변환 시작")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("TF-IDF 변환 완료")
print("학습 데이터 벡터 크기:", X_train_tfidf.shape)
print("테스트 데이터 벡터 크기:", X_test_tfidf.shape)

# 5. 모델 학습
print("모델 학습 시작")

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

print("모델 학습 완료")

# 6. 예측 및 평가
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)

print("\n모델 정확도")
print(accuracy)

print("\n분류 성능 리포트")
print(classification_report(
    y_test,
    y_pred,
    target_names=list(label_map.values())
))

print("\n혼동 행렬")
print(confusion_matrix(y_test, y_pred))

# 7. 모델 저장
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/emotion_model.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("\n모델 저장 완료")
print("저장 위치: model/emotion_model.pkl")
print("저장 위치: model/tfidf_vectorizer.pkl")
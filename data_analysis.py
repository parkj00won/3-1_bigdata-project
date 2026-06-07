from datasets import load_dataset
import pandas as pd

# HuggingFace 데이터셋 불러오기
dataset = load_dataset("dair-ai/emotion", "split")

# train 데이터를 pandas DataFrame으로 변환
df = pd.DataFrame(dataset["train"])

# 라벨 이름 붙이기
label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

df["emotion"] = df["label"].map(label_map)

# 데이터 확인
print("데이터 불러오기 성공!")
print(df.head())
print(df.info())
print(df["emotion"].value_counts())

import matplotlib.pyplot as plt

# 감정별 데이터 개수 확인
emotion_counts = df["emotion"].value_counts()

print("\n감정별 데이터 개수")
print(emotion_counts)

plt.figure(figsize=(8, 5))
emotion_counts.plot(kind="bar")
plt.title("Emotion Label Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 문장 길이 분석
df["text_length"] = df["text"].apply(len)
df["word_count"] = df["text"].apply(lambda x: len(x.split()))

print("\n감정별 평균 단어 수")
print(df.groupby("emotion")["word_count"].mean().sort_values(ascending=False))

plt.figure(figsize=(8, 5))
df.groupby("emotion")["word_count"].mean().sort_values(ascending=False).plot(kind="bar")
plt.title("Average Word Count by Emotion")
plt.xlabel("Emotion")
plt.ylabel("Average Word Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

from sklearn.feature_extraction.text import CountVectorizer

def show_top_words(emotion_name, top_n=15):
    texts = df[df["emotion"] == emotion_name]["text"]

    vectorizer = CountVectorizer(stop_words="english")
    word_matrix = vectorizer.fit_transform(texts)

    word_counts = word_matrix.sum(axis=0)
    words = vectorizer.get_feature_names_out()

    word_freq = []
    for word, count in zip(words, word_counts.tolist()[0]):
        word_freq.append((word, count))

    word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)

    print(f"\n[{emotion_name}] 자주 등장하는 단어 TOP {top_n}")
    for word, count in word_freq[:top_n]:
        print(word, count)

show_top_words("joy")
show_top_words("sadness")
show_top_words("anger")
show_top_words("fear")
show_top_words("love")
show_top_words("surprise")
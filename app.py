import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from datetime import datetime
import os

# =========================
# 기본 설정
# =========================

st.set_page_config(
    page_title="MoodMate",
    page_icon="🌙",
    layout="centered"
)

label_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

emotion_korean = {
    "sadness": "슬픔",
    "joy": "기쁨",
    "love": "애정",
    "anger": "분노",
    "fear": "불안/두려움",
    "surprise": "놀람"
}

# =========================
# 감정별 추천 데이터
# =========================

recommendations = {
    "sadness": {
        "title": "회복이 필요한 하루",
        "content": "잔잔한 플레이리스트나 짧은 산책을 추천합니다.",
        "action": "오늘 좋았던 일 1가지만 적어보세요.",
        "message": "지금의 감정을 없애려 하기보다 조금 천천히 지나가게 두어도 괜찮아요.",
        "contents_list": [
            "잔잔한 플레이리스트 듣기",
            "10분 산책하기",
            "따뜻한 차 마시기",
            "오늘 좋았던 일 1가지 적기",
            "위로가 되는 영화나 에세이 보기"
        ],
        "card_line_1": "오늘은 회복이 필요한 하루예요.",
        "card_line_2": "천천히 지나가도 괜찮아요."
    },
    "joy": {
        "title": "기분 좋은 에너지의 하루",
        "content": "밝은 음악이나 친구와의 대화를 추천합니다.",
        "action": "좋은 기분을 기록으로 남겨보세요.",
        "message": "오늘의 좋은 에너지를 작게 저장해두면 나중에 다시 꺼내볼 수 있어요.",
        "contents_list": [
            "밝은 음악 듣기",
            "친구에게 연락하기",
            "오늘의 사진 남기기",
            "좋았던 순간을 짧게 기록하기",
            "가벼운 산책이나 운동하기"
        ],
        "card_line_1": "오늘은 기분 좋은 에너지가 있는 하루예요.",
        "card_line_2": "이 마음을 작게 기록해두면 좋아요."
    },
    "love": {
        "title": "따뜻한 연결의 하루",
        "content": "감성적인 영화나 편지를 추천합니다.",
        "action": "고마운 사람에게 짧은 메시지를 보내보세요.",
        "message": "누군가를 떠올리는 마음은 그 자체로 꽤 좋은 기록이 될 수 있어요.",
        "contents_list": [
            "고마운 사람에게 메시지 보내기",
            "따뜻한 영화 보기",
            "편지나 메모 쓰기",
            "좋아하는 사람과 나눈 대화 떠올리기",
            "감성적인 플레이리스트 듣기"
        ],
        "card_line_1": "오늘은 마음이 따뜻하게 연결된 하루예요.",
        "card_line_2": "그 마음을 말이나 기록으로 남겨보세요."
    },
    "anger": {
        "title": "감정 정리가 필요한 하루",
        "content": "차분한 호흡 루틴이나 감정 정리 메모를 추천합니다.",
        "action": "화난 이유를 한 문장으로 적고 잠시 멈춰보세요.",
        "message": "화가 났다는 건 무언가 중요하게 여긴 마음이 있었다는 뜻일 수 있어요.",
        "contents_list": [
            "3분 호흡 루틴 하기",
            "화난 이유를 한 문장으로 쓰기",
            "잠깐 휴대폰 내려놓기",
            "빠르게 걷기",
            "차분한 음악 듣기"
        ],
        "card_line_1": "오늘은 감정 정리가 필요한 하루예요.",
        "card_line_2": "바로 답하지 말고 잠시 멈춰도 괜찮아요."
    },
    "fear": {
        "title": "안정감이 필요한 하루",
        "content": "익숙하고 편안한 콘텐츠를 추천합니다.",
        "action": "걱정되는 일을 체크리스트로 나눠보세요.",
        "message": "막연한 걱정은 작게 나누면 조금 덜 무서워질 수 있어요.",
        "contents_list": [
            "걱정되는 일 체크리스트 만들기",
            "해야 할 일 3개만 정리하기",
            "익숙한 영상이나 음악 듣기",
            "따뜻한 물 마시기",
            "5분 스트레칭하기"
        ],
        "card_line_1": "오늘은 안정감이 필요한 하루예요.",
        "card_line_2": "막연한 걱정은 작게 나누면 덜 무거워져요."
    },
    "surprise": {
        "title": "예상 밖의 일이 있었던 하루",
        "content": "새로운 콘텐츠 탐색이나 짧은 기록을 추천합니다.",
        "action": "오늘 기억에 남는 장면을 적어보세요.",
        "message": "예상하지 못한 순간도 지나고 나면 하루의 인상적인 장면이 될 수 있어요.",
        "contents_list": [
            "오늘의 unexpected moment 기록하기",
            "새로운 콘텐츠 둘러보기",
            "친구에게 오늘 있었던 일 말하기",
            "기억에 남는 장면 한 줄로 남기기",
            "가벼운 산책으로 생각 정리하기"
        ],
        "card_line_1": "오늘은 예상 밖의 장면이 있었던 하루예요.",
        "card_line_2": "그 순간도 나중에는 꽤 선명한 기록이 될 수 있어요."
    }
}

# =========================
# 모델 불러오기
# =========================

model = joblib.load("model/emotion_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# =========================
# 한국어 감정 키워드 처리
# =========================

korean_emotion_keywords = {
    "sadness": [
        "슬프", "우울", "외롭", "눈물", "힘들", "지치", "허무", "속상",
        "공허", "쓸쓸", "울고", "무기력", "괜찮지 않", "피곤"
    ],
    "joy": [
        "기쁘", "행복", "좋아", "신나", "즐거", "설레", "뿌듯", "웃",
        "만족", "재밌", "재미있", "최고", "잘됐다"
    ],
    "love": [
        "사랑", "고마", "감사", "따뜻", "보고싶", "그립", "좋아하",
        "소중", "애정", "다정", "위로받"
    ],
    "anger": [
        "화나", "짜증", "열받", "분노", "빡치", "싫어", "억울",
        "답답", "불쾌", "미워", "화가"
    ],
    "fear": [
        "무섭", "두렵", "불안", "걱정", "긴장", "떨려", "겁나",
        "초조", "압박", "망할까", "어떡하지"
    ],
    "surprise": [
        "놀라", "깜짝", "갑자기", "뜻밖", "예상", "신기", "당황",
        "어이없", "충격", "대박"
    ]
}

emotion_to_english_sentence = {
    "sadness": "I feel sad and lonely today",
    "joy": "I feel happy and joyful today",
    "love": "I feel loved and thankful today",
    "anger": "I feel angry and frustrated today",
    "fear": "I feel scared and anxious today",
    "surprise": "I feel surprised and shocked today"
}

def detect_korean_emotion(text):
    scores = {}

    for emotion, keywords in korean_emotion_keywords.items():
        score = 0
        for keyword in keywords:
            if keyword in text:
                score += 1
        scores[emotion] = score

    best_emotion = max(scores, key=scores.get)

    if scores[best_emotion] == 0:
        return None, scores

    return best_emotion, scores

def is_korean_text(text):
    return any("가" <= char <= "힣" for char in text)

def predict_emotion(user_text):
    original_text = user_text
    used_korean_rule = False
    detected_emotion = None
    keyword_scores = None

    if is_korean_text(user_text):
        detected_emotion, keyword_scores = detect_korean_emotion(user_text)

        if detected_emotion is not None:
            user_text = emotion_to_english_sentence[detected_emotion]
            used_korean_rule = True

    text_tfidf = vectorizer.transform([user_text])
    pred = model.predict(text_tfidf)[0]
    proba = model.predict_proba(text_tfidf)[0]

    emotion = label_map[pred]
    confidence = np.max(proba) * 100

    return {
        "original_text": original_text,
        "model_input_text": user_text,
        "emotion": emotion,
        "confidence": confidence,
        "proba": proba,
        "used_korean_rule": used_korean_rule,
        "detected_emotion": detected_emotion,
        "keyword_scores": keyword_scores
    }

# =========================
# 감정 로그 저장 함수
# =========================

LOG_FILE = "emotion_log.csv"

def save_emotion_log(user_text, emotion, confidence, action):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    new_log = pd.DataFrame([{
        "datetime": now,
        "text": user_text,
        "emotion": emotion,
        "emotion_korean": emotion_korean[emotion],
        "confidence": round(confidence, 2),
        "recommended_action": action
    }])

    if os.path.exists(LOG_FILE):
        old_log = pd.read_csv(LOG_FILE)
        updated_log = pd.concat([old_log, new_log], ignore_index=True)
    else:
        updated_log = new_log

    updated_log.to_csv(LOG_FILE, index=False, encoding="utf-8-sig")

def load_emotion_log():
    if os.path.exists(LOG_FILE):
        return pd.read_csv(LOG_FILE)
    return pd.DataFrame()

# =========================
# 화면 구성
# =========================

st.title("🌙 MoodMate")
st.subheader("오늘의 문장으로 알아보는 기분 맞춤 콘텐츠 추천 서비스")

st.write(
    "문장을 입력하면 감정을 분석하고 감정에 맞는 콘텐츠와 행동을 추천합니다. "
    "영어 문장은 모델이 직접 분석하고 한국어 문장은 감정 키워드를 먼저 감지해 분석합니다."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "서비스 소개",
    "데이터 분석",
    "감정 예측",
    "감정 기록"
])

# =========================
# 탭 1. 서비스 소개
# =========================

with tab1:
    st.write("## 프로젝트 소개")

    st.write(
        """
        MoodMate는 사용자가 작성한 짧은 문장을 바탕으로 감정을 분석하고  
        감정 유형에 따라 적절한 콘텐츠와 행동을 추천하는 서비스입니다.
        """
    )

    st.write("### 사용 데이터")
    st.write(
        """
        - 데이터셋: HuggingFace `dair-ai/emotion`
        - 데이터 형태: 영어 텍스트 문장 + 감정 라벨
        - 감정 라벨: sadness, joy, love, anger, fear, surprise
        - 모델: TF-IDF + Logistic Regression
        """
    )

    st.write("### 서비스 흐름")

    st.write(
        """
        1. 사용자가 오늘의 기분 문장을 입력합니다.  
        2. 입력 문장의 감정을 예측합니다.  
        3. 예측된 감정에 맞는 콘텐츠와 행동을 추천합니다.  
        4. 감정별 예측 확률을 그래프로 보여줍니다.  
        5. 분석 결과를 감정 로그로 저장하고 누적 기록을 확인합니다.
        """
    )

# =========================
# 탭 2. 데이터 분석
# =========================

with tab2:
    st.write("## 데이터 분석")

    with st.spinner("데이터를 불러오는 중입니다..."):
        dataset = load_dataset("dair-ai/emotion", "split")
        df = pd.DataFrame(dataset["train"])
        df["emotion"] = df["label"].map(label_map)
        df["text_length"] = df["text"].apply(len)
        df["word_count"] = df["text"].apply(lambda x: len(x.split()))

    st.write("### 데이터 미리보기")
    st.dataframe(df.head())

    st.write("### 감정별 데이터 개수")
    emotion_counts = df["emotion"].value_counts()
    st.bar_chart(emotion_counts)

    st.write("### 감정별 평균 단어 수")
    avg_words = df.groupby("emotion")["word_count"].mean().sort_values(ascending=False)
    st.bar_chart(avg_words)

    st.write("### 감정별 예시 문장")
    selected_emotion = st.selectbox(
        "예시를 보고 싶은 감정을 선택하세요.",
        list(emotion_korean.keys())
    )

    example_rows = df[df["emotion"] == selected_emotion]["text"].head(5)

    for idx, text in enumerate(example_rows, 1):
        st.write(f"{idx}. {text}")

# =========================
# 탭 3. 감정 예측
# =========================

with tab3:
    st.write("## 감정 예측")

    user_text = st.text_area(
        "오늘의 기분을 문장으로 적어보세요.",
        placeholder="예: 오늘은 너무 지치고 외로운 하루였어 / I feel tired and lonely today."
    )

    if st.button("감정 분석하기"):
        if user_text.strip() == "":
            st.warning("문장을 입력해주세요.")
        else:
            result = predict_emotion(user_text)

            emotion = result["emotion"]
            confidence = result["confidence"]
            proba = result["proba"]

            rec = recommendations[emotion]

            st.write("## 분석 결과")

            st.success(
                f"예측 감정: {emotion_korean[emotion]} ({emotion})"
            )

            st.write(f"예측 확률: **{confidence:.1f}%**")

            if result["used_korean_rule"]:
                st.caption(
                    f"한국어 입력이 감지되어 감정 키워드 기반으로 `{result['detected_emotion']}` 감정을 먼저 추정한 뒤 분석했습니다."
                )

            # =========================
            # 추천 기능 3. 오늘의 한 줄 카드
            # =========================

            st.write("## 오늘의 한 줄 카드")

            st.markdown(
                f"""
                <div style="
                    padding: 24px;
                    border-radius: 18px;
                    background-color: #F7F3EA;
                    border: 1px solid #E4D8C8;
                    text-align: center;
                    margin-top: 10px;
                    margin-bottom: 20px;
                ">
                    <h3 style="margin-bottom: 8px;">{rec['card_line_1']}</h3>
                    <p style="font-size: 18px;">{rec['card_line_2']}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # =========================
            # 추천 기능 2. 콘텐츠 추천 리스트
            # =========================

            st.write("## 오늘의 추천")

            st.info(f"**{rec['title']}**")
            st.write(rec["message"])

            st.write("### 추천 콘텐츠 리스트")
            for item in rec["contents_list"]:
                st.write(f"- {item}")

            st.write(f"### 추천 행동")
            st.write(rec["action"])

            # 감정별 확률 그래프
            st.write("## 감정별 예측 확률")

            prob_dict = {
                emotion_korean[label_map[i]]: float(proba[i])
                for i in range(len(proba))
            }

            st.bar_chart(prob_dict)

            # =========================
            # 추천 기능 1. 감정 기록 저장
            # =========================

            save_emotion_log(
                user_text=result["original_text"],
                emotion=emotion,
                confidence=confidence,
                action=rec["action"]
            )

            st.success("감정 기록이 저장되었습니다. 감정 기록 탭에서 확인할 수 있어요.")

            with st.expander("분석 과정 보기"):
                st.write("입력 문장")
                st.code(result["original_text"])

                st.write("모델에 들어간 문장")
                st.code(result["model_input_text"])

                if result["keyword_scores"] is not None:
                    st.write("한국어 감정 키워드 점수")
                    st.json(result["keyword_scores"])

# =========================
# 탭 4. 감정 기록
# =========================

with tab4:
    st.write("## 감정 기록")

    log_df = load_emotion_log()

    if log_df.empty:
        st.info("아직 저장된 감정 기록이 없습니다. 감정 예측 탭에서 문장을 분석해보세요.")
    else:
        st.write("### 누적 감정 기록")
        st.dataframe(log_df)

        st.write("### 내 감정 분포")
        emotion_log_counts = log_df["emotion_korean"].value_counts()
        st.bar_chart(emotion_log_counts)

        st.write("### 최근 기록")
        recent_logs = log_df.tail(5).sort_index(ascending=False)

        for _, row in recent_logs.iterrows():
            st.write(
                f"**{row['datetime']}** | {row['emotion_korean']} | "
                f"{row['confidence']}%"
            )
            st.caption(row["text"])
            st.write("---")

        csv_data = log_df.to_csv(index=False, encoding="utf-8-sig")

        st.download_button(
            label="감정 기록 CSV 다운로드",
            data=csv_data,
            file_name="emotion_log.csv",
            mime="text/csv"
        )

        if st.button("감정 기록 전체 삭제"):
            if os.path.exists(LOG_FILE):
                os.remove(LOG_FILE)
            st.success("감정 기록이 삭제되었습니다. 페이지를 새로고침하면 반영됩니다.")
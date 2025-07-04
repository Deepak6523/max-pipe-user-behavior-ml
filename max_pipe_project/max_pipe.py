import re
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Phase 1: Clean and preprocess text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

def preprocess_texts(text_list):
    cleaned = [clean_text(t) for t in text_list]
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(cleaned)
    return vectors, vectorizer

# Phase 2: Classify reasons (excuse or genuine)
def train_classifier(df):
    X, vectorizer = preprocess_texts(df['text'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("\n--- Classification Report ---")
    print(classification_report(y_test, preds))
    return clf, vectorizer

# Phase 3: Detect behavior patterns
def detect_patterns(log_df):
    summary = {}
    for user in log_df['user_id'].unique():
        user_logs = log_df[log_df['user_id'] == user]
        skipped = user_logs[user_logs['status'] == 'missed']
        if not skipped.empty:
            summary[user] = f"{len(skipped)} missed tasks. Suggest rescheduling mornings." \
                if any("am" in t for t in skipped['time'].str.lower()) else "Irregular performance."
    return summary

# Phase 4: Sentiment scoring of feedback
def analyze_feedback(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)
    return score['compound']

def score_feedback_batch(feedback_df):
    feedback_df['sentiment_score'] = feedback_df['feedback'].apply(analyze_feedback)
    return feedback_df

# Master pipeline
def run_pipeline():
    # Phase 1 & 2: Classification
    print("\n>>> Loading labeled reasons...")
    reasons_df = pd.read_csv("data/processed/labeled_reasons.csv")
    clf, vectorizer = train_classifier(reasons_df)

    # Phase 3: Behavior tracking
    print("\n>>> Analyzing task logs...")
    logs_df = pd.read_csv("data/processed/user_logs.csv")
    behavior_summary = detect_patterns(logs_df)
    print("\n--- Behavior Summary ---")
    print(json.dumps(behavior_summary, indent=2))

    # Phase 4: Feedback scoring
    print("\n>>> Scoring user feedback...")
    feedback_df = pd.read_csv("data/raw/feedback.csv")
    scored = score_feedback_batch(feedback_df)
    print("\n--- Sentiment Scores ---")
    print(scored[['user_id', 'sentiment_score']])

if __name__ == "__main__":
    run_pipeline()

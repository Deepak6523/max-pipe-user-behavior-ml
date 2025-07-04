# Max Pipe - User Behavior Classifier 🚀

A smart Python project that uses Machine Learning and Natural Language Processing (NLP) to analyze user behavior, classify task excuses, and detect sentiment from user feedback.

---

## 🔍 What It Does

**Max Pipe** performs three powerful tasks:

1. **Excuse Detection**: Identifies if a reason is genuine or an excuse using a logistic regression classifier.
2. **Behavior Analysis**: Analyzes when users often miss tasks (e.g., always at 9:00 AM).
3. **Sentiment Scoring**: Reads user feedback and scores its tone as positive, neutral, or negative.

---

## 🛠️ Technologies Used

- Python 🐍
- Pandas & NumPy
- Scikit-learn (Logistic Regression + TF-IDF)
- VADER Sentiment Analyzer (NLTK)
- Joblib (model saving)
- Jupyter Notebook or script-based execution

---

## 📁 Project Structure

max_pipe_project/
├── data/
│   ├── processed/
│   │   ├── labeled_reasons.csv ✅(contains text + label)
│   │   └── user_logs.csv ✅ (contains user_id, time, status)
│   └── raw/
│       └── feedback.csv ✅ (contains user_id, feedback)
├── max_pipe.py ✅
├── requirements.txt ✅
└── .venv/ ✅ (optional)



🙋 About Me
Deepak Singh Chouhan
I'm a Python developer passionate about AI and productivity tools.
📬 Let's connect on LinkedIn


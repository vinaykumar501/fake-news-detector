# 📰 Fake News Detection using Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vinaykumar501/fake-news-detector/blob/main/fake_news_detection.ipynb)

This project uses machine learning models to classify news as ...

# 📰 Fake News Detection using Machine Learning

This project uses machine learning models to classify news as **real** or **fake** based on its content. It trains multiple models (Logistic Regression, Naive Bayes, Random Forest, and Support Vector Machine), and selects the best performing one (SVM) for predictions.

## 📂 Dataset

The dataset used in this project comes from two CSV files:
- `Fake.csv`: Contains fake news articles.
- `True.csv`: Contains real news articles.

Each file includes a `title` and `text` field. A label is added manually:
- `1` → Real news  
- `0` → Fake news

## 🔍 Project Structure

📁 Your GitHub Repo
├── Fake.csv
├── True.csv
├── svm_model.pkl
├── tfidf_vectorizer.pkl
├── fake_news_detection.ipynb
└── README.md


## 🧠 ML Pipeline Overview

1. **Load Data**: Merge and label `Fake.csv` and `True.csv`.
2. **Preprocessing**:
   - Lowercase all text.
   - Remove URLs, symbols, and non-alphabet characters.
3. **Feature Extraction**:
   - TF-IDF Vectorization
4. **Train-Test Split**
5. **Model Training**:
   - Logistic Regression
   - Naive Bayes
   - Random Forest
   - SVM (selected)
6. **Evaluation**:
   - Accuracy Score
   - Confusion Matrix (Plotted using `seaborn`)
7. **Model Export**:
   - Saved as `svm_model.pkl`
   - TF-IDF saved as `tfidf_vectorizer.pkl`

## 🧪 Example Output

```python
>>> predict_news("Mexico to review need for tax changes after U.S. reform-document.")
Real News ✅

>>> predict_news("Banks, healthcare service firms among winners from U.S. tax bill")
Real News ✅

>>> predict_news("Donald Trump Sends Out Embarrassing New Year’s Eve Message; This is Disturbing")
Fake News ❌

# MindSense - Sentiment Analysis for Mental Health 🧠💬

This project explores the use of Natural Language Processing (NLP) techniques to analyze and classify sentiments associated with various mental health conditions. The goal is to extract meaningful insights from textual data and build robust models to predict mental health conditions based on linguistic patterns.

---

## 📊 Dataset Overview

- **Source**: <a href="https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health/data" target="_blank">
    <img src="https://kaggle.com/static/images/site-logo.png" alt="Kaggle" width="150" style="margin: 10px;"/>
</a>

- **Classes and Distribution**:
  - **Normal**: 16,343
  - **Depression**: 15,404
  - **Suicidal**: 10,652
  - **Anxiety**: 3,841
  - **Bipolar**: 2,777
  - **Stress**: 2,587
  - **Personality Disorder**: 1,077

---

## 🧪 Exploratory Data Analysis (EDA)

### Key Steps:
1. **Bar Graphs**:
   - Visualized the top 15 most frequent words after removing stop words for each condition.
   - Example insights:
     - **Anxiety**: Dominated by words like "anxiety", "feel", and "know".
     - **Normal**: Positive words like "good" and "day" were frequent.
     - **Suicidal**: Words like "want", "life", and "anymore" reflected deep contemplation.

2. **Most Common Words Analysis**:
   - Identified distinct word patterns for each mental health condition.
   - Common themes across conditions include "feel", "like", and "want".
   - Each condition exhibited unique linguistic markers (e.g., "anxiety" in Anxiety, "depression" in Depression).

3. **Word Clouds**:
   - Generated word clouds to visualize dominant terms across all conditions.
   - Condition-specific terms revealed distinct emotional and contextual focuses:
     - **Depression**: "feel", "life", "depression"
     - **Stress**: "stress", "help", "work"
     - **Suicidal Thoughts**: "want", "life", "die"

---

## 🔧 Preprocessing

1. **Steps**:
   - Tokenized text into words.
   - Removed punctuation, special characters, and stop words.
   - Converted text to lowercase.
   - Applied stemming/lemmatization.

2. **Balancing the Dataset**:
   - Used **data augmentation** techniques:
     - Synonym replacement and insertion.
     - Random word swapping within sentences.
   - Applied **TF-IDF** with **SMOTE** for further balancing.

---

## 🧠 Modeling and Results

### Techniques Used:
1. **TF-IDF + SMOTE**:
   - Trained classifiers like **XGBoost**, achieving:
     - **Accuracy**: 74%
     - **Improved F1 scores** on imbalanced classes.

2. **Word2Vec + Bi-LSTM**:
   - Leveraged word embeddings and a **Bidirectional LSTM** model:
     - **Accuracy**: 76%
     - Significantly improved F1 scores for minority classes.

3. **Other Models**:
   - Explored **CNN** and traditional **LSTM** models, but Bi-LSTM outperformed them.

---

## 🌟 Insights and Learnings

- **Emotional Patterns**:
  - Distinct words provide insights into the emotional and mental state of individuals.
  - For example, terms like "help" and "work" are stress indicators, while "life" and "anymore" are red flags for suicidal tendencies.

- **Feature Importance**:
  - Certain terms are key predictors for specific conditions, making them valuable features in classification models.

- **Challenges**:
  - Balancing the dataset required significant effort, but techniques like **SMOTE** and data augmentation proved effective.
  - Computational constraints limited deeper experimentation with larger models.

---

## 🚀 Applications

- **Mental Health Monitoring**:
  - Detect early signs of mental health issues from textual input.
  
- **Intervention Systems**:
  - Flag critical terms like "suicide" or "kill" for timely intervention.

- **Emotion Detection**:
  - Identify emotional states associated with different mental health conditions.

---

## 🛠️ Tools and Technologies

- **Libraries**:
  - `pandas`, `numpy`, `seaborn`, `matplotlib`, `sklearn`
  - `XGBoost`, `LightGBM`, `CatBoost`, `imblearn`, `gensim` (for Word2Vec)
  - `TensorFlow`, `Keras` (for LSTM/CNN modeling)

- **Techniques**:
  - Data preprocessing (Tokenization, Lemmatization, Stop word removal)
  - Data balancing (SMOTE, augmentation)
  - Visualization (Bar graphs, word clouds)

---

## 🖋️ Conclusion

This project showcases the potential of NLP in understanding and predicting mental health conditions through textual data. By combining EDA, feature engineering, and robust modeling techniques, we can build systems that not only provide insights but also serve as tools for early detection and intervention in mental health care.

---

Feel free to fork the repository and explore the code. Contributions and suggestions are welcome!

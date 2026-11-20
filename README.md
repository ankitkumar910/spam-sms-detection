# Spam SMS Detection

A machine learning application that identifies whether an SMS message is spam or legitimate (ham).

This project was developed as a college mini project to demonstrate a complete natural language processing (NLP) workflow, from data preprocessing and model training to a simple Streamlit web interface.

**GitHub Repository:** https://github.com/ankitkumar910/spam-sms-detection

---

## Project Overview

SMS spam is unwanted text that may advertise products, request personal information, or attempt to deceive users. This project uses supervised machine learning to classify SMS messages into two categories:

- `Spam`: Unwanted or potentially fraudulent messages
- `Ham`: Legitimate, normal messages

The application takes a message as input and displays:

- The predicted category
- The model's confidence level
- A clear visual warning or success message

---

## System Workflow

```text
SMS Message
     |
     v
Text Lowercasing
     |
     v
Punctuation and Extra-Space Removal
     |
     v
TF-IDF Feature Extraction
     |
     v
Logistic Regression Classifier
     |
     v
Spam or Ham Prediction
```

---

## Technologies Used

| Category | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Feature Extraction | TF-IDF Vectorizer |
| Classification Algorithm | Logistic Regression |
| Data Processing | Pandas |
| Numerical Operations | NumPy |
| Web Interface | Streamlit |
| Model Serialization | Joblib |

### Why Logistic Regression?

Logistic Regression is a supervised binary classification algorithm. It works well with TF-IDF text features and provides a fast, understandable baseline for a college-level spam detection project.

---

## NLP and Machine Learning Concepts

This project demonstrates the following concepts:

1. **Text Preprocessing**
   - Conversion to lowercase
   - Removal of punctuation and special characters
   - Removal of extra whitespace

2. **Tokenization and Feature Extraction**
   - TF-IDF converts text into numerical features
   - Both individual words and word pairs (bigrams) are included
   - Sublinear term frequency reduces the effect of very common words

3. **Train-Test Split**
   - 80% of the data is used for training
   - 20% is used for testing
   - Stratified sampling preserves class proportions

4. **Supervised Classification**
   - The Logistic Regression model learns patterns from labelled spam and ham messages

5. **Class Balancing**
   - Balanced class weights reduce bias toward the larger ham category

6. **Model Evaluation**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion matrix

7. **Model Persistence**
   - The trained classifier and TF-IDF vectorizer are saved with Joblib

---

## Dataset

The project uses labelled SMS messages from:

- The SMS Spam Collection dataset
- A custom collection of example spam and ham messages

The model combines both datasets and removes duplicate cleaned messages before training.

---

## Model Performance

The retrained model uses 10,000 TF-IDF features, unigram and bigram features, balanced class weights, and a fixed train-test split.

| Metric | Result |
|---|---:|
| Accuracy | 97.87% |
| Ham Precision | 99% |
| Ham Recall | 99% |
| Spam Precision | 91% |
| Spam Recall | 92% |
| Spam F1-score | 92% |

Performance may vary if the dataset or model configuration is changed.

---

## Project Structure

```text
spam-sms-detection/
├── app.py                       # Streamlit web application
├── main.py                      # Training and evaluation script
├── spam.csv                     # SMS Spam Collection dataset
├── sample_sms_messages.csv      # Additional example messages
├── spam_classifier_model.pkl    # Trained Logistic Regression model
├── tfidf_vectorizer.pkl         # Trained TF-IDF vectorizer
├── requirements.txt             # Required Python packages
└── README.md                    # Project documentation
```

---

## Installation and Run Instructions

### 1. Clone the repository

```bash
git clone https://github.com/ankitkumar910/spam-sms-detection.git
cd spam-sms-detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Train the model

```bash
python main.py
```

This command preprocesses the data, trains the classifier, prints evaluation metrics, and saves both model files.

### 6. Start the web application

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit, enter an SMS message, and select **Predict**.

---

## Example Messages

### Spam

```text
Congratulations! You have won a free prize. Claim now!
URGENT: Your account has been suspended. Verify now.
```

### Ham

```text
Hey, are we still meeting for dinner tonight?
Lunch was great today, let's do it again soon.
```

---

## Future Improvements

- Compare Logistic Regression with Naive Bayes and Support Vector Machines
- Apply cross-validation for more reliable evaluation
- Tune TF-IDF and classifier parameters
- Add a REST API
- Improve handling of URLs and shortened links
- Add a confidence threshold
- Expand the dataset to improve real-world performance

---

## Author

**Ankit Kumar**

- GitHub: https://github.com/ankitkumar910

---

## License

This project is created for educational purposes. The datasets and tools used are subject to their respective licenses.

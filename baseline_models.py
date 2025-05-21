import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Set random seed for reproducibility
np.random.seed(42)

def clean_tweet_text(text):
    """Clean and preprocess tweet text."""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove hashtag symbol
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespaces
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')]) # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()]) # Lemmatization
    return text

def train_and_evaluate_model(model_name, vectorizer, model, X_train, X_test, y_train, y_test):
    """Train and evaluate a model with given vectorizer."""
    # Create pipeline
    pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', model)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Calculate ROC-AUC score (one-vs-rest)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    except:
        roc_auc = None
    
    # Calculate average precision score
    try:
        avg_precision = average_precision_score(y_test, y_pred_proba, average='weighted')
    except:
        avg_precision = None
    
    # Print metrics
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC Score: {roc_auc:.4f}")
    if avg_precision is not None:
        print(f"Average Precision Score: {avg_precision:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{model_name.replace(" ", "_").lower()}_confusion_matrix.png')
    plt.close()
    
    # Plot precision-recall curve for each class
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(pipeline.classes_):
        precision, recall, _ = precision_recall_curve(
            (y_test == class_name).astype(int),
            y_pred_proba[:, i]
        )
        plt.plot(recall, precision, label=f'{class_name}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{model_name.replace(" ", "_").lower()}_pr_curve.png')
    plt.close()
    
    return pipeline

def main():
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = pd.read_csv("cyberbullying_tweets.csv")
    df['cleaned_tweet_text'] = df['tweet_text'].apply(clean_tweet_text)
    
    # Split data
    X = df['cleaned_tweet_text']
    y = df['cyberbullying_type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize models and vectorizers
    vectorizers = {
        'TF-IDF': TfidfVectorizer(max_features=5000),
        'Count': CountVectorizer(max_features=5000)
    }
    
    models = {
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }
    
    # Train and evaluate models
    trained_models = {}
    for vec_name, vectorizer in vectorizers.items():
        for model_name, model in models.items():
            print(f"\n{'='*50}")
            print(f"Training {model_name} with {vec_name}")
            print(f"{'='*50}")
            
            pipeline = train_and_evaluate_model(
                f"{model_name} ({vec_name})",
                vectorizer,
                model,
                X_train, X_test, y_train, y_test
            )
            
            # Save the trained model
            model_key = f"{model_name}_{vec_name}".replace(' ', '_').lower()
            trained_models[model_key] = pipeline
            
            # Save model to file
            with open(f'{model_key}_model.pkl', 'wb') as f:
                pickle.dump(pipeline, f)

if __name__ == "__main__":
    main() 
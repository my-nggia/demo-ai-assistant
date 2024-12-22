import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    X = data["Question"]  
    y = data["Category"] 
    return X, y

class QuestionClassifierModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=500)
        self.model = LogisticRegression()

    def train(self, X, y):
        X = [str(q) for q in X if q] 
        y = [label for label in y if label]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_test_tfidf = self.vectorizer.transform(X_test)
        
        self.model.fit(X_train_tfidf, y_train)
    
        y_pred = self.model.predict(X_test_tfidf)
        print(classification_report(y_test, y_pred))

    def predict(self, question):
        question_tfidf = self.vectorizer.transform([question])
        return self.model.predict(question_tfidf)[0]

    def save_model(self, model_path):
        with open(model_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)

if __name__ == "__main__":
    file_path = "marketing_questions.csv"  
    model_path = "question_classifier.pkl" 
    
    X, y = load_and_preprocess_data(file_path)

    classifier = QuestionClassifierModel()

    classifier.train(X, y)

    classifier.save_model(model_path)

    print(f"Model saved to {model_path}")

import joblib
from data_preprocessing import clean_text

# Load model + vectorizer
model = joblib.load('../model/model.pkl')
vectorizer = joblib.load('../model/vectorizer.pkl')

def predict(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return "Toxic" if prediction == 1 else "Not Toxic"

# CLI interaction
if __name__ == "__main__":
    while True:
        user_input = input("\nEnter a comment (or type 'exit'): ")
        if user_input.lower() == 'exit':
            break
        print("Prediction:", predict(user_input))
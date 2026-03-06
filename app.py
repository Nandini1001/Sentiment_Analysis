from flask import Flask, render_template, request
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load trained model
model = load_model("model.h5")

# Load tokenizer
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

max_length = 200


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    review = request.form["review"]

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([review])

    # Pad sequence
    padded = pad_sequences(sequence, maxlen=max_length)

    # Prediction
    prediction = model.predict(padded)[0][0]

    if prediction > 0.5:
        result = "Positive Review 😊"
        sentiment = "positive"
    else:
        result = "Negative Review 😞"
        sentiment = "negative"

    return render_template(
        "index.html",
        prediction_text=result,
        sentiment=sentiment
    )


if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, render_template_string
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Flatten, Dense
from PIL import Image
import numpy as np
import base64
import io

app = Flask(__name__)

# Try to load your actual model; if fails, use dummy model
try:
    model = load_model("cats_dogs_model.keras", compile=False)
except Exception as e:
    print("Error loading model:", e)
    # Dummy model for testing
    model = Sequential([
        Flatten(input_shape=(128,128,3)),
        Dense(1, activation='sigmoid')
    ])

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Cat Dog Classifier</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');

  * {
    box-sizing: border-box;
  }

  body {
    margin: 0;
    font-family: 'Montserrat', sans-serif;
    background: linear-gradient(135deg, #667eea, #764ba2);
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: #fff;
    padding: 20px;
  }

  h1 {
    font-weight: 700;
    font-size: 3rem;
    margin-bottom: 40px;
    text-shadow: 0 3px 6px rgba(0,0,0,0.3);
    letter-spacing: 3px;
  }

  .card {
    background: #fff;
    color: #333;
    border-radius: 30px;
    width: 480px;
    max-width: 90vw;
    padding: 40px 35px;
    box-shadow: 0 25px 50px rgba(0,0,0,0.15);
    text-align: center;
    transition: box-shadow 0.3s ease;
  }

  .card:hover {
    box-shadow: 0 35px 70px rgba(0,0,0,0.25);
  }

  form {
    margin-bottom: 30px;
  }

  input[type="file"] {
    display: block;
    margin: 0 auto 25px;
    cursor: pointer;
    width: 100%;
    padding: 15px 12px;
    border-radius: 16px;
    border: 2px solid #ddd;
    font-size: 1.1rem;
    transition: border-color 0.25s ease;
  }

  input[type="file"]:hover {
    border-color: #667eea;
  }

  button {
    background: #667eea;
    color: white;
    font-size: 1.25rem;
    font-weight: 700;
    padding: 15px 45px;
    border: none;
    border-radius: 25px;
    cursor: pointer;
    box-shadow: 0 10px 25px rgba(102,126,234,0.5);
    transition: background 0.3s ease;
  }

  button:hover {
    background: #5a6fd4;
  }

  img {
    max-width: 320px;
    border-radius: 28px;
    box-shadow: 0 12px 30px rgba(102,126,234,0.3);
    margin: 0 auto 25px;
    display: block;
  }

  h3 {
    font-size: 1.8rem;
    font-weight: 700;
    color: #667eea;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 0;
  }
</style>
</head>
<body>

<h1>Cat🐱 and Dog🐶 Classifier</h1>

<div class="card">
  <form method="POST" enctype="multipart/form-data" autocomplete="off">
    <input type="file" name="image" accept="image/*" required />
    <button type="submit">Predict</button>
  </form>

  {% if img_data %}
    <img src="data:image/png;base64,{{ img_data }}" alt="Uploaded Image" />
    <h3>Prediction: {{ prediction }}</h3>
  {% endif %}
</div>

</body>
</html>
"""


def predict(img):
    img = img.resize((128,128))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    p = model.predict(x)[0][0]
    return "Dog" if p > 0.5 else "Cat"

@app.route("/", methods=["GET","POST"])
def home():
    img_data = None
    prediction = None
    if request.method == "POST":
        file = request.files["image"]
        img = Image.open(file.stream).convert("RGB")
        prediction = predict(img)

        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return render_template_string(HTML, img_data=img_data, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

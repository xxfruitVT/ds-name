import pandas as pd
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

# Membuat aplikasi Flask
app = Flask(__name__)

# Aktifkan CORS
CORS(app)

# Muat model yang sudah disimpan
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Endpoint home
@app.route('/')
def welcome():
    return "<h1>Selamat Datang di API DS Model</h1>"

# Endpoint prediksi
@app.route('/predict', methods=['POST'])
def predict_diabetes():
    try:
        # Ambil data dari request
        data = request.get_json()

        # Buat DataFrame
        input_data = pd.DataFrame([{
            "Pregnancies": data['Pregnancies'],
            "Glucose": data['Glucose'],
            "BloodPressure": data['BloodPressure'],
            "SkinThickness": data['SkinThickness'],
            "Insulin": data['Insulin'],
            "BMI": data['BMI'],
            "DiabetesPedigreeFunction": data['DiabetesPedigreeFunction'],
            "Age": data['Age']
        }])

        # Prediksi
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)

        # Probabilitas %
        probability_negative = probabilities[0][0] * 100
        probability_positive = probabilities[0][1] * 100

        # Teks hasil prediksi
        if prediction[0] == 1:
            result = (
                f"Anda memiliki peluang menderita diabetes berdasarkan model KNN kami. "
                f"Probabilitas positif adalah {probability_positive:.2f}%."
            )
        else:
            result = (
                f"Hasil prediksi menunjukkan Anda kemungkinan rendah terkena diabetes. "
                f"Probabilitas negatif adalah {probability_negative:.2f}%."
            )

        # Return JSON
        return jsonify({
            'prediction': result,
            'probabilities': {
                'negative': f"{probability_negative:.2f}%",
                'positive': f"{probability_positive:.2f}%"
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Jalankan Flask
if __name__ == '__main__':
    app.run(debug=True)

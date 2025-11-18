from flask import Flask, request, jsonify
from src.inference import load_trained_model, get_treatment

app = Flask(__name__)
qa_pipeline = load_trained_model()

@app.route("/consultar", methods=["POST"])
def consultar():
    data = request.json
    symptom = data.get("sintoma", "")
    if not symptom:
        return jsonify({"erro": "Sintoma não informado"}), 400

    answer = get_treatment(qa_pipeline, symptom)
    return jsonify({"tratamento_sugerido": answer})
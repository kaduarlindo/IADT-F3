from src.parse_xml import load_xml_data
from src.prepare_dataset import create_hf_dataset
from src.train_model import train_model
from src.api import app

XML_PATH = "./data"
MODEL_NAME = "pierreguillou/bert-large-cased-squad-v1.1-portuguese"

if __name__ == "__main__":
    print("🔹 Lendo XMLs...")
    data = load_xml_data(XML_PATH)
    print(f"DEBUG: data carregado = {len(data)} samples")
    
    dataset = create_hf_dataset(data)
    print(f"DEBUG: dataset criado com {len(dataset)} samples")
    print(f"DEBUG: colunas = {dataset.column_names}")
    if len(dataset) > 0:
        print(f"DEBUG: primeiro sample = {dataset[0]}")

    if len(dataset) == 0:
        print("❌ ERRO: Dataset está vazio! Verifique XML_PATH e format dos XMLs.")
    else:
        print("🔹 Iniciando treinamento...")
        train_model(dataset, MODEL_NAME)

        print("🔹 Subindo API em http://localhost:5000 ...")
        app.run(host="0.0.0.0", port=5000)
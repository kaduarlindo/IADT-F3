# IADT-F3-LLM - Question Answering com Fine-tuning

Aplicação de **Question Answering (QA)** que realiza fine-tuning de um modelo BERT em português e expõe uma API REST para inferência.

## 📋 Requisitos

- Python 3.13+
- pip

## 🚀 Instalação

1. **Clone ou navegue até o diretório do projeto:**
   ```bash
   cd IADT-F3-LLM
   ```

2. **Crie um ambiente virtual (recomendado):**
   ```bash
   python -m venv .venv
   # PowerShell
   .\.venv\Scripts\Activate.ps1
   # cmd
   # .\.venv\Scripts\activate
   ```

3. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Estrutura do Projeto

```
IADT-F3-LLM/
├── data/                    # Diretório com arquivos XML de entrada
├── modelo_treinado/         # Modelo fine-tunado (gerado após treinamento)
├── src/
│   ├── parse_xml.py        # Carrega e parseia XMLs
│   ├── prepare_dataset.py  # Cria dataset do Hugging Face
│   ├── train_model.py      # Fine-tuning do modelo
│   ├── inference.py        # Inferência (carregamento do modelo)
│   └── api.py              # API Flask
├── main.py                 # Script principal
├── requirements.txt        # Dependências do projeto
└── README.md              # Este arquivo
```

## 📊 Preparação dos Dados

Os XMLs devem estar no diretório `./data` e conter a seguinte estrutura:

```xml
<qa_pair>
    <context>Texto de contexto com informação...</context>
    <question>Pergunta sobre o contexto?</question>
    <answer>Resposta esperada</answer>
</qa_pair>
```

**Exemplo:**
```xml
<qa_pair>
    <context>A diabetes é uma doença crônica que afeta o metabolismo da glicose.</context>
    <question>O que é diabetes?</question>
    <answer>Uma doença crônica que afeta o metabolismo da glicose</answer>
</qa_pair>
```

## ▶️ Uso

### 1. Treinamento

Execute o script principal para treinar o modelo:

```bash
python main.py
```

**O que acontece:**
- Carrega XMLs do diretório `./data`
- Cria um dataset Hugging Face
- Realiza fine-tuning do modelo BERT em português
- Salva o modelo treinado em `modelo_treinado/`
- Inicia a API em `http://localhost:5000`

### 2. Inferência via API

A API expõe um endpoint para fazer predições:

**POST** `/predict`

**Body (JSON):**
```json
{
    "context": "A febre é um aumento temporário da temperatura corporal.",
    "question": "O que é febre?"
}
```

**Response:**
```json
{
    "answer": "Um aumento temporário da temperatura corporal"
}
```

**Exemplo com curl:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"context":"A febre é um aumento temporário da temperatura corporal.","question":"O que é febre?"}'
```

### 3. Uso Programático

```python
from src.inference import load_trained_model, get_treatment

# Carrega modelo
qa_pipeline = load_trained_model()

# Faz inferência
result = get_treatment(qa_pipeline, "A febre é um sintoma de inflamação.")
print(result)
```

## ⚙️ Configurações

No arquivo `main.py`, você pode ajustar:

```python
XML_PATH = "./data"  # Caminho dos XMLs
MODEL_NAME = "pierreguillou/bert-large-cased-squad-v1.1-portuguese"  # Modelo base
```

No arquivo `src/train_model.py`, você pode ajustar hiperparâmetros:

```python
training_args = TrainingArguments(
    output_dir="modelo_treinado",
    per_device_train_batch_size=8,  # Batch size
    num_train_epochs=3,              # Número de épocas
    learning_rate=3e-5,              # Taxa de aprendizado
    weight_decay=0.01,               # Regularização
    remove_unused_columns=False
)
```

## 🔧 Solução de Problemas

### "No module named huggingface_hub"
Reinstale as dependências:
```bash
python -m pip install --force-reinstall -r requirements.txt
```

### "Dataset está vazio"
- Verifique se os XMLs estão em `./data`
- Confirme que os XMLs seguem a estrutura esperada (tags: `context`, `question`, `answer`)
- Verifique o encoding dos arquivos (deve ser UTF-8)

### "CUDA out of memory"
Reduza o `per_device_train_batch_size` em `train_model.py`:
```python
per_device_train_batch_size=4  # ou menor
```

### API não inicia
- Verifique se a porta 5000 já está em uso
- Tente uma porta diferente em `main.py`:
```python
app.run(host="0.0.0.0", port=5001)
```

## 📦 Dependências

- **transformers**: Modelos pré-treinados e fine-tuning
- **datasets**: Manipulação de datasets
- **torch**: Framework de deep learning
- **flask**: API REST
- **huggingface_hub**: Integração com Hugging Face

## 📝 Licença

Projeto educacional - Pós IA para Devs (GFT)

## 📞 Contato

Para dúvidas, verifique os arquivos de log gerados durante a execução.
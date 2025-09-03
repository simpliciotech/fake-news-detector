# 📰 Fake News Detector

## 📌 Descrição
Aplicação web que usa **NLP + Machine Learning** para classificar textos como **fake** ou **real**. 
Este repositório inclui um pipeline simples com TF‑IDF + Regressão Logística, um script de treino e uma API Flask com página web básica.

## 🛠️ Tecnologias
- Python, Flask
- scikit-learn (TF‑IDF + LogisticRegression)
- Pandas, NumPy, joblib
- Bootstrap (UI via template)

## 📂 Estrutura
```
fake-news-detector/
├── app.py
├── train.py
├── requirements.txt
├── .gitignore
├── data/
│   └── sample.csv
├── models/
│   └── .gitkeep
└── templates/
    └── index.html
```

## ⚙️ Como executar
```bash
# 1) Instale dependências
pip install -r requirements.txt

# 2) Treine o modelo (usa data/sample.csv por padrão)
python train.py

# 3) Rode a API
python app.py
```

- Acesse: http://127.0.0.1:5000
- Endpoint de predição: `POST /predict` com JSON: `{ "text": "..." }`

## 🧪 Trocar o dataset
Substitua `data/sample.csv` por um CSV com colunas `text,label` (labels: `fake` ou `real`). 
Defina `DATA_PATH` ao treinar, se quiser:
```bash
DATA_PATH="data/seu_arquivo.csv" python train.py
```

## 🚀 Próximos passos
- Avaliação mais robusta (k‑fold, métricas adicionais).
- Limpeza/Lematização de texto (spaCy/NLTK) e melhor pré‑processamento.
- Experimentar modelos transformer (HuggingFace).
- Deploy (Render/Heroku para API e Vercel/Netlify para front).

---

**Autor:** Robson Rodrigues Simplicio
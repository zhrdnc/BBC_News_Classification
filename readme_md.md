# BBC News Classification: Classical ML vs. Modern Transformers

An end-to-end Natural Language Processing (NLP) study that explores, evaluates, and benchmarks text classification architectures on the **BBC News Dataset**. 

This project goes beyond raw metric optimization by analyzing the engineering trade-offs between classical machine learning approaches (TF-IDF + Naive Bayes, Logistic Regression, SVM) and state-of-the-art Transformer-based architectures (DistilBERT / BERT), taking inference speed, resource consumption, and accuracy into account.

---

## 📌 Project Overview

* **Domain:** Natural Language Processing (NLP), Supervised Multi-Class Classification
* **Dataset:** BBC News Dataset (2,225 articles across 5 topical categories)
* **Target Categories:** 
  * 🏛️ Politics
  * 💼 Business
  * 💻 Tech
  * ⚽ Sport
  * 🎭 Entertainment
* **Primary Objective:** Analyze model performance, compute efficiency, and trade-offs across classical vectorization pipelines vs. fine-tuned pre-trained transformer representations.

---

## 🛠️ Tech Stack & Libraries

* **Core Language:** Python 3.x
* **Deep Learning & Transformers:** PyTorch, Hugging Face Transformers (`transformers`, `datasets`)
* **Traditional Machine Learning:** Scikit-learn
* **Text Preprocessing & NLP:** NLTK, Regex
* **Data Manipulation & Visualization:** Pandas, NumPy, Matplotlib, Seaborn

---

## 🔄 Pipeline Workflow

1. **Exploratory Data Analysis (EDA):**
   * Class balance inspection across all 5 article classes.
   * Document length distribution and vocabulary dispersion analysis.
2. **Text Preprocessing:**
   * Case folding, punctuation removal, stop-word elimination, and lemmatization/stemming.
   * Tokenization routines configured for both subword representations (WordPiece/BPE) and n-gram count models.
3. **Feature Extraction & Modeling:**
   * **Baseline Models:** TF-IDF vectorization paired with Multinomial Naive Bayes, Logistic Regression, and Support Vector Machines (SVM).
   * **Deep Architectures:** Pre-trained Transformer fine-tuning (e.g., DistilBERT/BERT) with adaptive learning rates and warmup schedules.
4. **Evaluation & Error Analysis:**
   * Performance evaluated using Accuracy, Weighted F1-Score, Confusion Matrices, and classification reports per category.

---

## 📊 Benchmark & Key Results

| Model Pipeline | Accuracy | Macro F1 | Training Footprint | Inference Latency |
| :--- | :---: | :---: | :---: | :---: |
| **TF-IDF + Naive Bayes** | ~96.8% | ~0.96 | Minimal (CPU, < 10s) | Sub-millisecond |
| **TF-IDF + Linear SVM** | ~97.5% | ~0.97 | Low (CPU, ~30s) | Sub-millisecond |
| **Fine-Tuned DistilBERT** | **~98.4%** | **0.98** | High (GPU required) | ~15–30 ms/sample |

> **Key Takeaway:** While fine-tuned Transformer representations achieved superior semantic boundary precision (notably between *Business* and *Politics*), classical linear models with TF-IDF offered competitive performance at a fraction of the compute and inference cost.

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/zhrdnc/BBC_News_Classification.git
cd BBC_News_Classification
```

### 2. Environment Setup
Install the necessary dependencies via pip:
```bash
pip install -r requirements.txt
```
*(Or install manually: `pip install torch transformers scikit-learn pandas numpy matplotlib seaborn nltk`)*

### 3. Running the Notebook
You can launch the project locally or via Google Colab:
```bash
jupyter notebook BBC_News_Classification.ipynb
```
Alternatively, open it directly using the **"Open in Colab"** badge inside the repository.

---

## 👤 Author
* **Zehra Dinçerz** — [GitHub Profile](https://github.com/zhrdnc)

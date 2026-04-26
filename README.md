# Movie Recommendation System Using Neural Networks

**Kartik Sehrawat | Reg. No: 241306124**  
**Subject: Deep Learning & Neural Networks**

---

## What This Project Does

This project builds a **Neural Collaborative Filtering (NCF)** model that predicts which movies a user will like based on their past viewing history. It learns from 100,000 real movie ratings and generates personalized Top-N recommendations using deep learning.

---

## Project Structure

```
RecommendationSystem/
|-- streamlit_app.py            # Interactive web app (Streamlit)
|-- recommendation_model.py     # Model training & evaluation script
|-- recommendation_notebook.ipynb  # Jupyter Notebook (step-by-step)
|-- requirements.txt            # Python dependencies
|-- README.md                   # This file
|-- model.keras                 # Trained model (auto-generated)
|-- encodings.pkl               # User/item encodings (auto-generated)
|-- training_curves.png         # Loss & accuracy plots (auto-generated)
|-- evaluation_metrics.png      # HR@10 & NDCG@10 chart (auto-generated)
|-- ml-100k/                    # MovieLens 100K dataset
    |-- u.data                  # 100,000 ratings
    |-- u.item                  # 1,682 movie titles
```

---

## Model Architecture

**Neural Collaborative Filtering (NCF)** combines two branches:

```
              +---------------------------+
User ID ----> | GMF: User Emb x Item Emb  | ---+
              +---------------------------+    |
                                               +---> Concat ---> Dense(1) ---> Sigmoid ---> Score
              +---------------------------+    |
Item ID ----> | MLP: 64 -> 32 -> 16 -> 8  | ---+
              +---------------------------+
```

| Component     | Details                                        |
|---------------|------------------------------------------------|
| GMF Branch    | Element-wise product of user & item embeddings  |
| MLP Branch    | 4 Dense layers (64, 32, 16, 8) + ReLU + Dropout |
| Fusion        | Concatenate GMF + MLP outputs (NeuMF)           |
| Output        | Dense(1) + Sigmoid = predicted interaction score |
| Loss          | Binary Cross-Entropy                            |
| Optimizer     | Adam (learning rate = 0.001)                    |
| Embedding Dim | 32                                              |

---

## Results

| Metric          | Value      |
|-----------------|------------|
| Hit Rate @ 10   | **77.0%**  |
| NDCG @ 10       | **0.5067** |
| Training Accuracy | 92.8%    |
| Epochs          | 8 (early stopping) |

---

## Dataset

**MovieLens 100K** by GroupLens Research  
- 100,000 ratings (1-5 scale)  
- 943 users, 1,682 movies  
- Auto-downloaded on first run  

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model (if not already trained)

```bash
python recommendation_model.py
```

This downloads the dataset, trains the NCF model, and saves:
- `model.keras` (trained model)
- `encodings.pkl` (user/item mappings)
- `training_curves.png` and `evaluation_metrics.png` (plots)

### 3. Launch the Web App

```bash
streamlit run streamlit_app.py
```

Open **http://localhost:8501** in your browser.

### 4. (Optional) Run the Jupyter Notebook

```bash
jupyter notebook recommendation_notebook.ipynb
```

---

## Reference

He, X., Liao, L., Zhang, H., Nie, L., Hu, X., & Chua, T.S. (2017).  
*Neural Collaborative Filtering.*  
Proceedings of the 26th International Conference on World Wide Web (WWW 2017), pp. 173-182.

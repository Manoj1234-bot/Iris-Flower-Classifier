# 🌸 Iris Flower Classifier

A machine learning project that classifies **Iris flowers** into three different species (**Setosa, Versicolor, and Virginica**) using **Scikit-learn**. This project trains and compares multiple classification algorithms, evaluates their performance using accuracy, confusion matrix, and classification reports, visualizes feature relationships, and predicts the species of a flower based on user-provided measurements.

---

# ✨ Features

- 🌸 Classifies Iris flowers into 3 species
- 🤖 Compares Multiple Machine Learning Models
- 📊 Accuracy Comparison
- 📈 Confusion Matrix
- 📋 Classification Report
- 🌼 Interactive Custom Flower Prediction
- 📉 Feature Relationship Visualization
- ⚡ Feature Scaling using StandardScaler
- 📚 Uses the built-in Iris Dataset
- 💻 Simple Command-Line Interface

---

# 🛠️ Technologies Used

- Python 3
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

---

# 📁 Project Structure

```text
iris-flower-classifier/
│
├── classifier.py
├── Requirements.txt
├── iris_feature_plot.png
└── README.md
```

---

# 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Manoj1234-bot/Iris-Flower-Classifier.git
```

### 2. Navigate to the project

```bash
cd iris-flower-classifier
```

### 3. Install dependencies

```bash
pip install -r Requirements.txt
```

Required libraries:

- pandas
- numpy
- scikit-learn
- matplotlib

---

# ▶️ Run the Project

```bash
python classifier.py
```

---

# 🌸 Dataset

This project uses the **Iris Dataset**, one of the most popular datasets for machine learning classification.

It contains:

- **150 flower samples**
- **3 Iris species**
  - Setosa
  - Versicolor
  - Virginica
- **4 numerical features**
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width

The dataset is built into **Scikit-learn**, so no external download is required.

---

# 🤖 Machine Learning Models Used

The project trains and compares the following classifiers:

- 🌟 K-Nearest Neighbors (KNN)
- 🌳 Decision Tree
- 📈 Logistic Regression

The model with the highest accuracy is automatically selected for prediction.

---

# ⚙️ How It Works

1. Loads the Iris dataset from Scikit-learn.
2. Splits the dataset into training and testing sets.
3. Applies feature scaling using StandardScaler.
4. Trains three different classification algorithms.
5. Compares model accuracy.
6. Displays:
   - Accuracy
   - Confusion Matrix
   - Classification Report
7. Creates scatter plots showing feature relationships.
8. Allows users to enter custom flower measurements.
9. Predicts the flower species with confidence scores.

---

# 📊 Model Evaluation

The project evaluates models using:

- ✅ Accuracy Score
- ✅ Confusion Matrix
- ✅ Classification Report

The best-performing classifier is selected automatically for predictions.

---

# 📈 Visualization

The project generates an informative scatter plot showing feature relationships.

### Generated Graph

- Sepal Length vs Sepal Width
- Petal Length vs Petal Width

The visualization helps understand how different flower species are separated based on their measurements.

---

# 🌼 Sample Prediction

```text
Predict a custom flower's species?

Sepal Length : 5.8
Sepal Width  : 3.0
Petal Length : 4.2
Petal Width  : 1.3

Predicted Species:

Versicolor

Confidence:

Setosa       : 0.0%
Versicolor   : 96.7%
Virginica    : 3.3%
```

---

# 📚 Concepts Used

- Supervised Machine Learning
- Classification
- Data Preprocessing
- Feature Scaling
- Train-Test Split
- Accuracy Evaluation
- Confusion Matrix
- Classification Report
- Data Visualization
- Probability Prediction

---

# 📚 What I Learned

- Learned how to build a supervised machine learning classification model using Scikit-learn by comparing multiple algorithms such as K-Nearest Neighbors, Decision Tree, and Logistic Regression.
- Improved my understanding of feature scaling, train-test splitting, model evaluation using accuracy, confusion matrix, classification reports, data visualization with Matplotlib, and predicting classes using probability scores.

---

# 🚀 Future Improvements

- 🌐 Flask Web Application
- 📱 Responsive Frontend
- ☁️ Deploy on Render or Railway
- 📊 Interactive Dashboard
- 💾 Save Trained Model using Joblib
- 🤖 Add Support Vector Machine (SVM)
- 🌳 Random Forest Classifier
- 🚀 XGBoost Classifier
- 📈 Cross Validation
- 🔍 Hyperparameter Tuning

---

# 👨‍💻 Author

**Manoj S**

BCA Final Year Student

Passionate about Web Development, Python, Machine Learning, Data Science, and Open Source.

---

# ⭐ Show Your Support

If you found this project useful, please consider giving it a ⭐ **Star** on GitHub.

It motivates me to build more real-world Machine Learning projects.

---

# 📜 License

This project is licensed under the **MIT License**.

Feel free to use, modify, and share it for learning and educational purposes.

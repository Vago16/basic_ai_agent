# Position Salary Prediction (PyTorch Regression Model)

This project uses a simple neural network built with PyTorch to predict salaries based on position levels using a small dataset. It demonstrates data preprocessing, normalization, model training, and visualization.

---

##  Dataset

The dataset (`Position_Salaries.csv`) contains:
- Position level
- Corresponding salary

Example:

| Position | Level | Salary |
|----------|-------|--------|
| Intern   | 1     | 45000  |
| Junior   | 2     | 50000  |
| Senior   | 3     | 60000  |

---

## Requirements listed

- Python
- Pandas
- PyTorch
- Scikit-learn
- Matplotlib

---

## Model Architecture

A simple feedforward neural network:

- Input layer: 1 feature (position level)
- Hidden layer: 8 neurons + ReLU
- Output layer: 1 value (salary prediction)

---

#

1. Load dataset using Pandas  
2. Convert data into tensors  
3. Split into training and test sets  
4. Standardize features using `StandardScaler`  
5. Train neural network using MSE loss  
6. Evaluate model performance  
7. Plot predictions vs actual data(accuracy increased with number of epochs till about 300)  

---

## 🚀 How to Run

1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Run the model
```python model.py ```

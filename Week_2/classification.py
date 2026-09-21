# Data Classification Using K-Nearest Neighbors (KNN)

## Project Title
Data Classification Using AI — Iris Dataset with K-Nearest Neighbors

## Objective
Build a beginner-friendly supervised learning project that demonstrates:
- Loading and inspecting a public dataset
- Data preprocessing (scaling, train-test split)
- Training a K-Nearest Neighbors classifier
- Evaluating model performance using multiple metrics
- Generating visualizations (confusion matrix heatmap)

## Dataset Description
The **Iris Dataset** is a classic machine learning dataset containing 150 samples of iris flower measurements.

### Dataset Properties
- **Total Samples**: 150
- **Classes**: 3 (Setosa, Versicolor, Virginica)
- **Features per Sample**: 4
- **Feature Names**:
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)

### Target Classes
- **0**: Setosa
- **1**: Versicolor
- **2**: Virginica

## Workflow

1. **Data Loading & Inspection** → Load Iris dataset, verify shape and classes
2. **Train-Test Split** → Divide into 80% training (120 samples) and 20% testing (30 samples) using stratified split
3. **Feature Scaling** → Apply StandardScaler to normalize feature ranges (fit on training data only)
4. **Model Training** → Train K-Nearest Neighbors with K=5 on scaled training data
5. **Prediction** → Generate predictions on test set
6. **Evaluation** → Calculate accuracy, F1 score, confusion matrix, and classification report
7. **Visualization** → Plot and save confusion matrix heatmap

## Algorithm: K-Nearest Neighbors (KNN)

### How KNN Works
K-Nearest Neighbors is a simple, non-parametric supervised learning algorithm:
1. Given a test sample, find the K closest samples in the training data (based on Euclidean distance)
2. The prediction is the majority class among these K neighbors
3. For Iris classification, K=5 means we check the 5 nearest training samples

### Advantages
- Simple to understand and implement
- No training phase (lazy learner)
- Works well for small to medium datasets
- Non-parametric (makes no assumptions about data distribution)

### Limitations
- Computationally expensive for large datasets (must check all training samples)
- Sensitive to feature scaling (why we use StandardScaler)
- Performance depends on choosing the right K value
- High memory usage for large training sets

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Setup
1. Clone or download this repository
2. Navigate to the project directory:
   ```bash
   cd project-2-data-classification
   ```
3. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### Run the Classification Pipeline
```bash
cd src
python classification.py
```

This will:
- Load and inspect the Iris dataset
- Split data into training (80%) and testing (20%) sets
- Scale features using StandardScaler
- Train a KNN classifier (K=5)
- Print evaluation metrics (accuracy, F1 score)
- Print confusion matrix and classification report
- Generate and save `outputs/confusion_matrix.png`

### Run the Test Suite
```bash
pytest tests/
```

Or with verbose output:
```bash
pytest tests/ -v
```

Tests verify:
- Dataset shape and class distribution
- Train-test split ratios and reproducibility
- Feature scaling application
- Model type and K value
- Prediction counts and validity
- Evaluation metric ranges
- Confusion matrix structure

## Evaluation Metrics

### Accuracy
- Percentage of correct predictions out of total predictions
- Range: 0 to 1 (0% to 100%)
- Formula: `(True Positives + True Negatives) / Total Samples`

### Weighted F1 Score
- Harmonic mean of precision and recall, weighted by class support
- Range: 0 to 1
- Useful for imbalanced datasets
- Formula: `2 * (Precision * Recall) / (Precision + Recall)`

### Confusion Matrix
- 3×3 grid showing true vs predicted class distribution
- Diagonal elements = correct predictions
- Off-diagonal elements = misclassifications
- Helps identify which classes are confused with each other

### Classification Report
- Precision, recall, F1 score per class
- Shows per-class performance breakdown

## Expected Generated Output

### Console Output
```
============================================================
DATASET INSPECTION
============================================================
Dataset shape: (150, 4)
Number of features: 4
Number of samples: 150
Number of classes: 3
Class labels: [0 1 2]
Class names: ['setosa' 'versicolor' 'virginica']
Feature names: ['sepal length (cm)', 'sepal width (cm)', ...]

...

============================================================
MODEL EVALUATION
============================================================
Accuracy: 1.0000
Weighted F1 Score: 1.0000

Confusion Matrix:
[[10  0  0]
 [ 0 10  0]
 [ 0  0 10]]

Classification Report:
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00        10
   virginica       1.00      1.00      1.00        10
    accuracy                           1.00        30
   macro avg       1.00      1.00      1.00        30
weighted avg       1.00      1.00      1.00        30
```

### Generated Files
- `outputs/confusion_matrix.png` — Heatmap visualization of confusion matrix

## Project Structure
```
project-2-data-classification/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore patterns
├── src/
│   └── classification.py              # Main classification module
├── tests/
│   └── test_classification.py         # Pytest test suite
├── docs/
│   ├── PROJECT_SPEC.md                # Project specification
│   ├── REPORT_OUTLINE.md              # Report template
│   └── ACCEPTANCE_TESTS.md            # Acceptance test criteria
└── outputs/
    └── .gitkeep                       # Placeholder for generated outputs
```

## Limitations

### Model Limitations
- **Curse of Dimensionality**: KNN performance degrades in high-dimensional spaces
- **K=5 Selection**: The K value was fixed at 5 (could be optimized via cross-validation)
- **No Feature Selection**: Uses all 4 features equally (could benefit from feature importance analysis)
- **Symmetric Distance**: Euclidean distance treats all features equally (could use weighted distances)

### Dataset Limitations
- **Small Dataset**: 150 samples is relatively small (results may not generalize to larger datasets)
- **Balanced Classes**: Each class has exactly 50 samples (doesn't test imbalanced scenarios)
- **Simple Features**: Features are manually collected measurements (not high-dimensional)

### Implementation Limitations
- **No Cross-Validation**: Uses single 80-20 split (doesn't use k-fold for robustness)
- **No Hyperparameter Tuning**: K=5 fixed (could search for optimal K)
- **No Class Weighting**: All classes treated equally (works because dataset is balanced)

## Learning Outcomes

After completing this project, you should understand:
1. How to load and inspect datasets using scikit-learn
2. Why and how to scale features (data normalization)
3. The importance of stratified train-test splits
4. How K-Nearest Neighbors algorithm works
5. What accuracy, precision, recall, and F1 score mean
6. How to interpret a confusion matrix
7. Best practices for preventing data leakage
8. How to structure a Python ML project for reproducibility
9. How to write tests for ML code
10. How to visualize model performance

## Files Included
- `src/classification.py` — Main pipeline (runnable script)
- `tests/test_classification.py` — 20+ pytest test cases
- `requirements.txt` — Dependencies (scikit-learn, numpy, matplotlib, seaborn, pytest)
- `README.md` — This documentation
- `docs/` — Report template and specification files

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python src/classification.py

# Run tests
pytest tests/ -v
```

## Author
Generated as a beginner supervised learning project.

## License
Educational use.

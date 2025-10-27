# Diabetes Prediction using K-Nearest Neighbors (KNN)

A machine learning classification project that predicts diabetes occurrence using the K-Nearest Neighbors algorithm. This project implements a complete ML pipeline including exploratory data analysis, hyperparameter tuning, and model evaluation using various medical and demographic features.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Technologies Used](#technologies-used)
- [Results](#results)

## Overview

This project uses machine learning to predict whether a patient has diabetes based on diagnostic measurements. The model employs the K-Nearest Neighbors (KNN) classification algorithm with optimized hyperparameters to achieve accurate predictions. The project includes comprehensive data analysis, visualization, and model evaluation.

## Dataset

The project uses a diabetes dataset (Training.csv) containing medical diagnostic measurements for diabetes prediction.

### Features

The dataset includes the following 8 features:

1. **Pregnancies** - Number of times pregnant
2. **Glucose** - Plasma glucose concentration (2 hours in an oral glucose tolerance test)
3. **BloodPressure** - Diastolic blood pressure (mm Hg)
4. **SkinThickness** - Triceps skin fold thickness (mm)
5. **Insulin** - 2-Hour serum insulin (mu U/ml)
6. **BMI** - Body Mass Index (weight in kg/(height in m)^2)
7. **DiabetesPedigreeFunction** - Diabetes pedigree function (genetic influence)
8. **Age** - Age in years

### Target Variable

- **Outcome** - Binary classification (0: No diabetes, 1: Diabetes)

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Or create a `requirements.txt` file:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install using:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Project

1. Clone the repository:
```bash
git clone <repository-url>
cd diabetes-prediction
```

2. Ensure your dataset file (`Training.csv`) is in the correct location or update the file path in the script:
```python
df = pd.read_csv("Training.csv")  # Update path as needed
```

3. Run the main script:
```bash
python diabetes_prediction.py
```

### What the Script Does

1. **Data Loading & Exploration**
   - Loads the diabetes dataset
   - Displays basic statistics and data structure
   - Checks for missing values

2. **Exploratory Data Analysis (EDA)**
   - Creates correlation heatmap
   - Generates outcome distribution plot
   - Produces box plots for outlier detection
   - Creates pair plots with outcome classification
   - Generates histograms with KDE for feature distributions

3. **Data Preprocessing**
   - Standardizes features using StandardScaler
   - Splits data into training (80%) and testing (20%) sets

4. **Model Training & Optimization**
   - Tests KNN algorithm with k values from 1 to 19
   - Identifies optimal k value
   - Trains final model with best k value

5. **Model Evaluation**
   - Generates confusion matrix
   - Produces detailed classification report

## Project Structure

```
diabetes-prediction/
│
├── diabetes_prediction.py    # Main script with complete pipeline
├── Training.csv              # Dataset file (not included in repo)
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

## Model Performance

### K-Nearest Neighbors (KNN) Algorithm

The project implements KNN classification with hyperparameter tuning:

- **Algorithm**: K-Nearest Neighbors Classifier
- **Optimal k value**: Determined through iterative testing (k=1 to k=19)
- **Final model**: k=13 (as determined by the script)

### Evaluation Metrics

The model is evaluated using:

- **Accuracy Score**: Overall prediction accuracy
- **Confusion Matrix**: True/False Positives and Negatives
- **Classification Report**:
  - Precision: Accuracy of positive predictions
  - Recall: Ability to find all positive cases
  - F1-Score: Harmonic mean of precision and recall
  - Support: Number of samples in each class

### Model Pipeline

1. **Data Exploration**: Understand dataset characteristics
2. **Visualization**: Identify patterns and correlations
3. **Feature Scaling**: Standardize features using StandardScaler
4. **Train-Test Split**: 80-20 split for model validation
5. **Hyperparameter Tuning**: Test multiple k values (1-19)
6. **Model Selection**: Choose optimal k value
7. **Final Training**: Train model with best k value
8. **Evaluation**: Comprehensive performance metrics

## Technologies Used

- **Python 3.x** - Programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning library
  - KNeighborsClassifier
  - StandardScaler
  - train_test_split
  - confusion_matrix
  - classification_report
- **Matplotlib** - Plotting library
- **Seaborn** - Statistical data visualization

## Results

### Visualizations Generated

1. **Correlation Heatmap** (12x6)
   - Shows relationships between all features
   - Helps identify multicollinearity

2. **Outcome Distribution** (12x6)
   - Count plot showing class balance
   - Identifies potential class imbalance issues

3. **Box Plots** (12x12)
   - Individual box plots for key features
   - Identifies outliers and data distribution

4. **Pair Plot**
   - Comprehensive feature relationship visualization
   - Color-coded by outcome (diabetes/no diabetes)

5. **Histograms with KDE** (12x12)
   - Distribution plots for key features
   - Kernel Density Estimation overlay

### Model Insights

- The script automatically identifies the optimal k value by testing multiple configurations
- Both training and testing scores are tracked to detect overfitting
- The confusion matrix provides insights into false positives and false negatives
- The classification report offers detailed per-class performance metrics

## Key Features

✅ Comprehensive exploratory data analysis  
✅ Automated hyperparameter tuning for KNN  
✅ Feature standardization for improved model performance  
✅ Multiple visualization techniques  
✅ Detailed model evaluation metrics  
✅ Clean and well-documented code  

## Future Improvements

- Implement cross-validation for more robust evaluation
- Try additional classification algorithms (Random Forest, SVM, Logistic Regression)
- Perform feature selection/engineering
- Address class imbalance if present (SMOTE, class weights)
- Create a prediction interface (web app or CLI)
- Add model persistence (save/load trained model)
- Implement grid search for more thorough hyperparameter tuning
- Add ROC curve and AUC score analysis

## Important Notes

- Ensure the dataset path is correctly specified before running the script
- The model uses standardized features, so new predictions must also be standardized using the same scaler
- Results may vary slightly due to random train-test split (consider setting random_state for reproducibility)

## Dataset Requirements

To run this project, you need a CSV file named `Training.csv` with the following columns:
- Pregnancies
- Glucose
- BloodPressure
- SkinThickness
- Insulin
- BMI
- Age
- DiabetesPedigreeFunction
- Outcome

## License

This project is open source and available under the MIT License.

## Contact

For questions, suggestions, or feedback, please open an issue in the repository.

---

**Disclaimer**: This project is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis or advice.

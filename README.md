# 🌳 Task 5: Decision Trees and Random Forests (Machine Learning Project)

## 🎯 Objective
This task focuses on learning and implementing *tree-based models* — primarily *Decision Trees* and *Random Forests* — for solving a *binary classification problem* related to loan approvals. We will analyze, visualize, and evaluate how well these models predict whether a loan will be approved or not.

---

## 🧰 Tools & Technologies Used
| Tool / Library     | Purpose                                      |
|--------------------|----------------------------------------------|
| *Python 3.x*     | Programming language                         |
| *Scikit-learn*   | ML models, preprocessing, evaluation         |
| *Graphviz*       | Tree visualization (better than matplotlib) |
| *Matplotlib & Seaborn* | Plotting and visual analysis             |
| *Pandas & NumPy* | Data manipulation                           |

---

## 📁 Project Files
| File Name         | Description |
|-------------------|-------------|
| task5.py        | Main Python script for this task |
| loan_data.csv   | Dataset used to train and test models |
| loan_tree.png   | Visualization of the trained decision tree |
| README.md       | This documentation file |

---

## 🔎 Dataset Overview

The dataset (loan_data.csv) contains personal, financial, and loan-related information for individuals. Some key features:
- person_age, person_income, loan_amnt, etc. (numerical)
- person_education, person_home_ownership, etc. (categorical)
- loan_status: *Target variable* (1 = loan approved, 0 = rejected)

The task is to **predict loan_status** using all other features.

---

## 🧪 What This Script Does (Step-by-Step)

### 1. *Data Preprocessing*
- Categorical features like gender and education are *converted into numbers* using LabelEncoder.
- Features (X) and target (y) are split.
- Then, data is split into training (70%) and testing (30%).

### 2. *Decision Tree Classifier*
- A *Decision Tree* is trained using the training data.
- Tree depth is limited (max_depth=4) to avoid overfitting.
- Accuracy is calculated on test data.
- Tree is *visualized* using both:
  - matplotlib (tree.plot_tree)
  - Graphviz (graphviz.Source)

### 3. *Random Forest Classifier*
- A *Random Forest* (ensemble of 100 decision trees) is trained.
- Performance is evaluated and compared with the Decision Tree.

### 4. *Feature Importance*
- Random Forest gives *importance scores* for each feature.
- A bar plot shows which features contribute most to the decision-making.

### 5. *Cross-Validation*
- A 5-fold *cross-validation* is performed to test how well the model generalizes.

---

## 📊 Results

### ✅ Accuracy Scores
| Model              | Accuracy (Test Set) |
|-------------------|---------------------|
| Decision Tree      | ~90.83%             |
| Random Forest      | Higher (Exact value printed in output) |

### 🔝 Top 5 Important Features (from Random Forest)
1. previous_loan_defaults_on_file
2. loan_percent_income
3. loan_int_rate
4. person_income
5. person_home_ownership

These features have the most influence on predicting loan approval.

---

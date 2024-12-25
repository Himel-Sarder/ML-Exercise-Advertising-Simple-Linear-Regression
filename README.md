# ML-Exercise-Advertising-Simple-Linear-Regression
Dataset - https://www.kaggle.com/code/ashydv/sales-prediction-simple-linear-regression/input   

![image](https://github.com/user-attachments/assets/f7cbf4a0-98d5-4cef-9493-b6061c80b2f2)
# Advertising Dataset Analysis

## Project Overview
This project involves analyzing Kaggle's Advertising Dataset to build a simple linear regression model that predicts sales based on advertising spend. The dataset includes columns for TV, Radio, Newspaper advertising budgets, and Sales.

## Author
- **Name**: Himel Sarder
- **Email**: info.himelcse@gmail.com
- **GitHub**: [Himel-Sarder](https://github.com/Himel-Sarder)

## Objective
The goal of this project is to:
1. Explore the relationship between advertising spend on TV and sales.
2. Build a linear regression model to predict sales based on TV advertising.
3. Evaluate the model's performance using statistical metrics.

## Dataset
The dataset contains the following columns:
- **TV**: Advertising budget for TV (in thousands of dollars).
- **Radio**: Advertising budget for Radio (in thousands of dollars).
- **Newspaper**: Advertising budget for Newspaper (in thousands of dollars).
- **Sales**: Sales generated (in thousands of units).

### Sample Data
| TV     | Radio  | Newspaper | Sales |
|--------|--------|-----------|-------|
| 230.1  | 37.8   | 69.2      | 22.1  |
| 44.5   | 39.3   | 45.1      | 10.4  |
| 17.2   | 45.9   | 69.3      | 12.0  |
| 151.5  | 41.3   | 58.5      | 16.5  |
| 180.8  | 10.8   | 58.4      | 17.9  |

## Steps

### 1. Import Libraries
The following Python libraries are used:
- `warnings` to suppress warnings.
- `matplotlib.pyplot` and `seaborn` for data visualization.
- `pandas` and `numpy` for data manipulation.
- `math` for mathematical operations.
- `sklearn` for machine learning tasks.
- `statsmodels` for statistical modeling.

### 2. Load the Dataset
The dataset is loaded using:
```python
import pandas as pd
df = pd.read_csv('advertising.csv')
```

### 3. Exploratory Data Analysis (EDA)
- Scatter plot to visualize the relationship between TV advertising and Sales:
```python
plt.figure(figsize=(12, 8))
plt.scatter(df['TV'], df['Sales'])
plt.xlabel('TV')
plt.ylabel('Sales')
plt.show()
```

### 4. Data Preparation
- Features (`X`) and target (`y`) are defined as:
```python
X = df.iloc[:, 0:1]  # TV column
y = df.iloc[:, -1]   # Sales column
```
- Split the dataset into training and testing sets:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
```

### 5. Model Training
- A linear regression model is trained using:
```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
```

### 6. Model Evaluation
- The model's performance is evaluated using:
  - **Mean Squared Error (MSE)**
  - **Root Mean Squared Error (RMSE)**
  - **R-squared**
```python
from sklearn.metrics import mean_squared_error
predictions = lr.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse}")
```
- The regression equation is printed:
```python
print(f"The linear model is: Y = {lr.intercept_:.5f} + {lr.coef_[0]:.5f}X")
```

### 7. Statistical Analysis
- The model is refitted using `statsmodels` for detailed statistical insights:
```python
import statsmodels.api as sm
X_train_sm = sm.add_constant(X_train)
lr_sm = sm.OLS(y_train, X_train_sm).fit()
print(lr_sm.summary())
```

### 8. Error Analysis
- Residuals are analyzed to ensure the model's assumptions are met:
```python
res = y_train - lr_sm.predict(X_train_sm)
sns.distplot(res, bins=15)
plt.title('Error Terms')
plt.xlabel('y_train - predictions')
plt.show()
```

## Results
- The regression equation is:
  ```
  Y = 6.94868 + 0.05455X
  ```
- Key performance metrics:
  - **RMSE**: 2.019
  - **R-squared**: 0.816
- The model explains approximately 81.6% of the variance in sales based on TV advertising.

## Conclusion
- TV advertising has a significant positive relationship with sales.
- The linear regression model is effective in predicting sales based on TV advertising spend.

## References
- Kaggle's Advertising Dataset
- Python libraries: `sklearn`, `statsmodels`, `matplotlib`, `seaborn`, `pandas`, `numpy`

---
For more details, visit [GitHub](https://github.com/Himel-Sarder).


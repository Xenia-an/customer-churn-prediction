# Customer Churn Prediction

## Project Overview

Цель проекта — выявить факторы, влияющие на отток клиентов телеком-компании, и построить модель для прогнозирования вероятности ухода клиента.

## Dataset

7043 клиентов, 21 признак

Целевая переменная: Churn

## Business Requirements

* Выявить факторы, влияющие на отток клиентов (потребительский churn).
* Построить модель для прогнозирования риска оттока.
* Разработать рекомендации по улучшению удержания клиентов.

## Approach
1. Exploratory Data Analysis (EDA)
2. Data Cleaning
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Business Recommendations

## Models

| Model               | ROC-AUC              |
| ------------------- | -------------------- |
| Logistic Regression | 0.84                 |
| Random Forest       | 0.82                 |

## Key Findings

Наиболее значимые факторы оттока:

* tenure
* MonthlyCharges
* TotalCharges
* Contract
* PaymentMethod

## Business Recommendations

* стимулировать долгосрочные контракты
* развивать автоматические платежи
* улучшать качество Fiber Optic услуг
* уделять внимание новым клиентам

## Technologies

Python, pandas, scikit-learn, matplotlib, seaborn


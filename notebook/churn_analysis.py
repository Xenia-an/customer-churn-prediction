#!/usr/bin/env python
# coding: utf-8

# In[71]:


import pandas as pd
import numpy as np
import sklearn as sk

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\Kseniya\Desktop\customer churn prediction\data\WA_Fn-UseC_-Telco-Customer-Churn.csv")
df.head()


# In[72]:


df.shape


# In[73]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.info()


# In[74]:


df.describe()


# In[75]:


#Churn — ушёл клиент или нет
#Все остальные колонки, кроме Churn и customerID


# In[76]:


df['Churn'].value_counts()


# In[77]:


df['Churn'].value_counts(normalize=True) * 100


# In[86]:


sns.countplot(x='Churn', data=df)
plt.title('Churn distribution')
plt.savefig(r'C:\Users\Kseniya\Desktop\customer churn prediction\images\\' + 'Churn Distribution.png', dpi=300) 
plt.show()


# In[79]:


num_features = ['tenure', 'MonthlyCharges', 'TotalCharges']


# In[80]:


sns.boxplot(x='Churn', y='tenure', data=df)
plt.title('Tenure vs Churn')
plt.show()


# In[87]:


for col in num_features:
    sns.boxplot(x='Churn', y=col, data=df)
    plt.title(f'{col} vs Churn')
    plt.savefig(r'C:\Users\Kseniya\Desktop\customer churn prediction\images\\' + f'{col}_vs_Churn.png', dpi=300, bbox_inches='tight')
    plt.show()


# In[37]:


cat_features = df.select_dtypes(include='object').columns.tolist()
cat_features.remove('Churn')
cat_features.remove('customerID')


# In[38]:


contract_churn = (
    df.groupby('Contract')['Churn']
    .value_counts(normalize=True)
    .rename('share')
    .reset_index()
)

contract_churn


# In[39]:


sns.barplot(
    data=contract_churn,
    x='Contract',
    y='share',
    hue='Churn'
)
plt.title('Churn rate by Contract type')
plt.ylabel('Share')
plt.show()


# In[40]:


internet_churn = (
    df.groupby('InternetService')['Churn']
    .value_counts(normalize=True)
    .rename('share')
    .reset_index()
)

sns.barplot(
    data=internet_churn,
    x='InternetService',
    y='share',
    hue='Churn'
)
plt.title('Churn by Internet Service')
plt.show()


# In[41]:


payment_churn = (
    df.groupby('PaymentMethod')['Churn']
    .value_counts(normalize=True)
    .rename('share')
    .reset_index()
)

sns.barplot(
    data=payment_churn,
    x='PaymentMethod',
    y='share',
    hue='Churn'
)
plt.xticks(rotation=45)
plt.title('Churn by Payment Method')
plt.show()


# In[42]:


for col in ['OnlineSecurity', 'TechSupport', 'Partner']:
    temp = (
        df.groupby(col)['Churn']
        .value_counts(normalize=True)
        .rename('share')
        .reset_index()
    )

    sns.barplot(data=temp, x=col, y='share', hue='Churn')
    plt.title(f'Churn by {col}')
    plt.show()


# In[43]:


df.isna().sum()


# In[44]:


df['TotalCharges'] = df['TotalCharges'].fillna(0)


# In[45]:


X = df.drop(columns=['Churn', 'customerID'])
y = df['Churn']
y = y.map({'Yes': 1, 'No': 0})


# In[46]:


num_features = X.select_dtypes(include=['int64', 'float64']).columns
cat_features = X.select_dtypes(include='object').columns


# In[47]:


df.info()


# In[48]:


from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', num_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
    ]
)


# In[49]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# In[50]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])


# In[51]:


from sklearn.preprocessing import StandardScaler
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_features)
    ]
)


# In[52]:


model.fit(X_train, y_train)


# In[53]:


y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


# In[54]:


from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report


# In[55]:


accuracy_score(y_test, y_pred)


# In[56]:


roc_auc_score(y_test, y_proba)


# In[57]:


confusion_matrix(y_test, y_pred)


# In[58]:


print(classification_report(y_test, y_pred))


# In[59]:


from sklearn.ensemble import RandomForestClassifier
model_rf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


# In[60]:


model_rf.fit(X_train, y_train)


# In[61]:


y_pred_rf = model_rf.predict(X_test)
y_proba_rf = model_rf.predict_proba(X_test)[:, 1]


# In[62]:


roc_auc_score(y_test, y_proba_rf)


# In[63]:


import pandas as pd

# достаём модель из pipeline
rf_model = model_rf.named_steps['classifier']

# получаем признаки после OneHotEncoding
feature_names = model_rf.named_steps['preprocessor'].get_feature_names_out()

# важности
importances = rf_model.feature_importances_

# собираем в DataFrame
feat_imp = pd.DataFrame({
    'feature': feature_names,
    'importance': importances
}).sort_values(by='importance', ascending=False)

feat_imp.head(10)


# In[64]:


##на churn сильнее всего влияют финансовые факторы (TotalCharges, MonthlyCharges) и длительность использования (tenure).
##Также важны тип контракта и способ оплаты — клиенты с краткосрочными контрактами и электронными платежами чаще уходят.


# In[65]:


## Final Conclusion

## Построена модель для предсказания churn
## Лучший результат показала Logistic Regression (ROC-AUC ≈ 0.84)
## Основные факторы churn: - Длительность использования (tenure) - Стоимость услуг (MonthlyCharges, TotalCharges)  - Тип контракта  - Способ оплаты

### Бизнес-рекомендации:стимулировать долгосрочные контракты, снижать стоимость для новых клиентов, продвигать автоплатежи, улучшать качество Fiber optic услуг


# In[ ]:





# In[ ]:





# In[ ]:





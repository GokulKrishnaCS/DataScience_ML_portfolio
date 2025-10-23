# Insurance Claim Classification – Predicting Customer Decisions

This project focuses on predicting whether a customer will opt for a holiday package based on their demographic and family information. The goal is to help a travel company identify the most promising customers to target, reduce marketing costs, and improve sales efficiency.

This work demonstrates practical data analysis, feature engineering, and classification model building using Logistic Regression and Linear Discriminant Analysis (LDA). The entire process follows a clear structure — from understanding the data to drawing actionable business insights.

---

## 1. Business Context

A tour and travel agency provides holiday packages to corporate employees.  
The dataset includes details of **872 employees**, such as age, education level, salary, and the number of children. Some employees accepted the holiday package, while others did not.

The company wants to:
- Predict which employees are likely to accept the offer.
- Understand the most important factors influencing acceptance.
- Design better, data-driven marketing strategies.

By accurately identifying interested customers, the company can:
- Increase package conversion rates.
- Personalize offers and focus on the right audience.
- Improve resource allocation in marketing campaigns.

---

## 2. Project Objective

The goal is to build and evaluate classification models that can predict whether an employee will buy a holiday package.  
Two models were developed and compared:
1. **Logistic Regression**
2. **Linear Discriminant Analysis (LDA)**

The model performance was evaluated using Accuracy, Precision, Recall, ROC Curve, and AUC Score.

---

## 3. Methodology

### Step 1: Data Exploration
- Verified **no missing values** and consistent data types.  
- Identified **5 numerical** and **2 categorical** variables.  
- Conducted **univariate and bivariate analysis** to understand patterns.

### Step 2: Data Preparation
- Encoded binary categorical variables (`Yes/No`) into numeric form.  
- Checked for **outliers** and treated extreme values for better model learning.  
- Split the dataset into **70% training** and **30% testing** sets.

### Step 3: Model Building
- Applied **Logistic Regression** and **Linear Discriminant Analysis (LDA)** using scikit-learn.  
- Both models were trained on the same data split for fair comparison.

### Step 4: Model Evaluation
| Model | Accuracy (Train) | Accuracy (Test) | AUC (Train) | AUC (Test) |
|--------|------------------|----------------|--------------|-------------|
| Logistic Regression | 0.55 | 0.53 | 0.709 | 0.717 |
| Linear Discriminant Analysis | 0.68 | 0.68 | 0.73 | 0.74 |

The **LDA model** performed better overall, showing higher accuracy and AUC with balanced precision and recall. Both models were stable, indicating no overfitting or underfitting.

---

## 4. Results and Key Insights

### Technical Findings
1. **LDA outperformed Logistic Regression**, achieving higher accuracy (0.68) and a better ROC-AUC score.  
   It handled overlapping class distributions more effectively.
2. Model performance was consistent between training and test sets, showing that the data was well-prepared and models generalized properly.
3. The classification workflow demonstrates core data science steps — encoding, model training, evaluation, and interpretation — all in a simple, reproducible format.

### Business Insights
1. **Number of Young Children** had a **negative impact** on accepting the holiday package.  
   Parents with small children are less likely to travel.  
   *Recommendation:* Offer family care or childcare tie-ups during promotions to increase acceptance.
2. **Number of Older Children** had a **positive relationship** with acceptance.  
   Families with older kids are more likely to take vacations.  
   *Recommendation:* Design packages suited for older children and family travel experiences.
3. **Salary and Education** showed a moderate correlation.  
   Higher-income employees are more likely to purchase premium packages.  
   *Recommendation:* Focus marketing efforts in high-income workplaces or premium residential areas.

---

## 5. Business Impact

By using this predictive model:
- The company can **target the right customers**, improving conversion efficiency.
- Marketing efforts become **more focused**, reducing campaign costs.
- The model provides **data-driven insights** for designing offers that appeal to the right demographic.

---

## 6. Technical Skills Demonstrated

- **Exploratory Data Analysis (EDA):** Univariate and bivariate analysis with visualization.  
- **Data Preprocessing:** Encoding categorical variables, handling outliers, and preparing clean datasets.  
- **Model Building:** Logistic Regression and Linear Discriminant Analysis.  
- **Performance Evaluation:** Accuracy, Confusion Matrix, Precision, Recall, ROC Curve, and AUC.  
- **Interpretation:** Translating model outcomes into meaningful business recommendations.


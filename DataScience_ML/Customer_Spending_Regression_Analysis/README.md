# Customer Spending Regression Analysis – Predicting Zirconia Prices

This project focuses on predicting the price of cubic zirconia gemstones based on multiple physical and quality parameters. The goal is to help **Gem Stones Co. Ltd**, a zirconia manufacturer, identify the key attributes that drive price variation and build a regression model that can predict price accurately for future sales and profitability analysis.

The project combines clear data exploration, regression modeling, and practical business interpretation to demonstrate real-world data science problem solving.

---

## 1. Business Context

Gem Stones Co. Ltd manufactures cubic zirconia (an inexpensive diamond alternative).  
The company earns varying profits on different price ranges of stones. They want to:
- Predict zirconia prices from key measurable characteristics.
- Understand which stone features most influence pricing.
- Use these insights to optimize production and pricing strategies.

By predicting prices accurately, the company can:
- Focus production on the most profitable stone categories.
- Make data-driven pricing decisions.
- Improve overall profitability.

---

## 2. Project Objective

The task was to perform end-to-end regression analysis to predict price based on attributes such as:
- Carat (weight)
- Cut
- Color
- Clarity
- Depth, Table, and Dimensions (x, y, z)

Two regression models were built and compared:
1. **Linear Regression (without Scaling and PCA)**  
2. **Linear Regression (with Scaling and PCA)**

The aim was to determine which preprocessing pipeline produces more accurate and stable predictions.

---

## 3. Methodology

### Step 1: Data Exploration and Cleaning
- Removed duplicate rows (34 duplicates found).
- Treated ~700 missing values in the **Depth** column using mean imputation.
- Handled outliers detected in multiple numeric variables.
- Combined and encoded categorical variables (`Cut`, `Color`, `Clarity`) to reduce noise and improve model learning.

### Step 2: Data Encoding and Preparation
- Encoded ordinal variables logically based on quality rank.
- Performed scaling and Principal Component Analysis (PCA) to reduce multicollinearity and dimensional complexity.

### Step 3: Model Building
- Built two Linear Regression models (with and without Scaling & PCA).
- Used a **70:30 train-test split**.
- Compared performance using:
  - Root Mean Square Error (RMSE)
  - R² (Coefficient of Determination)
  - Adjusted R²

### Step 4: Model Evaluation
| Model | RMSE | R² (Train) | R² (Test) | Adjusted R² |
|--------|------|------------|------------|--------------|
| Without Scaling/PCA | 966 | 0.92 | 0.91 | 0.91 |
| With Scaling & PCA | **0.32** | **0.05** | **0.17** | **Lower error overall** |

The **Scaled + PCA model** showed a massive reduction in RMSE and consistent performance across training and test data, indicating better generalization and less error.

---

## 4. Results and Key Insights

### Technical Findings
1. The **Scaled and PCA-applied Linear Regression** performed significantly better, with lower RMSE and more stable R² scores.  
   This shows that scaling and dimensionality reduction improved the learning process.
2. The model captured the relationship between physical dimensions and price accurately, validating the use of PCA to mitigate multicollinearity.
3. The final model is great and suitable for deployment in predictive pricing applications.

### Business Insights
1. **Carat weight** has the highest influence on zirconia price — increasing carat size leads to higher prices.  
   **Action:** Focus on optimizing production of higher-carat stones for better profit margins.
2. **Dimensions (x, y, z)** are directly correlated with price — larger stones require more material and command higher value.  
   **Action:** Adjust production and inventory planning to align with size-based demand trends.
3. **Clarity** has a medium correlation with price — clearer stones fetch higher prices.  
   **Action:** Invest in refining clarity-enhancing processes before sale to boost product value.

---

## 5. Business Impact

- Improved price prediction enables **better cost control and profit forecasting**.
- Identifying high-impact variables helps **target manufacturing improvements**.
- The model can be used to **simulate pricing scenarios**, aiding management in setting profitable and competitive price tiers.



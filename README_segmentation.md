# Customer Segmentation Analysis Project

An interactive customer segmentation dashboard using K-Means clustering on behavioral and demographic data.

## 🚀 Features

- **Automated Segmentation**: K-Means clustering with adjustable number of segments
- **RFM Analysis**: Automatic Recency, Frequency, Monetary scoring when available
- **Interactive Visualizations**:
  - PCA scatter plot showing cluster separation
  - Segment distribution (donut chart)
  - RFM radar chart per segment
  - Income vs Spend scatter analysis
  - Gender & channel demographics by segment
- **Auto-Generated Insights**: Business recommendations extracted from segment profiles
- **Data Upload**: Supports CSV and Excel files with auto column detection
- **Download Results**: Export segmented customer data and summary reports

## 📦 Installation

```bash
pip install -r requirements_segmentation.txt
```

## 🎯 Run the Dashboard

```bash
streamlit run app_segmentation.py
```

## 📁 Project Files

| File | Description |
|------|-------------|
| `app_segmentation.py` | Main Streamlit application |
| `sample_customer_data.csv` | 3,000 synthetic customer records with 4 pre-computed segments |
| `customer_segmentation_dashboard.html` | Static interactive HTML preview (no server needed) |
| `requirements_segmentation.txt` | Python dependencies |
| `README.md` | This file |

## 📊 Data Format

The app auto-detects common column names. For best results, include:

| Column | Alternative Names Accepted |
|--------|---------------------------|
| `age` | Age |
| `income` | Income, Salary, Earnings |
| `total_spend` | Total Spend, Monetary, Revenue |
| `num_orders` | Orders, Frequency, Purchases, Transactions |
| `avg_order_value` | AOV, Avg Order, Average Order |
| `recency_days` | Recency, Days Since, Last Purchase |
| `satisfaction_score` | Satisfaction, Rating, NPS, Score |
| `tenure_months` | Tenure, Membership, Length |
| `website_visits_per_month` | Visits, Website, Engagement |
| `discount_sensitivity` | Discount, Promo, Coupon |
| `gender` | Gender, Sex |
| `region` | Region, Area, City, State |
| `preferred_channel` | Channel, Platform, Device |

## 🧠 Segments Identified (Sample Data)

| Segment | Size | Key Traits | Strategy |
|---------|------|------------|----------|
| **Loyal Customers** | 28.1% | Frequent, recent, medium spend | Reward programs, upselling |
| **Bargain Hunters** | 27.6% | Price-sensitive, low AOV, high visits | Bundle deals, free shipping thresholds |
| **VIP Big Spenders** | 19.6% | High AOV, high income, low frequency | Premium experiences, exclusive access |
| **At Risk** | 24.7% | High recency, low frequency | Win-back campaigns, special offers |

## 🎯 Internship Learning Outcomes

By completing this project, you will demonstrate:

1. **Unsupervised Machine Learning**: Implementing K-Means clustering with scikit-learn
2. **Feature Engineering**: Creating RFM scores, CLV proxies, and engagement metrics
3. **Dimensionality Reduction**: Using PCA for visualization of high-dimensional clusters
4. **Data Visualization**: Multi-chart dashboards with Plotly
5. **Business Interpretation**: Translating cluster profiles into actionable segment names and strategies
6. **Interactive Analytics**: Building adjustable clustering parameters and dynamic filters

## 💡 Extension Ideas

- **Hierarchical Clustering**: Compare K-Means with Agglomerative or DBSCAN
- **Segment Stability**: Track how segments shift over time (cohort analysis)
- **Predictive Segmentation**: Train a classifier to assign new customers to segments
- **Lookalike Modeling**: Find prospects similar to VIPs for ad targeting
- **Churn Prediction**: Use segment + recency to predict churn probability
- **Automated Reporting**: Generate PDF reports with segment profiles

## 🛠️ Tech Stack

- **ML/Analytics**: scikit-learn, pandas, numpy
- **Visualization**: Plotly
- **Dashboard**: Streamlit
- **Data**: Synthetic customer dataset with realistic correlations

## 📄 License

Created for educational and internship purposes.

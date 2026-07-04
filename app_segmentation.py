import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="🎯", layout="wide")

st.markdown("""
    <style>
    .main-header { font-size: 2.4rem; font-weight: bold; color: #0f172a; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #64748b; margin-bottom: 2rem; }
    .segment-card { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-left: 4px solid; margin-bottom: 12px; }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent
SAMPLE_CSV = BASE_DIR / "sample_customer_data.csv"

# ============================================================
# BULLETPROOF DATA LOADING: File OR Auto-Generated Fallback
# ============================================================
@st.cache_data
def generate_sample_data(n=3000):
    np.random.seed(42)
    age = np.random.normal(38, 12, n).clip(18, 75).astype(int)
    income = np.random.lognormal(10.8, 0.5, n).clip(15000, 250000).astype(int)
    tenure = np.random.poisson(24, n).clip(1, 60)
    total_spend = (income * 0.02 + age * 15 + np.random.normal(0, 500, n)).clip(200, 20000).astype(int)
    num_orders = (tenure * 1.5 + np.random.poisson(5, n)).clip(1, 120).astype(int)
    recency = np.random.exponential(30, n).clip(1, 365).astype(int)
    recency = (recency - (total_spend / 1000)).clip(1, 365).astype(int)
    aov = (total_spend / num_orders).clip(20, 500).astype(int)
    satisfaction = (7 + (total_spend / 5000) - (recency / 100) + np.random.normal(0, 0.8, n)).clip(1, 10).round(1)
    visits = (num_orders * 2.5 + np.random.poisson(3, n)).clip(1, 300).astype(int)
    discount = np.random.beta(2, 3, n) * 100

    df = pd.DataFrame({
        'customer_id': [f'CUST-{i+10001}' for i in range(n)],
        'age': age,
        'gender': np.random.choice(['Male', 'Female', 'Non-binary'], n, p=[0.48, 0.48, 0.04]),
        'income': income,
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n, p=[0.25, 0.40, 0.25, 0.10]),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n, p=[0.35, 0.50, 0.15]),
        'tenure_months': tenure,
        'total_spend': total_spend,
        'num_orders': num_orders,
        'avg_order_value': aov,
        'recency_days': recency,
        'satisfaction_score': satisfaction,
        'website_visits_per_month': visits,
        'discount_sensitivity': discount.round(1),
        'preferred_channel': np.random.choice(['Online', 'Mobile App', 'In-Store'], n),
        'pct_electronics': np.random.beta(2, 5, n) * 100,
        'pct_clothing': np.random.beta(3, 4, n) * 100,
        'pct_home_garden': np.random.beta(2, 6, n) * 100,
        'pct_sports': np.random.beta(1.5, 7, n) * 100,
    })
    df['pct_books'] = (100 - df['pct_electronics'] - df['pct_clothing'] - df['pct_home_garden'] - df['pct_sports']).clip(0, 100)
    prefs = df[['pct_electronics', 'pct_clothing', 'pct_home_garden', 'pct_sports', 'pct_books']].values
    prefs = prefs / prefs.sum(axis=1, keepdims=True) * 100
    df[['pct_electronics', 'pct_clothing', 'pct_home_garden', 'pct_sports', 'pct_books']] = prefs.round(1)
    return df

@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

# Sidebar
st.sidebar.title("📁 Data Import")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])

st.sidebar.markdown("---")
st.sidebar.title("⚙️ Clustering Settings")
n_clusters = st.sidebar.slider("Number of Segments", 2, 6, 4)

# Load data: uploaded file > local CSV > auto-generated fallback
if uploaded_file is not None:
    df = load_data(uploaded_file)
    source = f"Uploaded: {uploaded_file.name}"
elif SAMPLE_CSV.exists():
    df = pd.read_csv(SAMPLE_CSV)
    source = "sample_customer_data.csv"
else:
    df = generate_sample_data()
    source = "Auto-generated sample data (3,000 customers)"

st.sidebar.info(f"Using: {source}")

# ============================================================
# AUTO-DETECT COLUMNS (SAFE: no duplicates)
# ============================================================
def auto_detect_columns(df):
    col_map = {}
    used_names = set(df.columns.tolist())  # track already-used target names

    for col in df.columns:
        cl = col.lower().replace(' ', '_')
        target = None
        if 'age' in cl:
            target = 'age'
        elif any(x in cl for x in ['income', 'salary', 'earnings']):
            target = 'income'
        elif any(x in cl for x in ['spend', 'monetary', 'revenue']):
            target = 'total_spend'
        elif any(x in cl for x in ['orders', 'frequency', 'purchases', 'transactions']):
            target = 'num_orders'
        elif any(x in cl for x in ['aov', 'avg_order', 'average_order']):
            target = 'avg_order_value'
        elif any(x in cl for x in ['recency', 'days_since', 'last_purchase']):
            target = 'recency_days'
        elif any(x in cl for x in ['satisfaction', 'rating', 'nps', 'score']):
            target = 'satisfaction_score'
        elif any(x in cl for x in ['tenure', 'membership', 'length']):
            target = 'tenure_months'
        elif any(x in cl for x in ['visits', 'website', 'engagement']):
            target = 'website_visits_per_month'
        elif any(x in cl for x in ['discount', 'promo', 'coupon']):
            target = 'discount_sensitivity'
        elif 'gender' in cl:
            target = 'gender'
        elif any(x in cl for x in ['region', 'area', 'city', 'state']):
            target = 'region'
        elif 'education' in cl:
            target = 'education'
        elif any(x in cl for x in ['marital', 'married', 'status']):
            target = 'marital_status'
        elif any(x in cl for x in ['channel', 'platform', 'device']):
            target = 'preferred_channel'
        elif any(x in cl for x in ['customer_id', 'user_id']):
            target = 'customer_id'

        # Only rename if target doesn't already exist and hasn't been used
        if target and target != col and target not in used_names and col not in col_map:
            col_map[col] = target
            used_names.add(target)

    return col_map

col_map = auto_detect_columns(df)
if col_map:
    df = df.rename(columns=col_map)

# ============================================================
# REMOVE DUPLICATE COLUMNS (safety net)
# ============================================================
if df.columns.duplicated().any():
    dupes = df.columns[df.columns.duplicated()].tolist()
    st.warning(f"Removed duplicate columns: {dupes}")
    df = df.loc[:, ~df.columns.duplicated()]

# Required columns
cluster_cols = ['age', 'income', 'total_spend', 'num_orders', 'avg_order_value',
                'recency_days', 'satisfaction_score', 'tenure_months',
                'website_visits_per_month', 'discount_sensitivity']

available_cols = [c for c in cluster_cols if c in df.columns]

if len(available_cols) < 3:
    st.error("Need at least 3 numeric columns for clustering. Please upload data with customer behavior metrics.")
    st.stop()

# Feature engineering
df['clv_proxy'] = df['total_spend'] / df['tenure_months'].clip(lower=1) * 12 if 'total_spend' in df.columns and 'tenure_months' in df.columns else 0
df['engagement_score'] = (df['website_visits_per_month'] / df['num_orders'].clip(lower=1)).clip(0, 20) if 'website_visits_per_month' in df.columns and 'num_orders' in df.columns else 0

# RFM scoring
rfm_available = False
if 'recency_days' in df.columns and 'num_orders' in df.columns and 'total_spend' in df.columns:
    df['R'] = pd.qcut(df['recency_days'].rank(method='first'), 5, labels=[5,4,3,2,1]).astype(int)
    df['F'] = pd.qcut(df['num_orders'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    df['M'] = pd.qcut(df['total_spend'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    rfm_available = True

# Clustering
features_for_cluster = available_cols.copy()
if 'clv_proxy' in df.columns:
    features_for_cluster.append('clv_proxy')
if 'engagement_score' in df.columns:
    features_for_cluster.append('engagement_score')

X = df[features_for_cluster].fillna(df[features_for_cluster].median())

# Final safety check: ensure no duplicates in feature columns
X = X.loc[:, ~X.columns.duplicated()]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['segment'] = kmeans.fit_predict(X_scaled)

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['pca1'] = pca_result[:, 0]
df['pca2'] = pca_result[:, 1]

# Segment naming
seg_summary = df.groupby('segment').agg({
    'total_spend': 'mean' if 'total_spend' in df.columns else 'count',
    'num_orders': 'mean' if 'num_orders' in df.columns else 'count',
    'recency_days': 'mean' if 'recency_days' in df.columns else 'count',
    'avg_order_value': 'mean' if 'avg_order_value' in df.columns else 'count',
    'customer_id': 'count' if 'customer_id' in df.columns else 'count'
}).round(1)
seg_summary.rename(columns={seg_summary.columns[-1]: 'count'}, inplace=True)

segment_names = {}
for seg in seg_summary.index:
    name = f"Segment {seg}"
    if 'total_spend' in seg_summary.columns and 'num_orders' in seg_summary.columns and 'recency_days' in seg_summary.columns:
        spend = seg_summary.loc[seg, 'total_spend']
        orders = seg_summary.loc[seg, 'num_orders']
        recency = seg_summary.loc[seg, 'recency_days']
        aov = seg_summary.loc[seg, 'avg_order_value'] if 'avg_order_value' in seg_summary.columns else 0

        if spend >= seg_summary['total_spend'].quantile(0.75) and aov >= seg_summary['avg_order_value'].quantile(0.75):
            name = 'VIP Big Spenders'
        elif orders >= seg_summary['num_orders'].quantile(0.75) and recency <= seg_summary['recency_days'].quantile(0.25):
            name = 'Loyal Customers'
        elif recency >= seg_summary['recency_days'].quantile(0.75):
            name = 'At Risk'
        elif aov <= seg_summary['avg_order_value'].quantile(0.25):
            name = 'Bargain Hunters'
        elif spend >= seg_summary['total_spend'].quantile(0.5) and orders >= seg_summary['num_orders'].quantile(0.5):
            name = 'Potential Loyalists'
        else:
            name = 'Regular Customers'
    segment_names[seg] = name

df['segment_name'] = df['segment'].map(segment_names)

# DASHBOARD UI
st.markdown('<div class="main-header">🎯 Customer Segmentation Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">K-Means Clustering on Behavioral & Demographic Data</div>', unsafe_allow_html=True)

total_customers = len(df)
total_revenue = df['total_spend'].sum() if 'total_spend' in df.columns else 0
avg_clv = df['clv_proxy'].mean() if 'clv_proxy' in df.columns else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Total Customers", f"{total_customers:,}")
with kpi2:
    st.metric("Total Revenue", f"${total_revenue:,.0f}") if total_revenue > 0 else st.metric("Segments", f"{n_clusters}")
with kpi3:
    st.metric("Avg Annual CLV", f"${avg_clv:,.0f}") if avg_clv > 0 else st.metric("Features Used", f"{len(features_for_cluster)}")
with kpi4:
    st.metric("PCA Variance", f"{sum(pca.explained_variance_ratio_)*100:.1f}%")

st.markdown("---")

# Segment Cards
st.subheader("📊 Segment Profiles")
seg_cols = st.columns(n_clusters)
colors = ['#00CC96', '#FFA15A', '#AB63FA', '#EF553B', '#636EFA', '#19D3F3']
for idx, seg in enumerate(seg_summary.index):
    with seg_cols[idx]:
        name = segment_names[seg]
        count = int(seg_summary.loc[seg, 'count'])
        pct = count / total_customers * 100
        color = colors[idx % 6]
        aov_text = f"AOV: ${seg_summary.loc[seg, 'avg_order_value']:.0f}" if 'avg_order_value' in seg_summary.columns else ""
        rec_text = f"Recency: {seg_summary.loc[seg, 'recency_days']:.0f}d" if 'recency_days' in seg_summary.columns else ""
        st.markdown(f"""
            <div class="segment-card" style="border-left-color: {color};">
                <h3 style="margin:0; color: {color};">{name}</h3>
                <p style="font-size:1.5rem; font-weight:bold; margin:8px 0;">{count:,} <span style="font-size:0.9rem; color:#64748b;">({pct:.1f}%)</span></p>
                <p style="font-size:0.85rem; color:#64748b; margin:0;">{aov_text}</p>
                <p style="font-size:0.85rem; color:#64748b; margin:0;">{rec_text}</p>
            </div>
        """, unsafe_allow_html=True)

st.markdown("---")

# Charts
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("🔬 PCA Cluster Visualization")
    fig = px.scatter(df, x='pca1', y='pca2', color='segment_name',
                     template='plotly_white', opacity=0.7, height=400,
                     hover_data=available_cols[:4] if available_cols else None)
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02))
    st.plotly_chart(fig, use_container_width=True)

with chart_col2:
    st.subheader("📈 Segment Distribution")
    dist_data = df['segment_name'].value_counts().reset_index()
    dist_data.columns = ['segment_name', 'count']
    fig = px.pie(dist_data, values='count', names='segment_name', hole=0.5, template='plotly_white', height=400)
    fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

chart_col3, chart_col4 = st.columns(2)

with chart_col3:
    if rfm_available:
        st.subheader("🎯 RFM Radar Chart")
        rfm_summary = df.groupby('segment_name')[['R', 'F', 'M']].mean().round(1)
        fig = go.Figure()
        for seg in rfm_summary.index:
            fig.add_trace(go.Scatterpolar(
                r=[rfm_summary.loc[seg, 'R'], rfm_summary.loc[seg, 'F'], rfm_summary.loc[seg, 'M']],
                theta=['Recency', 'Frequency', 'Monetary'],
                fill='toself',
                name=seg
            ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 5])), height=400, template='plotly_white',
                          margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation='h', yanchor='bottom', y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("RFM radar chart requires Recency, Frequency (orders), and Monetary (spend) columns.")

with chart_col4:
    if 'avg_order_value' in df.columns:
        st.subheader("💰 Average Order Value by Segment")
        aov_data = df.groupby('segment_name')['avg_order_value'].mean().reset_index().sort_values('avg_order_value', ascending=False)
        fig = px.bar(aov_data, x='segment_name', y='avg_order_value', color='segment_name', template='plotly_white', height=400)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), showlegend=False, xaxis_title='', yaxis_title='AOV ($)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("AOV chart requires average order value column.")

chart_col5, chart_col6 = st.columns(2)

with chart_col5:
    if 'income' in df.columns and 'total_spend' in df.columns:
        st.subheader("💵 Income vs Total Spend")
        fig = px.scatter(df, x='income', y='total_spend', color='segment_name', template='plotly_white',
                         opacity=0.6, height=400, hover_data=['age'] if 'age' in df.columns else None)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation='h', yanchor='bottom', y=1.02))
        st.plotly_chart(fig, use_container_width=True)

with chart_col6:
    if 'gender' in df.columns:
        st.subheader("⚥ Gender Distribution by Segment")
        gender_data = pd.crosstab(df['segment_name'], df['gender'], normalize='index') * 100
        gender_data = gender_data.reset_index().melt(id_vars='segment_name', var_name='gender', value_name='pct')
        fig = px.bar(gender_data, x='segment_name', y='pct', color='gender', barmode='group', template='plotly_white', height=400)
        fig.update_layout(margin=dict(l=20, r=20, t=30, b=20), xaxis_title='', yaxis_title='Percentage (%)', legend=dict(orientation='h', yanchor='bottom', y=1.02))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Gender distribution chart requires a gender column.")

st.markdown("---")

# Auto Insights
st.subheader("💡 Auto-Generated Business Insights")
insights = []

if 'total_spend' in df.columns:
    spend_by_seg = df.groupby('segment_name')['total_spend'].sum().sort_values(ascending=False)
    total = df['total_spend'].sum()
    insights.append(f"**{spend_by_seg.index[0]}** generate the most revenue (${spend_by_seg.iloc[0]:,.0f}, {spend_by_seg.iloc[0]/total*100:.1f}% of total).")

if 'recency_days' in df.columns:
    recency_by_seg = df.groupby('segment_name')['recency_days'].mean().sort_values(ascending=False)
    insights.append(f"**{recency_by_seg.index[0]}** have the highest average recency ({recency_by_seg.iloc[0]:.0f} days) — re-engagement campaigns recommended.")

if 'avg_order_value' in df.columns:
    aov_by_seg = df.groupby('segment_name')['avg_order_value'].mean().sort_values(ascending=False)
    insights.append(f"**{aov_by_seg.index[0]}** have the highest AOV (${aov_by_seg.iloc[0]:.0f}), {aov_by_seg.iloc[0]/aov_by_seg.iloc[-1]:.1f}x higher than **{aov_by_seg.index[-1]}**.")

if 'clv_proxy' in df.columns:
    clv_by_seg = df.groupby('segment_name')['clv_proxy'].mean().sort_values(ascending=False)
    insights.append(f"**{clv_by_seg.index[0]}** have the highest projected annual CLV (${clv_by_seg.iloc[0]:,.0f}).")

if 'preferred_channel' in df.columns:
    for seg in df['segment_name'].unique():
        channel_dist = df[df['segment_name']==seg]['preferred_channel'].value_counts(normalize=True)
        if len(channel_dist) > 0 and channel_dist.iloc[0] > 0.5:
            insights.append(f"**{seg}** prefer **{channel_dist.index[0]}** ({channel_dist.iloc[0]*100:.0f}% of segment).")
            break

for i, ins in enumerate(insights, 1):
    st.markdown(f"{i}. {ins}")

st.markdown("---")

# Data Table
st.subheader("📋 Customer Data with Segments")
display_cols = [c for c in ['customer_id', 'segment_name', 'age', 'gender', 'income', 'region', 'total_spend', 'num_orders', 'avg_order_value', 'recency_days', 'satisfaction_score', 'R', 'F', 'M'] if c in df.columns]
if display_cols:
    st.dataframe(df[display_cols], use_container_width=True, height=400)

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Segmented Data (CSV)", csv, "segmented_customers.csv", "text/csv")

with col_dl2:
    seg_export = df.groupby('segment_name').agg({
        'total_spend': ['sum', 'mean', 'count'] if 'total_spend' in df.columns else 'count',
        'num_orders': 'mean' if 'num_orders' in df.columns else 'count',
        'avg_order_value': 'mean' if 'avg_order_value' in df.columns else 'count',
        'recency_days': 'mean' if 'recency_days' in df.columns else 'count'
    }).round(1).reset_index()
    csv_seg = seg_export.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Segment Summary (CSV)", csv_seg, "segment_summary.csv", "text/csv")

st.markdown("---")
st.caption("Built with Streamlit, Plotly & Scikit-Learn | Internship Project - Customer Segmentation")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Customer Segmentation Dashboard", page_icon="🎯", layout="wide")

# Custom styling
st.markdown("""
    <style>
    .main-header { font-size: 2.4rem; font-weight: bold; color: #0f172a; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.1rem; color: #64748b; margin-bottom: 2rem; }
    .segment-card { background-color: white; padding: 20px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); border-left: 4px solid; margin-bottom: 12px; }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent if "__file__" in dir() else Path.cwd()
SAMPLE_CSV = BASE_DIR / "sample_customer_data.csv"

# Sidebar
st.sidebar.title("📁 Data Import")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])

st.sidebar.markdown("---")
st.sidebar.title("⚙️ Clustering Settings")
n_clusters = st.sidebar.slider("Number of Segments", 2, 6, 4)

@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

# Load data
if uploaded_file is None:
    st.sidebar.info("Using sample data. Upload your own to segment real customers.")
    try:
        df = pd.read_csv(SAMPLE_CSV)
    except Exception:
        st.error("Sample data not found. Please upload a file.")
        st.stop()
else:
    df = load_data(uploaded_file)

# Auto-detect columns
def auto_detect_columns(df):
    col_map = {}
    for col in df.columns:
        cl = col.lower().replace(' ', '_')
        if any(x in cl for x in ['age']):
            col_map[col] = 'age'
        elif any(x in cl for x in ['income', 'salary', 'earnings']):
            col_map[col] = 'income'
        elif any(x in cl for x in ['spend', 'total_spend', 'monetary', 'revenue']):
            col_map[col] = 'total_spend'
        elif any(x in cl for x in ['orders', 'frequency', 'purchases', 'transactions']):
            col_map[col] = 'num_orders'
        elif any(x in cl for x in ['aov', 'avg_order', 'average_order']):
            col_map[col] = 'avg_order_value'
        elif any(x in cl for x in ['recency', 'days_since', 'last_purchase']):
            col_map[col] = 'recency_days'
        elif any(x in cl for x in ['satisfaction', 'rating', 'nps', 'score']):
            col_map[col] = 'satisfaction_score'
        elif any(x in cl for x in ['tenure', 'membership', 'length']):
            col_map[col] = 'tenure_months'
        elif any(x in cl for x in ['visits', 'website', 'engagement']):
            col_map[col] = 'website_visits_per_month'
        elif any(x in cl for x in ['discount', 'promo', 'coupon']):
            col_map[col] = 'discount_sensitivity'
        elif any(x in cl for x in ['gender', 'sex']):
            col_map[col] = 'gender'
        elif any(x in cl for x in ['region', 'area', 'city', 'state']):
            col_map[col] = 'region'
        elif any(x in cl for x in ['education', 'degree']):
            col_map[col] = 'education'
        elif any(x in cl for x in ['marital', 'married', 'status']):
            col_map[col] = 'marital_status'
        elif any(x in cl for x in ['channel', 'platform', 'device']):
            col_map[col] = 'preferred_channel'
        elif any(x in cl for x in ['customer_id', 'id', 'user_id']):
            col_map[col] = 'customer_id'
    return col_map

col_map = auto_detect_columns(df)
df.rename(columns=col_map, inplace=True)

# Required columns for clustering
cluster_cols = ['age', 'income', 'total_spend', 'num_orders', 'avg_order_value', 
                'recency_days', 'satisfaction_score', 'tenure_months', 
                'website_visits_per_month', 'discount_sensitivity']

available_cols = [c for c in cluster_cols if c in df.columns]
missing_cols = [c for c in cluster_cols if c not in df.columns]

if missing_cols:
    st.warning(f"Missing columns for full RFM clustering: {missing_cols}. Using available features: {available_cols}")

if len(available_cols) < 3:
    st.error("Need at least 3 numeric columns for clustering. Please upload data with customer behavior metrics.")
    st.stop()

# Feature engineering
df['clv_proxy'] = df['total_spend'] / df['tenure_months'].clip(lower=1) * 12 if 'total_spend' in df.columns and 'tenure_months' in df.columns else 0
df['engagement_score'] = (df['website_visits_per_month'] / df['num_orders'].clip(lower=1)).clip(0, 20) if 'website_visits_per_month' in df.columns and 'num_orders' in df.columns else 0

# RFM scoring if all available
if 'recency_days' in df.columns and 'num_orders' in df.columns and 'total_spend' in df.columns:
    df['R'] = pd.qcut(df['recency_days'].rank(method='first'), 5, labels=[5,4,3,2,1]).astype(int)
    df['F'] = pd.qcut(df['num_orders'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    df['M'] = pd.qcut(df['total_spend'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    rfm_available = True
else:
    rfm_available = False

# Prepare clustering features
features_for_cluster = available_cols.copy()
if 'clv_proxy' in df.columns:
    features_for_cluster.append('clv_proxy')
if 'engagement_score' in df.columns:
    features_for_cluster.append('engagement_score')

X = df[features_for_cluster].fillna(df[features_for_cluster].median())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KMeans clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df['segment'] = kmeans.fit_predict(X_scaled)

# PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
df['pca1'] = pca_result[:, 0]
df['pca2'] = pca_result[:, 1]

# Segment naming heuristic
seg_summary = df.groupby('segment').agg({
    'total_spend': 'mean' if 'total_spend' in df.columns else 'count',
    'num_orders': 'mean' if 'num_orders' in df.columns else 'count',
    'recency_days': 'mean' if 'recency_days' in df.columns else 'count',
    'avg_order_value': 'mean' if 'avg_order_value' in df.columns else 'count',
    'customer_id': 'count' if 'customer_id' in df.columns else 'count'
}).round(1)

if 'customer_id' in df.columns:
    seg_summary.rename(columns={'customer_id': 'count'}, inplace=True)
else:
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

# Main Dashboard
st.markdown('<div class="main-header">🎯 Customer Segmentation Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">K-Means Clustering on Behavioral & Demographic Data</div>', unsafe_allow_html=True)

# KPI Row
total_customers = len(df)
if 'total_spend' in df.columns:
    total_revenue = df['total_spend'].sum()
    avg_clv = df['clv_proxy'].mean() if 'clv_proxy' in df.columns else 0
else:
    total_revenue = 0
    avg_clv = 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric("Total Customers", f"{total_customers:,}")
with kpi2:
    if total_revenue > 0:
        st.metric("Total Revenue", f"${total_revenue:,.0f}")
    else:
        st.metric("Segments", f"{n_clusters}")
with kpi3:
    if avg_clv > 0:
        st.metric("Avg Annual CLV", f"${avg_clv:,.0f}")
    else:
        st.metric("Features Used", f"{len(features_for_cluster)}")
with kpi4:
    st.metric("PCA Variance", f"{sum(pca.explained_variance_ratio_)*100:.1f}%")

st.markdown("---")

# Segment Cards
st.subheader("📊 Segment Profiles")
seg_cols = st.columns(n_clusters)
for idx, seg in enumerate(seg_summary.index):
    with seg_cols[idx]:
        name = segment_names[seg]
        count = int(seg_summary.loc[seg, 'count'])
        pct = count / total_customers * 100
        color = ['#00CC96', '#FFA15A', '#AB63FA', '#EF553B', '#636EFA', '#19D3F3'][idx % 6]
        st.markdown(f"""
            <div class="segment-card" style="border-left-color: {color};">
                <h3 style="margin:0; color: {color};">{name}</h3>
                <p style="font-size:1.5rem; font-weight:bold; margin:8px 0;">{count:,} <span style="font-size:0.9rem; color:#64748b;">({pct:.1f}%)</span></p>
                {'<p style="font-size:0.85rem; color:#64748b; margin:0;">AOV: $' + f"{seg_summary.loc[seg, 'avg_order_value']:.0f}" + '</p>' if 'avg_order_value' in seg_summary.columns else ''}
                {'<p style="font-size:0.85rem; color:#64748b; margin:0;">Recency: ' + f"{seg_summary.loc[seg, 'recency_days']:.0f}d" + '</p>' if 'recency_days' in seg_summary.columns else ''}
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

# Second row
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

# Third row
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

# Downloads
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

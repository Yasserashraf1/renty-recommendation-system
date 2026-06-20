import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add current dir to path to import local utilities
sys.path.append(os.path.dirname(__file__))

from recommender_utils import (
    load_and_preprocess_data,
    engineer_features,
    extract_text_features,
    load_model_artifacts,
    get_recommendations_for_input_user
)

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="Renty Recommender System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling — Renty Brand Colors: Teal #2DD4BF / Navy #0F1B4C / White
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* ── Global Font ── */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* ── App Background ── */
    .stApp {
        background: linear-gradient(160deg, #0a1628 0%, #0f2044 60%, #0d2a3a 100%);
    }

    /* ── Remove anchor link icons from headings ── */
    h1 a, h2 a, h3 a, h4 a { display: none !important; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0c1e3a 0%, #0a2a2a 100%);
        border-right: 1px solid rgba(45, 212, 191, 0.2);
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(13, 31, 60, 0.8);
        border-radius: 10px;
        padding: 4px;
        border: 1px solid rgba(45, 212, 191, 0.2);
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 500;
        border-radius: 8px;
        padding: 8px 18px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #2DD4BF, #1a9e8f) !important;
        color: #ffffff !important;
        font-weight: 700;
    }

    /* ── Main Title ── */
    .main-title {
        background: linear-gradient(135deg, #2DD4BF 0%, #5eead4 60%, #ffffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.8rem;
        font-weight: 700;
        margin-bottom: 4px;
        line-height: 1.2;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #7dd3c8;
        margin-bottom: 28px;
        letter-spacing: 0.03em;
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: rgba(13, 31, 60, 0.85);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(45, 212, 191, 0.25);
        border-radius: 14px;
        padding: 24px 20px;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        margin-bottom: 8px;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(45, 212, 191, 0.5);
        box-shadow: 0 8px 32px rgba(45, 212, 191, 0.15);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #7dd3c8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 0 0 10px 0;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.1;
    }

    /* ── Recommendation Cards ── */
    .rec-card {
        background: rgba(10, 22, 42, 0.9);
        border: 1px solid rgba(45, 212, 191, 0.18);
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 14px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
        transition: all 0.25s ease;
    }
    .rec-card:hover {
        transform: scale(1.01);
        border-color: rgba(45, 212, 191, 0.5);
        box-shadow: 0 10px 30px rgba(45, 212, 191, 0.12);
    }

    /* ── Dividers ── */
    hr { border-color: rgba(45, 212, 191, 0.15) !important; }

    /* ── Tags ── */
    .tag {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 5px;
        margin-bottom: 4px;
        letter-spacing: 0.02em;
    }
    .tag-platinum { background: rgba(224,247,245,0.15); color: #a7f3d0; border: 1px solid #a7f3d0; }
    .tag-gold     { background: rgba(254,240,138,0.15); color: #fde68a; border: 1px solid #fde68a; }
    .tag-silver   { background: rgba(203,213,225,0.12); color: #cbd5e1; border: 1px solid #94a3b8; }
    .tag-bronze   { background: rgba(254,215,170,0.15); color: #fdba74; border: 1px solid #fdba74; }
    .tag-cold     { background: rgba(252,165,165,0.12); color: #fca5a5; border: 1px solid #fca5a5; }
    .tag-pers     { background: rgba(45,212,191,0.15);  color: #2DD4BF; border: 1px solid #2DD4BF; }

    /* ── Profile section text ── */
    .profile-label { color: #7dd3c8; font-size: 0.85rem; font-weight: 500; }

    /* ── Success / Info / Warning box overrides ── */
    div[data-testid="stNotification"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------
# 1. DATA AND MODEL CACHING
# ----------------------------------------------------

@st.cache_data(show_spinner="Loading and preprocessing Excel dataset...")
def get_cached_df():
    try:
        # Load local cleaned data
        filepath = os.path.join(os.path.dirname(__file__), "Cleaned", "GradProject_final_1.xlsx")
        df_raw = load_and_preprocess_data(filepath)
        df_eng = engineer_features(df_raw)
        df_final, _ = extract_text_features(df_eng)
        return df_final
    except Exception as e:
        st.error(f"Error loading Excel data: {e}")
        return None

@st.cache_resource(show_spinner="Loading trained LightFM model artifacts...")
def get_cached_model():
    try:
        filepath = os.path.join(os.path.dirname(__file__), "renty_lightfm_model_artifacts.pkl")
        return load_model_artifacts(filepath)
    except Exception as e:
        st.error(f"Error loading recommendation model `.pkl`: {e}")
        return None, None, None, None

# Load Resources
df = get_cached_df()
model, dataset, user_features, item_features = get_cached_model()

# Header Layout
st.markdown('<div class="main-title">Renty Recommender System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Graduation Project Dashboard &nbsp;•&nbsp; LightFM Hybrid Recommendation Engine</div>', unsafe_allow_html=True)

if df is None or model is None:
    st.error("Missing resources! Please check your file paths and installation.")
    st.stop()

# ----------------------------------------------------
# 2. SIDEBAR CONFIGURATION
# ----------------------------------------------------
# Logo at top of sidebar
st.sidebar.image(
    os.path.join(os.path.dirname(__file__), "Logo1.png"),
    width=260
)
st.sidebar.markdown("---")
st.sidebar.header("UI Controls")

# Sampling users from each customer segment for easy selection
segments = ['Bronze', 'Silver', 'Gold', 'Platinum']
sample_users = {}
for seg in segments:
    subset = df[df['CustomerSegment'] == seg]
    if not subset.empty:
        sample_users[seg] = int(subset['CustomerKey'].sample(1, random_state=42).iloc[0])
    else:
        sample_users[seg] = "No Data"

st.sidebar.subheader("Sample Customer Key Presets")
st.sidebar.write("Use these existing Customer IDs to test different segmentation profiles:")
for seg, val in sample_users.items():
    st.sidebar.markdown(f"**{seg}:** `{val}`")

st.sidebar.write("---")

# User Input ID
user_input = st.sidebar.text_input("Enter Customer ID Key:", value=str(sample_users['Gold']))

# Configs
n_recs = st.sidebar.slider("Number of Recommendations:", min_value=1, max_value=20, value=10)
filter_purchased = st.sidebar.checkbox("Filter out already purchased items", value=True)

# ----------------------------------------------------
# 3. TABS STRUCTURE
# ----------------------------------------------------
tab_dashboard, tab_recommender, tab_performance = st.tabs([
    "Customer Insights Dashboard", 
    "Recommendation Engine", 
    "Technical Performance & Evaluation"
])

# ----------------------------------------------------
# TAB 1: CUSTOMER INSIGHTS DASHBOARD
# ----------------------------------------------------
with tab_dashboard:
    st.subheader("Data Overview & EDA Metrics")
    
    # 4 Key Metric Columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Unique Customers</p>
            <p class="metric-value" style="color:#2DD4BF;">{df['CustomerKey'].nunique():,}</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Unique Products</p>
            <p class="metric-value" style="color:#5eead4;">{df['ProductKey'].nunique():,}</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Total Orders</p>
            <p class="metric-value" style="color:#ffffff;">{df['OrderNumber'].nunique():,}</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-label">Dataset Rows</p>
            <p class="metric-value" style="color:#a5f3fc;">{df.shape[0]:,}</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("---")

    # Visualizations layout
    col_vis1, col_vis2 = st.columns(2)
    
    with col_vis1:
        st.subheader("Customer Segments (RFM Tiers)")
        # Calculate segments count
        seg_counts = df.groupby('CustomerKey')['CustomerSegment'].first().value_counts().reset_index()
        seg_counts.columns = ['Segment', 'Count']
        
        # Plotly/Streamlit native bar chart
        st.bar_chart(data=seg_counts, x='Segment', y='Count', color='#2DD4BF')
        st.caption("Distribution of Bronze, Silver, Gold, Platinum based on customer transaction frequency & purchase quantity.")
        
    with col_vis2:
        st.subheader("Top 10 Selling Products by Model")
        top_items = df.groupby('ModelName')['OrderQuantity'].sum().reset_index()
        top_items = top_items.nlargest(10, 'OrderQuantity').sort_values('OrderQuantity', ascending=True)
        
        st.bar_chart(data=top_items, x='ModelName', y='OrderQuantity', color='#5eead4')
        st.caption("Model names with the highest order volume.")

    st.write("---")

    # Demographic correlations
    st.subheader("Customer Demographics Analysis")
    col_demo1, col_demo2 = st.columns(2)
    with col_demo1:
        # Annual Income Distribution by Occupation
        occ_income = df.groupby('CustomerKey')[['Occupation', 'AnnualIncome']].first().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(data=occ_income, x='Occupation', y='AnnualIncome', palette=['#2DD4BF','#5eead4','#1a9e8f','#0d7a6e','#a5f3fc'], hue='Occupation', legend=False, ax=ax)
        plt.title("Annual Income Range by Occupation Type", color='white')
        plt.xticks(rotation=15, color='white')
        plt.yticks(color='white')
        ax.set_facecolor('#0a1628')
        fig.patch.set_facecolor('#0a1628')
        ax.spines['bottom'].set_color('#2DD4BF')
        ax.spines['left'].set_color('#2DD4BF')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        st.pyplot(fig)
    with col_demo2:
        # Order quantity distribution by Income Bracket
        income_bracket_qty = df.groupby('CustomerKey')[['IncomeBracket', 'TotalQuantity']].first().reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=income_bracket_qty, x='IncomeBracket', y='TotalQuantity', hue='IncomeBracket', palette=['#2DD4BF','#5eead4','#1a9e8f','#0d7a6e','#a5f3fc','#67e8f9'], legend=False, errorbar=None, ax=ax)
        plt.title("Average Purchased Items per Income Bracket", color='white')
        plt.xticks(color='white')
        plt.yticks(color='white')
        ax.set_facecolor('#0a1628')
        fig.patch.set_facecolor('#0a1628')
        ax.spines['bottom'].set_color('#2DD4BF')
        ax.spines['left'].set_color('#2DD4BF')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        st.pyplot(fig)

# ----------------------------------------------------
# TAB 2: RECOMMENDATION ENGINE
# ----------------------------------------------------
with tab_recommender:
    st.subheader("Real-time Recommendation Inference")

    if user_input:
        try:
            target_user = int(user_input)
        except ValueError:
            st.error("Please enter a valid numeric Customer ID.")
            st.stop()
            
        # Get recommendations
        recs_df, rec_type = get_recommendations_for_input_user(
            target_user, model, dataset, user_features, item_features, df,
            n_recommendations=n_recs, filter_already_purchased=filter_purchased
        )
        
        # Display Customer Profile Card
        user_rows = df[df['CustomerKey'] == target_user]
        
        if not user_rows.empty:
            # Existing Customer Profile
            profile = user_rows.iloc[0]
            
            st.markdown("### Customer Profile Summary")
            
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                st.markdown(f"**Customer ID:** `{target_user}`")
                st.markdown(f"**Gender / Marital Status:** {profile['Gender']} / {profile['MaritalStatus']}")
                st.markdown(f"**Home Owner:** {profile['HomeOwner']}")
            with col_p2:
                st.markdown(f"**Annual Income:** `${profile['AnnualIncome']:,}` ({profile['IncomeBracket']})")
                st.markdown(f"**Education / Occupation:** {profile['EducationLevel']} / {profile['Occupation']}")
                st.markdown(f"**Total Children:** {profile['TotalChildren']}")
            with col_p3:
                # Segment Tag
                seg_label = str(profile['CustomerSegment']).lower()
                st.markdown(f"**Tier Segment:** <span class='tag tag-{seg_label}'>{profile['CustomerSegment']}</span>", unsafe_allow_html=True)
                st.markdown(f"**LTV Score:** `{profile['CustomerValueScore']:.2f}`")
                st.markdown(f"**Total Past Orders:** `{profile['TotalOrders']}` ({profile['TotalQuantity']} items total)")

            st.write("---")

            # Display Purchase History
            st.markdown("#### Purchase History (Items already bought)")
            purchase_hist = user_rows.groupby(['ProductKey', 'ModelName', 'ProductDescription']).agg({
                'OrderQuantity': 'sum',
                'OrderDate': 'max'
            }).reset_index().sort_values('OrderDate', ascending=False)
            
            # Show history table
            st.dataframe(
                purchase_hist.rename(columns={
                    'ProductKey': 'Product ID',
                    'ModelName': 'Model Name',
                    'ProductDescription': 'Description',
                    'OrderQuantity': 'Qty Purchased',
                    'OrderDate': 'Last Ordered'
                }),
                width='stretch',
                hide_index=True
            )
            
        else:
            # Cold Start user details
            st.markdown("### Customer Profile Summary")
            st.warning(f"Customer ID Key `{target_user}` was not found in the historical training dataset.")
            st.markdown("<span class='tag tag-cold'>New User Profile</span> (Cold Start Scenario)", unsafe_allow_html=True)
            
        st.write("---")
        
        # Display recommendations
        st.markdown("### Generated Recommendations")
        
        if rec_type == "Personalized":
            st.markdown(f"Personalized Recommendations generated using **LightFM Hybrid model** user & item embeddings. <span class='tag tag-pers'>Hybrid Prediction</span>", unsafe_allow_html=True)
        elif rec_type == "ColdStart":
            st.markdown(f"Popular Recommendations generated using **Popularity Fallback score** due to missing user history. <span class='tag tag-cold'>Cold Start Popularity Fallback</span>", unsafe_allow_html=True)
        elif rec_type == "FallbackPopular":
            st.markdown(f"Personalized recommendations exhausted. Displaying **Popular items not yet purchased**. <span class='tag tag-cold'>Fallback Popularity List</span>", unsafe_allow_html=True)
            
        st.write("")
        
        if recs_df is not None and not recs_df.empty:
            # Display grid of recommendation cards
            # We will show 2 cards per row
            for i in range(0, len(recs_df), 2):
                row_cols = st.columns(2)
                for j in range(2):
                    idx = i + j
                    if idx < len(recs_df):
                        row = recs_df.iloc[idx]
                        score_label = "Popularity" if row['IsColdStart'] else "Match Score"
                        score_val = row['NormalizedScore']
                        
                        with row_cols[j]:
                            st.markdown(f"""
                            <div class="rec-card">
                                <span class="tag tag-{'cold' if row['IsColdStart'] else 'pers'}" style="float: right;">{score_label}: {score_val:.4f}</span>
                                <h4 style="margin: 0 0 10px 0; color: #2DD4BF;">Rank {row['Rank']} &nbsp;•&nbsp; {row['ModelName']}</h4>
                                <p style="font-size: 0.95rem; line-height: 1.5; color: #d1faf5;">{row['ProductDescription']}</p>
                                <span style="font-size: 0.8rem; color:#5eead4;">Product ID: {row['ProductKey']}</span>
                            </div>
                            """, unsafe_allow_html=True)
        else:
            st.info("No recommendations found for this configuration.")

# ----------------------------------------------------
# TAB 3: TECHNICAL PERFORMANCE
# ----------------------------------------------------
with tab_performance:
    st.subheader("Model Evaluation Summary")
    st.write("Below are the evaluation metrics comparing the baseline collaborative filtering model against the improved Hybrid LightFM model (which includes user metadata, text description embeddings, and hyperparameter regularization):")
    
    col_t1, col_t2 = st.columns(2)
    
    with col_t1:
        # Comparison Table
        metric_data = {
            'Metric': ['Test AUC', 'Precision@10', 'Overfitting Gap (Train - Test)'],
            'Baseline Collaborative Filtering': ['0.7493', '0.0944', '0.2501 (High Overfitting)'],
            'Improved Hybrid LightFM Model': ['0.8417', '0.1216', '0.0252 (Generalizes Well)']
        }
        st.table(pd.DataFrame(metric_data))
        st.success("**Hybrid Model Enhancements:** incorporation of demographics, RFM segmentation, description TF-IDF, and user/item Alpha regularization reduced overfitting by 90% and improved test AUC by 12.3%.")
        
    with col_t2:
        # Hyperparameters Info
        st.markdown("""
        **Optimal Model Hyperparameters:**
        * **Loss Function:** `WARP` (Weighted Approximate-Rank Pairwise)
        * **Latent Components (Dimensions):** `50`
        * **Learning Rate:** `0.05`
        * **Item Regularization (Alpha):** `0.0001`
        * **User Regularization (Alpha):** `0.0001`
        * **Training Epochs:** `60`
        """)
        
        st.info("**WARP Loss** optimizes the top of the recommendation list by gradient updates only when a random negative item is ranked higher than a positive item, making it ideal for e-commerce ranking tasks.")

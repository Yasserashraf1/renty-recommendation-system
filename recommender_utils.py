import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_preprocess_data(filepath):
    """Load and perform initial preprocessing on the dataset"""
    df = pd.read_excel(filepath)
    # Convert date columns
    df['OrderDate'] = pd.to_datetime(df['OrderDate'])
    df['StockDate'] = pd.to_datetime(df['StockDate'])
    # Remove duplicates
    df = df.drop_duplicates()
    return df

def engineer_features(df):
    """Create advanced features with scaling and text encoding"""
    df_features = df.copy()

    # ===== TEMPORAL FEATURES =====
    df_features['OrderYear'] = df_features['OrderDate'].dt.year
    df_features['OrderMonth'] = df_features['OrderDate'].dt.month
    df_features['OrderQuarter'] = df_features['OrderDate'].dt.quarter
    df_features['DayOfWeek'] = df_features['OrderDate'].dt.dayofweek
    df_features['IsWeekend'] = df_features['DayOfWeek'].isin([5, 6]).astype(int)

    # Seasonality
    df_features['Season'] = df_features['OrderMonth'].map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })

    # ===== IMPROVED CATEGORICAL FEATURES =====
    # Fine-grained income brackets
    df_features['IncomeBracket'] = pd.cut(df_features['AnnualIncome'],
                                           bins=[0, 25000, 40000, 60000, 80000, 100000, 200000],
                                           labels=['VeryLow', 'Low', 'Medium', 'High', 'VeryHigh', 'Premium'])

    # Children category
    df_features['ChildrenCategory'] = pd.cut(df_features['TotalChildren'],
                                              bins=[-1, 0, 1, 2, 10],
                                              labels=['NoChildren', 'OneChild', 'TwoChildren', 'ManyChildren'])

    # ===== USER ENGAGEMENT METRICS =====
    user_stats = df_features.groupby('CustomerKey').agg({
        'OrderNumber': 'nunique',
        'ProductKey': 'nunique',
        'OrderQuantity': ['sum', 'mean', 'std'],
        'OrderDate': lambda x: (x.max() - x.min()).days,
        'AnnualIncome': 'first'
    }).reset_index()

    user_stats.columns = ['CustomerKey', 'TotalOrders', 'UniqueProducts',
                          'TotalQuantity', 'AvgOrderQuantity', 'StdOrderQuantity',
                          'CustomerLifetimeDays', 'AnnualIncome']

    # Replace NaN std with 0 for single-order customers
    user_stats['StdOrderQuantity'] = user_stats['StdOrderQuantity'].fillna(0)

    # Customer Value Score (RFM-inspired)
    user_stats['CustomerValueScore'] = (
        user_stats['TotalOrders'] * 0.3 +
        user_stats['UniqueProducts'] * 0.3 +
        (user_stats['TotalQuantity'] / user_stats['TotalQuantity'].max()) * 100 * 0.4
    )

    # Customer segments
    user_stats['CustomerSegment'] = pd.qcut(user_stats['CustomerValueScore'],
                                             q=4, labels=['Bronze', 'Silver', 'Gold', 'Platinum'],
                                             duplicates='drop')

    # Merge user stats back
    df_features = df_features.merge(user_stats[['CustomerKey', 'TotalOrders', 'UniqueProducts',
                                                 'TotalQuantity', 'AvgOrderQuantity', 'StdOrderQuantity',
                                                 'CustomerLifetimeDays', 'CustomerValueScore', 'CustomerSegment']],
                                    on='CustomerKey', how='left')

    # ===== ITEM FEATURES =====
    item_stats = df_features.groupby('ProductKey').agg({
        'OrderQuantity': ['sum', 'mean', 'count'],
        'CustomerKey': 'nunique',
        'ModelName': 'first',
        'ProductDescription': 'first'
    }).reset_index()

    item_stats.columns = ['ProductKey', 'TotalItemsSold', 'AvgItemOrderQty',
                          'ItemPopularity', 'UniqueCustomers', 'ModelName', 'ProductDescription']

    # Item category from ModelName
    item_stats['ItemCategory'] = item_stats['ModelName'].apply(lambda x: x.split('-')[0] if '-' in str(x) else 'Other')

    # Popularity percentile
    item_stats['PopularityPercentile'] = pd.qcut(item_stats['ItemPopularity'],
                                                   q=5, labels=['Niche', 'LowPop', 'MedPop', 'HighPop', 'Viral'],
                                                   duplicates='drop')

    df_features = df_features.merge(item_stats[['ProductKey', 'TotalItemsSold', 'AvgItemOrderQty',
                                                 'ItemPopularity', 'UniqueCustomers', 'ItemCategory',
                                                 'PopularityPercentile']],
                                    on='ProductKey', how='left')

    # ===== SCALED NUMERICAL FEATURES =====
    scaler = MinMaxScaler()
    numerical_cols = ['AnnualIncome', 'TotalChildren', 'TotalOrders', 'UniqueProducts',
                      'AvgOrderQuantity', 'CustomerLifetimeDays', 'CustomerValueScore']

    for col in numerical_cols:
        if col in df_features.columns:
            df_features[f'{col}_Scaled'] = scaler.fit_transform(df_features[[col]])

    return df_features

def extract_text_features(df, max_features=50):
    """Extract TF-IDF features from product descriptions"""
    # Get unique products with descriptions
    unique_products = df.groupby('ProductKey')['ProductDescription'].first().reset_index()

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english',
                            ngram_range=(1, 2), min_df=2)
    tfidf_matrix = tfidf.fit_transform(unique_products['ProductDescription'].fillna(''))

    # Create text feature dataframe
    text_features_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f'text_{word}' for word in tfidf.get_feature_names_out()]
    )
    text_features_df['ProductKey'] = unique_products['ProductKey'].values

    # Merge back to main dataframe
    df = df.merge(text_features_df, on='ProductKey', how='left')

    return df, list(text_features_df.columns[:-1])

def load_model_artifacts(filepath='renty_lightfm_model_artifacts.pkl'):
    """Loads the LightFM model and associated artifacts from a pickle file."""
    with open(filepath, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts['model'], artifacts['dataset'], artifacts['user_features'], artifacts['item_features']

def get_recommendations(model, user_id, dataset, user_features, item_features,
                       df, n_recommendations=10, filter_already_purchased=True):
    """Generate top-N recommendations for a specific user"""
    # Get mappings
    user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()

    # Check if user exists
    if user_id not in user_id_map:
        return None

    internal_user_id = user_id_map[user_id]
    n_items = len(item_id_map)

    # Predict scores for all items
    scores = model.predict(
        internal_user_id,
        np.arange(n_items),
        user_features=user_features,
        item_features=item_features
    )

    # Filter already purchased items
    if filter_already_purchased:
        purchased_items = df[df['CustomerKey'] == user_id]['ProductKey'].unique()
        purchased_internal_ids = [item_id_map[item] for item in purchased_items if item in item_id_map]
        scores[purchased_internal_ids] = -np.inf

    # Get top N recommendations
    top_items_internal = np.argsort(-scores)[:n_recommendations]

    # Map back to external IDs
    reverse_item_map = {v: k for k, v in item_id_map.items()}
    top_items = [reverse_item_map[i] for i in top_items_internal]
    top_scores = scores[top_items_internal]

    # Create recommendations dataframe
    recommendations = []
    for item_id, score in zip(top_items, top_scores):
        item_info = df[df['ProductKey'] == item_id].iloc[0]
        recommendations.append({
            'Rank': len(recommendations) + 1,
            'ProductKey': item_id,
            'ModelName': item_info['ModelName'],
            'ProductDescription': item_info['ProductDescription'][:150] + ('...' if len(item_info['ProductDescription']) > 150 else ''),
            'Score': score
        })

    return pd.DataFrame(recommendations)

def normalize_recommendation_scores(recommendations_df):
    """Normalizes recommendation scores to [0, 1] range while preserving ranking."""
    if recommendations_df is None or recommendations_df.empty:
        return recommendations_df

    if 'Score' not in recommendations_df.columns:
        return recommendations_df

    # Convert Score column to numeric
    scores = recommendations_df['Score'].astype(float)

    min_score = scores.min()
    max_score = scores.max()

    # Handle edge case where all scores are identical
    if max_score == min_score:
        recommendations_df['NormalizedScore'] = 1.0
    else:
        normalized = (scores - min_score) / (max_score - min_score)
        recommendations_df['NormalizedScore'] = normalized.round(4)

    recommendations_df['OriginalScore'] = scores.round(4)
    recommendations_df = recommendations_df.drop('Score', axis=1)

    # Reorder columns
    cols = ['Rank', 'ProductKey', 'ModelName', 'ProductDescription', 'NormalizedScore', 'OriginalScore']
    cols_exist = [c for c in cols if c in recommendations_df.columns]
    recommendations_df = recommendations_df[cols_exist]

    return recommendations_df

def get_recommendations_for_input_user(user_id_input, model, dataset, user_features, item_features, df, n_recommendations=10, filter_already_purchased=True):
    """
    Takes a user ID input and provides recommendations using the loaded LightFM model.
    Handles cold start problem by recommending popular items for new users.
    """
    try:
        user_id_input = int(user_id_input)
    except ValueError:
        return None, "Invalid ID"

    # Get user mapping to check if user exists
    user_id_map, _, _, _ = dataset.mapping()

    # Check if user exists in training data
    if user_id_input not in user_id_map:
        # COLD START STRATEGY: Recommending most popular items
        item_popularity = df.groupby('ProductKey').agg({
            'OrderQuantity': 'sum',  # Total quantity sold
            'CustomerKey': 'nunique',  # Number of unique customers
            'OrderNumber': 'count'  # Number of orders
        }).reset_index()

        item_popularity.columns = ['ProductKey', 'TotalQuantitySold', 'UniqueCustomers', 'OrderCount']

        # Calculate composite popularity score (Min-Max normalization)
        item_popularity['NormQuantity'] = (item_popularity['TotalQuantitySold'] - item_popularity['TotalQuantitySold'].min()) / \
                                           (item_popularity['TotalQuantitySold'].max() - item_popularity['TotalQuantitySold'].min() + 1e-9)
        item_popularity['NormCustomers'] = (item_popularity['UniqueCustomers'] - item_popularity['UniqueCustomers'].min()) / \
                                            (item_popularity['UniqueCustomers'].max() - item_popularity['UniqueCustomers'].min() + 1e-9)
        item_popularity['NormOrders'] = (item_popularity['OrderCount'] - item_popularity['OrderCount'].min()) / \
                                         (item_popularity['OrderCount'].max() - item_popularity['OrderCount'].min() + 1e-9)

        # Weighted popularity score (40% quantity, 30% customers, 30% orders)
        item_popularity['PopularityScore'] = (0.4 * item_popularity['NormQuantity'] +
                                              0.3 * item_popularity['NormCustomers'] +
                                              0.3 * item_popularity['NormOrders'])

        # Sort by popularity and get top N
        top_popular_items = item_popularity.nlargest(n_recommendations, 'PopularityScore')

        # Create recommendations dataframe with product details
        recommendations = []
        for idx, row in top_popular_items.iterrows():
            product_key = row['ProductKey']
            item_info = df[df['ProductKey'] == product_key].iloc[0]

            recommendations.append({
                'Rank': len(recommendations) + 1,
                'ProductKey': product_key,
                'ModelName': item_info['ModelName'],
                'ProductDescription': item_info['ProductDescription'][:150] + ('...' if len(item_info['ProductDescription']) > 150 else ''),
                'NormalizedScore': round(float(row['PopularityScore']), 4),
                'IsColdStart': True
            })

        return pd.DataFrame(recommendations), "ColdStart"

    # User exists in training data - use personalized recommendations
    recs = get_recommendations(
        model, user_id_input, dataset, user_features, item_features,
        df, n_recommendations=n_recommendations, filter_already_purchased=filter_already_purchased
    )

    if recs is not None and not recs.empty:
        normalized_recs = normalize_recommendation_scores(recs)
        normalized_recs['IsColdStart'] = False
        return normalized_recs, "Personalized"
    
    # Fallback to popular items not yet purchased
    purchased_items = set(df[df['CustomerKey'] == user_id_input]['ProductKey'].unique())

    item_popularity = df.groupby('ProductKey').agg({
        'OrderQuantity': 'sum',
        'CustomerKey': 'nunique',
        'OrderNumber': 'count'
    }).reset_index()

    item_popularity.columns = ['ProductKey', 'TotalQuantitySold', 'UniqueCustomers', 'OrderCount']

    # Filter out purchased items
    item_popularity = item_popularity[~item_popularity['ProductKey'].isin(purchased_items)]

    item_popularity['NormQuantity'] = (item_popularity['TotalQuantitySold'] - item_popularity['TotalQuantitySold'].min()) / \
                                       (item_popularity['TotalQuantitySold'].max() - item_popularity['TotalQuantitySold'].min() + 1e-9)
    item_popularity['NormCustomers'] = (item_popularity['UniqueCustomers'] - item_popularity['UniqueCustomers'].min()) / \
                                        (item_popularity['UniqueCustomers'].max() - item_popularity['UniqueCustomers'].min() + 1e-9)
    item_popularity['NormOrders'] = (item_popularity['OrderCount'] - item_popularity['OrderCount'].min()) / \
                                     (item_popularity['OrderCount'].max() - item_popularity['OrderCount'].min() + 1e-9)

    item_popularity['PopularityScore'] = (0.4 * item_popularity['NormQuantity'] +
                                          0.3 * item_popularity['NormCustomers'] +
                                          0.3 * item_popularity['NormOrders'])

    top_popular_items = item_popularity.nlargest(n_recommendations, 'PopularityScore')

    fallback_recommendations = []
    for idx, row in top_popular_items.iterrows():
        product_key = row['ProductKey']
        item_info = df[df['ProductKey'] == product_key].iloc[0]

        fallback_recommendations.append({
            'Rank': len(fallback_recommendations) + 1,
            'ProductKey': product_key,
            'ModelName': item_info['ModelName'],
            'ProductDescription': item_info['ProductDescription'][:150] + ('...' if len(item_info['ProductDescription']) > 150 else ''),
            'NormalizedScore': round(float(row['PopularityScore']), 4),
            'IsColdStart': True
        })

    return pd.DataFrame(fallback_recommendations), "FallbackPopular"

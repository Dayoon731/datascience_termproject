"""
Data Processing and Visualization Module for Movie Dataset Analysis
This module contains functions for data preprocessing, feature engineering, and visualization generation.
It handles both classification and regression tasks for movie success prediction.
"""

import pandas as pd
import numpy as np
import ast
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer, OneHotEncoder

def load_and_preprocess_data(file_path):
    """
    Load and preprocess movie metadata
    - Converts numeric columns to appropriate types
    - Handles missing values
    - Filters out invalid budget entries
    - Processes release dates and language information
    """
    df = pd.read_csv('/Users/user/OneDrive/바탕 화면/데과 텀프/movies_metadata.csv', low_memory=False)
    
    # Convert numeric columns
    df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
    df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
    df = df[df['budget'] > 0].copy()
    
    # Handle missing values
    df['runtime'] = df['runtime'].fillna(df['runtime'].median())
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df = df.dropna(subset=['release_date'])
    
    # Handle original_language
    most_common_lang = df['original_language'].mode()[0]
    df['original_language'] = df['original_language'].fillna(most_common_lang)
    
    return df

def process_genres(df):
    """
    Process and encode movie genres using multi-label binarization
    - Parses genre strings into lists
    - Creates binary features for each genre
    - Returns encoded genres and the encoder for future use
    """
    def parse_genres(genres_str):
        try:
            genres_list = ast.literal_eval(genres_str)
            return [genre['name'] for genre in genres_list]
        except:
            return []
    
    df['genres_list'] = df['genres'].apply(parse_genres)
    mlb = MultiLabelBinarizer()
    genres_encoded = pd.DataFrame(mlb.fit_transform(df['genres_list']),
                                columns=mlb.classes_, index=df.index)
    return df, genres_encoded, mlb

def process_languages(df):
    """
    Encode movie languages using one-hot encoding
    - Creates binary features for each language
    - Handles unknown languages gracefully
    """
    lang_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    lang_encoded = pd.DataFrame(lang_encoder.fit_transform(df[['original_language']]),
                              columns=lang_encoder.get_feature_names_out(['original_language']),
                              index=df.index)
    return lang_encoded, lang_encoder

def create_success_label(df):
    """
    Create binary success label based on revenue/budget ratio threshold
    - A movie is considered successful if it makes at least 2x its budget
    """
    return (df['revenue'] / df['budget'] >= 2).astype(int)

def generate_data_visualizations(df, output_dir='./visualizations'):
    """
    Generate exploratory data analysis visualizations
    - Creates correlation heatmap for numerical features
    - Shows distribution of movie success
    - Displays top 10 language distribution
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Correlation analysis of numerical features
    numeric_columns = ['budget', 'revenue', 'popularity', 'runtime', 'vote_average', 'vote_count']
    correlation = df[numeric_columns].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(f'{output_dir}/correlation_heatmap.png')
    plt.close()
    
    # Distribution of movie success
    plt.figure(figsize=(8, 6))
    df['success'] = create_success_label(df)
    sns.countplot(data=df, x='success')
    plt.title("Distribution of Movie Success")
    plt.savefig(f'{output_dir}/success_distribution.png')
    plt.close()
    
    # Top 10 language distribution
    plt.figure(figsize=(12, 6))
    df['original_language'].value_counts().head(10).plot(kind='bar')
    plt.title("Top 10 Original Languages")
    plt.savefig(f'{output_dir}/language_distribution.png')
    plt.close()

def get_feature_sets(df, genres_encoded, lang_encoded):
    """
    Define feature sets for classification and regression tasks
    - Specifies numerical features for classification
    - Defines features for revenue prediction
    """
    # Classification features
    numerical_features_cls = ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']
    final_features_cls = numerical_features_cls
    
    # Regression features
    features_reg = ['vote_count', 'budget', 'popularity', 'runtime', 'vote_average']
    
    return final_features_cls, features_reg

def generate_model_performance_plots(results, output_dir='./visualizations'):
    """
    Generate comparative visualizations of model performance metrics
    - Creates bar plots comparing different models
    - Handles both classification and regression metrics
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Classification models performance comparison
    if 'mean_accuracy' in results[list(results.keys())[0]]:
        models = list(results.keys())
        accuracies = [results[m]['mean_accuracy'] for m in models]
        roc_aucs = [results[m]['mean_roc_auc'] for m in models]
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(models))
        width = 0.35
        
        plt.bar(x - width/2, accuracies, width, label='Accuracy')
        plt.bar(x + width/2, roc_aucs, width, label='ROC-AUC')
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Classification Models Performance Comparison')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/classification_performance.png')
        plt.close()
    
    # Regression models performance comparison
    if 'mean_mae' in results[list(results.keys())[0]]:
        models = list(results.keys())
        maes = [results[m]['mean_mae'] for m in models]
        rmses = [results[m]['mean_rmse'] for m in models]
        r2s = [results[m]['mean_r2'] for m in models]
        
        plt.figure(figsize=(15, 5))
        x = np.arange(len(models))
        width = 0.25
        
        plt.bar(x - width, maes, width, label='MAE')
        plt.bar(x, rmses, width, label='RMSE')
        plt.bar(x + width, r2s, width, label='R²')
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Regression Models Performance Comparison')
        plt.xticks(x, models, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{output_dir}/regression_performance.png')
        plt.close()

def analyze_feature_importance(model, feature_names, output_dir='./visualizations'):
    """
    Analyze and visualize feature importance for tree-based models
    - Creates bar plot of feature importances
    - Returns top 10 most important features
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title('Feature Importance')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance.png')
        plt.close()
        
        return [(feature_names[i], importances[i]) for i in indices[:10]]
    return None

def generate_prediction_analysis(y_true, y_pred, output_dir='./visualizations'):
    """
    Generate scatter plot of predicted vs actual values
    - Shows prediction accuracy
    - Includes perfect prediction line for reference
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Prediction vs True Values')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/prediction_analysis.png')
    plt.close()

def generate_residual_plot(y_true, y_pred, output_dir='./visualizations'):
    """
    Generate residual plot for regression analysis
    - Shows distribution of prediction errors
    - Helps identify potential model biases
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/residual_plot.png')
    plt.close() 
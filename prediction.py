"""
[Movie Success Prediction Pipeline]
This script implements the main workflow for predicting movie success and revenue.
It combines data processing, model training, and evaluation in a comprehensive pipeline.
"""

import pandas as pd
from data_processor import (load_and_preprocess_data, process_genres, 
                          process_languages, create_success_label, 
                          generate_data_visualizations, get_feature_sets,
                          generate_model_performance_plots, analyze_feature_importance,
                          generate_prediction_analysis, generate_residual_plot)
from model_trainer import ModelTrainer

def main():
    """
    Main execution function that orchestrates the entire prediction pipeline:
    1. Data loading and preprocessing
    2. Feature engineering
    3. Model training and evaluation
    4. Performance analysis and visualization
    """
    # Load and preprocess movie metadata
    df = load_and_preprocess_data('./movies_metadata.csv')
    
    # Process categorical features: genres and languages
    # Convert genres and languages into numerical features using encoding
    df, genres_encoded, mlb = process_genres(df)
    lang_encoded, lang_encoder = process_languages(df)
    
    # Generate exploratory data analysis visualizations
    # Creates correlation heatmap, success distribution, and language distribution plots
    generate_data_visualizations(df)
    
    # Define feature sets for classification and regression tasks
    # Classification: predict movie success (binary)
    # Regression: predict movie revenue (continuous)
    final_features_cls, features_reg = get_feature_sets(df, genres_encoded, lang_encoded)
    
    # Prepare feature matrices for classification
    # Combine numerical features with encoded categorical features
    X_cls = pd.concat([df[final_features_cls], genres_encoded, lang_encoded], axis=1)
    y_cls = create_success_label(df)
    
    # Prepare feature matrices for regression
    # Use selected numerical features for revenue prediction
    X_reg = df[features_reg]
    y_reg = df['revenue']
    
    # Initialize model trainer with 10-fold cross validation
    trainer = ModelTrainer(n_splits=10)
    
    # Train and evaluate classification models
    print("\n--- Classification Models ---")
    X_cls_scaled, y_cls = trainer.prepare_data(X_cls, y_cls)
    cls_results = trainer.train_classification_models(X_cls_scaled, y_cls)
    
    # Visualize classification model performance
    # Creates bar plots comparing accuracy, ROC-AUC, and F1 scores
    generate_model_performance_plots(cls_results)
    
    # Print classification metrics for each model
    for model_name, result in cls_results.items():
        print(f"\n{model_name} Results:")
        print(f"Mean Accuracy: {result['mean_accuracy']:.4f} (±{result['std_accuracy']:.4f})")
        print(f"Mean ROC-AUC: {result['mean_roc_auc']:.4f} (±{result['std_roc_auc']:.4f})")
        print(f"Mean F1-Score: {result['mean_f1']:.4f} (±{result['std_f1']:.4f})")
    
    # Train and evaluate regression models
    print("\n--- Regression Models ---")
    X_reg_scaled, y_reg = trainer.prepare_data(X_reg, y_reg)
    reg_results = trainer.train_regression_models(X_reg_scaled, y_reg)
    
    # Visualize regression model performance
    # Creates bar plots comparing MAE, RMSE, and R² scores
    generate_model_performance_plots(reg_results)
    
    # Print regression metrics for each model
    for model_name, result in reg_results.items():
        print(f"\n{model_name} Results:")
        print(f"Mean MAE: {result['mean_mae']:,.2f} (±{result['std_mae']:,.2f})")
        print(f"Mean RMSE: {result['mean_rmse']:,.2f} (±{result['std_rmse']:,.2f})")
        print(f"Mean R²: {result['mean_r2']:.4f} (±{result['std_r2']:.4f})")
    
    # Perform hyperparameter tuning for RandomForest classifier
    # Uses GridSearchCV to find optimal parameters
    print("\nTuning hyperparameters for RandomForest...")
    best_params, best_score = trainer.tune_random_forest(X_cls_scaled, y_cls, task='classification')
    
    print("\nBest Parameters:")
    print(f"Parameters: {best_params}")
    print(f"Score: {best_score:.4f}")
    
    # Analyze and visualize feature importance
    # Identifies and displays the most influential features
    best_model = cls_results['RandomForest']['model']
    feature_names = X_cls.columns
    important_features = analyze_feature_importance(best_model, feature_names)
    
    if important_features:
        print("\nTop 10 Important Features:")
        for feature, importance in important_features:
            print(f"{feature}: {importance:.4f}")
    
    # Generate regression analysis visualizations
    # Creates prediction vs actual plot and residual plot
    best_reg_model = max(reg_results.items(), key=lambda x: x[1]['mean_r2'])[0]
    y_pred_reg = reg_results[best_reg_model]['model'].predict(X_reg_scaled)
    generate_prediction_analysis(y_reg, y_pred_reg)
    generate_residual_plot(y_reg, y_pred_reg)

if __name__ == "__main__":
    main()
import pandas as pd
import numpy as np
import math
from collections import Counter
from flask import Flask, render_template, request, jsonify, send_file
from io import BytesIO
import json
import os

app = Flask(__name__) # Initialize Flask app

# --- Data Loading and Preprocessing ---
df = pd.read_excel('data_latih.xlsx') # Load data from Excel file

# Preprocess data: Split 'Tekanan_Darah' into 'Sistolik' and 'Diastolik'
df[['Sistolik', 'Diastolik']] = df['Tekanan_Darah'].str.split('/', expand=True).astype(int)
df = df.drop('Tekanan_Darah', axis=1) # Drop original 'Tekanan_Darah' column
df['Jenis_Kelamin'] = df['Jenis_Kelamin'].map({'L': 0, 'P': 1}) # Map 'Jenis_Kelamin' to numerical values
df['Nyeri_Dada'] = df['Nyeri_Dada'].map({'Tidak': 0, 'Ya': 1}) # Map 'Nyeri_Dada' to numerical values
df['Sesak_Napas'] = df['Sesak_Napas'].map({'Tidak': 0, 'Ya': 1}) # Map 'Sesak_Napas' to numerical values
df['Kelelahan'] = df['Kelelahan'].map({'Tidak': 0, 'Ya': 1}) # Map 'Kelelahan' to numerical values

# --- Data Splitting ---
train_data = df.sample(frac=0.8, random_state=42) # Split data into 80% training
test_data = df.drop(train_data.index) # Remaining 20% for testing

# --- C5.0 Decision Tree Implementation ---
def entropy(target, sample_weights=None):
    if sample_weights is None:
        sample_weights = np.ones(len(target))
    
    total_weight = np.sum(sample_weights)
    if total_weight == 0:
        return 0
        
    weighted_counts = {}
    for i, label in enumerate(target):
        weighted_counts[label] = weighted_counts.get(label, 0) + sample_weights[i]
        
    probabilities = [count / total_weight for count in weighted_counts.values()]
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

def information_gain(data, feature, target, sample_weights=None):
    if sample_weights is None:
        sample_weights = np.ones(len(data))

    total_entropy = entropy(data[target], sample_weights)
    
    values = data[feature].unique()
    weighted_entropy = 0
    total_weight = np.sum(sample_weights)

    if total_weight == 0:
        return 0

    for value in values:
        subset_indices = data[feature] == value
        subset_target = data.loc[subset_indices, target]
        subset_weights = sample_weights[subset_indices]
        
        subset_weight_sum = np.sum(subset_weights)
        if subset_weight_sum > 0:
             subset_entropy = entropy(subset_target, subset_weights)
             weighted_entropy += (subset_weight_sum / total_weight) * subset_entropy
             
    return total_entropy - weighted_entropy

def split_info(data, feature, sample_weights=None):
    if sample_weights is None:
        sample_weights = np.ones(len(data))

    total_weight = np.sum(sample_weights)
    if total_weight == 0:
        return 0 # Avoid division by zero and log(0)

    values = data[feature].unique()
    split_entropy = 0
    
    for value in values:
        subset_indices = data[feature] == value
        subset_weight = np.sum(sample_weights[subset_indices])
        
        if subset_weight > 0:
            probability = subset_weight / total_weight
            split_entropy -= probability * math.log2(probability)
            
    # Handle potential floating point issues causing split_entropy to be slightly negative or zero
    return split_entropy if split_entropy > 1e-9 else 1e-9 # Return a small positive number if zero or negative

def gain_ratio(data, feature, target, sample_weights=None):
    ig = information_gain(data, feature, target, sample_weights)
    si = split_info(data, feature, sample_weights)
    # si is adjusted in split_info to avoid division by zero
    return ig / si

class DecisionNode:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right

def build_tree(data, target, features, depth=0, max_depth=3, min_samples_split=2, sample_weights=None):
    # Initialize weights if not provided (for the first call or non-boosting scenario)
    if sample_weights is None:
        sample_weights = np.ones(len(data))

    # --- Base Cases ---
    # 1. All samples belong to the same class
    unique_classes = data[target].unique()
    if len(unique_classes) == 1:
        return DecisionNode(value=unique_classes[0])

    # 2. Max depth reached or minimum samples for split not met
    #    Use weighted count for the majority class if weights are present
    if depth == max_depth or len(data) < min_samples_split or np.sum(sample_weights) < min_samples_split: # Check weighted samples too
        weighted_counts = {}
        for i, label in enumerate(data[target]):
            weighted_counts[label] = weighted_counts.get(label, 0) + sample_weights[i]
        majority_class = max(weighted_counts, key=weighted_counts.get) if weighted_counts else None # Handle empty data case
        return DecisionNode(value=majority_class)

    # 3. No features left to split on
    if not features:
        weighted_counts = {}
        for i, label in enumerate(data[target]):
            weighted_counts[label] = weighted_counts.get(label, 0) + sample_weights[i]
        majority_class = max(weighted_counts, key=weighted_counts.get) if weighted_counts else None
        return DecisionNode(value=majority_class)

    # --- Find Best Split ---
    best_gain_ratio = -1
    best_feature = None
    best_threshold = None # For numerical features
    best_split_type = None # 'categorical' or 'numerical'

    current_features = list(features) # Create a modifiable list

    for feature in current_features:
        unique_values = data[feature].unique()

        # Skip features with only one unique value in the current subset
        if len(unique_values) <= 1:
            continue

        if pd.api.types.is_numeric_dtype(data[feature]):
            # --- Handle Numerical Features ---
            # Sort unique values to find potential split points
            sorted_values = sorted(unique_values)
            for i in range(len(sorted_values) - 1):
                # Threshold is the midpoint between consecutive unique values
                threshold = (sorted_values[i] + sorted_values[i+1]) / 2
                
                # Create temporary binary feature based on threshold
                temp_feature_name = f"{feature}_le_{threshold}"
                data[temp_feature_name] = (data[feature] <= threshold).astype(int)
                
                # Calculate gain ratio using the temporary binary feature
                gain = gain_ratio(data, temp_feature_name, target, sample_weights)
                
                # Clean up temporary column
                del data[temp_feature_name]

                if gain > best_gain_ratio:
                    best_gain_ratio = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_split_type = 'numerical'

        else: # --- Handle Categorical Features --- (Assuming Gain Ratio works directly for multi-value splits)
             # Note: C5.0 often groups categories, but for simplicity, we'll use standard gain ratio here.
             # A more complex implementation would evaluate subsets of categories.
            gain = gain_ratio(data, feature, target, sample_weights)
            if gain > best_gain_ratio:
                best_gain_ratio = gain
                best_feature = feature
                best_threshold = None # No threshold for categorical split as implemented here
                best_split_type = 'categorical'


    # --- Check if a useful split was found ---
    # If no split improves information gain (gain ratio <= 0), create a leaf node
    if best_gain_ratio <= 0:
        weighted_counts = {}
        for i, label in enumerate(data[target]):
            weighted_counts[label] = weighted_counts.get(label, 0) + sample_weights[i]
        majority_class = max(weighted_counts, key=weighted_counts.get) if weighted_counts else None
        return DecisionNode(value=majority_class)

    # --- Split Data and Recurse ---
    remaining_features = [f for f in current_features if f != best_feature] # Pass remaining features

    if best_split_type == 'numerical':
        left_indices = data[best_feature] <= best_threshold
        right_indices = data[best_feature] > best_threshold
        
        left_data = data[left_indices]
        right_data = data[right_indices]
        left_weights = sample_weights[left_indices]
        right_weights = sample_weights[right_indices]

        # Check if splits are empty or too small
        if len(left_data) == 0 or len(right_data) == 0 or np.sum(left_weights) == 0 or np.sum(right_weights) == 0:
             weighted_counts = {}
             for i, label in enumerate(data[target]):
                 weighted_counts[label] = weighted_counts.get(label, 0) + sample_weights[i]
             majority_class = max(weighted_counts, key=weighted_counts.get) if weighted_counts else None
             return DecisionNode(value=majority_class)


        left_child = build_tree(left_data, target, remaining_features, depth + 1, max_depth, min_samples_split, left_weights)
        right_child = build_tree(right_data, target, remaining_features, depth + 1, max_depth, min_samples_split, right_weights)
        
        return DecisionNode(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    elif best_split_type == 'categorical':
         # For categorical, we need to handle multiple branches if we were fully implementing C5.0 grouping.
         # Simplified: Create branches for each unique value. This is more like ID3/C4.5 splitting.
         # A true C5.0 implementation would group categories based on gain.
         # For now, let's stick to a binary split based on the most common value vs others for simplicity,
         # or revert to the previous less accurate numerical split logic if categorical is too complex now.
         # Let's try a simple binary split: most common category vs rest
         
         # --- Reverting to a simpler (less C5.0 accurate) categorical split for now ---
         # This part needs significant refinement for true C5.0 categorical handling (grouping)
         # Using the previous dummy variable approach as a placeholder
         dummies = pd.get_dummies(data[best_feature], prefix=best_feature, drop_first=True) # Use drop_first for binary split idea
         best_dummy_feature = None
         best_dummy_gain = -1

         # Re-evaluate gain for dummy variables derived from the chosen categorical feature
         for dummy in dummies.columns:
             gain = gain_ratio(data.assign(**{dummy: dummies[dummy]}), dummy, target, sample_weights)
             if gain > best_dummy_gain:
                 best_dummy_gain = gain
                 best_dummy_feature = dummy

         if best_dummy_feature is None: # No improvement found even with dummies
            weighted_counts = {}
            for i, label in enumerate(data[target]):
                weighted_counts[label] = weighted_counts.get(label, 0) + sample_weights[i]
            majority_class = max(weighted_counts, key=weighted_counts.get) if weighted_counts else None
            return DecisionNode(value=majority_class)

         # Split based on the best dummy variable
         temp_data = data.assign(**{best_dummy_feature: dummies[best_dummy_feature]})
         left_indices = temp_data[best_dummy_feature] == 0
         right_indices = temp_data[best_dummy_feature] == 1

         left_data = data[left_indices]
         right_data = data[right_indices]
         left_weights = sample_weights[left_indices]
         right_weights = sample_weights[right_indices]

         # Check if splits are empty or too small
         if len(left_data) == 0 or len(right_data) == 0 or np.sum(left_weights) == 0 or np.sum(right_weights) == 0:
             weighted_counts = {}
             for i, label in enumerate(data[target]):
                 weighted_counts[label] = weighted_counts.get(label, 0) + sample_weights[i]
             majority_class = max(weighted_counts, key=weighted_counts.get) if weighted_counts else None
             return DecisionNode(value=majority_class)

         # Use the dummy feature name and threshold 0.5 (or 1) for the node
         # This represents the split: category corresponding to dummy vs others
         left_child = build_tree(left_data, target, remaining_features, depth + 1, max_depth, min_samples_split, left_weights)
         right_child = build_tree(right_data, target, remaining_features, depth + 1, max_depth, min_samples_split, right_weights)

         # The 'feature' stored is the dummy, threshold is effectively 0.5 or 1
         return DecisionNode(feature=best_dummy_feature, threshold=0.5, left=left_child, right=right_child) # Threshold 0.5 for <= comparison

    else: # Should not happen if a split was found
        weighted_counts = {}
        for i, label in enumerate(data[target]):
            weighted_counts[label] = weighted_counts.get(label, 0) + sample_weights[i]
        majority_class = max(weighted_counts, key=weighted_counts.get) if weighted_counts else None
        return DecisionNode(value=majority_class)

# Initial model training
target = 'Penyakit_Jantung'
features = df.columns.drop(target)
tree = build_tree(train_data, target, features, max_depth=3)

# Global variables to store the potentially boosted model
boosted_trees = []
boosted_alphas = []
is_boosted = False # Flag to indicate if the current model is boosted

def predict_single_tree(node, sample):
    """Helper function to predict using a single decision tree."""
    if node is None: # Handle case where a branch might be missing (should ideally not happen in a full tree)
        # print("Warning: Encountered None node during prediction.")
        return None # Or return a default value based on parent?

    if node.value is not None: # Leaf node
        return node.value
        
    feature = node.feature
    
    # --- Handle Numerical Feature ---
    if node.threshold is not None and feature in sample.index and pd.api.types.is_numeric_dtype(sample[feature]):
         try:
             sample_value = sample[feature]
             if sample_value <= node.threshold:
                 return predict_single_tree(node.left, sample)
             else:
                 return predict_single_tree(node.right, sample)
         except TypeError: # Handle potential comparison errors
              # print(f"Warning: Type error comparing {sample[feature]} and {node.threshold} for feature '{feature}'. Defaulting left.")
              # Fallback: Go left (arbitrary, could be improved by passing majority class down)
              return predict_single_tree(node.left, sample)

    # --- Handle Categorical Feature (using dummy variable name) ---
    # Assumes dummy feature name like 'OriginalFeature_CategoryValue' and threshold 0.5
    elif node.threshold == 0.5 and feature in sample.index: # Check if it looks like our dummy split
         # This logic is still fragile and depends on the exact dummy naming in build_tree.
         # It assumes the 'feature' name stored in the node IS the dummy variable name.
         # We need to infer the original feature and the category value from the dummy name.
         
         # Example: feature = 'Nyeri_Dada_Ya' (if build_tree created this)
         # We need to check if sample['Nyeri_Dada'] == 'Ya' (or its mapped value, e.g., 1)
         
         # --- This requires a more robust way to store split info in the Node ---
         # --- Simplified/Placeholder Logic: ---
         # Let's assume the feature name directly corresponds to a column that *might* exist
         # in the sample IF it was preprocessed with the same get_dummies. This is unlikely for raw input.
         # A better build_tree would store: node.feature = 'Nyeri_Dada', node.category_split = 'Ya'
         
         # Fallback: If we can't reliably determine the split, default to a branch.
         # print(f"Warning: Cannot reliably predict categorical split for '{feature}'. Defaulting left.")
         return predict_single_tree(node.left, sample)


    # --- Feature not found or other issues ---
    else:
        # Feature might be missing in the sample, or it's a type not handled above.
        # print(f"Warning: Feature '{feature}' not found in sample or type mismatch. Defaulting left.")
        # Defaulting left is arbitrary. A better strategy might involve using majority class if available at node.
        return predict_single_tree(node.left, sample)


def predict(sample, use_boosting=False):
    """Predicts the class for a single sample, using boosting if enabled globally."""
    global boosted_trees, boosted_alphas, is_boosted, tree # Access global model

    current_model_is_boosted = is_boosted # Check the global flag

    if use_boosting and current_model_is_boosted and boosted_trees and boosted_alphas:
        # --- Boosted Prediction ---
        weighted_votes = {}
        # total_alpha = sum(boosted_alphas) # Not strictly needed for max vote

        for b_tree, alpha in zip(boosted_trees, boosted_alphas):
            pred = predict_single_tree(b_tree, sample)
            
            if pred is not None: # Ensure prediction is valid
                 weighted_votes[pred] = weighted_votes.get(pred, 0) + alpha

        if not weighted_votes:
             # Handle case where no tree could make a prediction
             # print("Warning: No boosted tree made a prediction for the sample.")
             return None # Or a default value

        # Return class with the highest weighted vote
        return max(weighted_votes, key=weighted_votes.get)
        
    elif not use_boosting and tree: # Use the single global tree if not boosting OR if boosting failed/not selected
        # --- Single Tree Prediction ---
         return predict_single_tree(tree, sample)
    elif use_boosting and not current_model_is_boosted and tree:
         # Requested boosting but model isn't boosted, use single tree
         # print("Warning: Boosting requested but model is not boosted. Using single tree.")
         return predict_single_tree(tree, sample)
    else:
         # No model trained or available
         # print("Error: No model available for prediction.")
         return None # Or raise an error

# Generate predictions
results = []
for idx, row in df.iterrows():
    pred = predict(tree, row)
    result_dict = {
        'Usia': row['Usia'],
        'Jenis_Kelamin': 'Laki-laki' if row['Jenis_Kelamin'] == 0 else 'Perempuan',
        'Sistolik': row['Sistolik'],
        'Diastolik': row['Diastolik'],
        'Kolesterol': row['Kolesterol'],
        'Gula_Darah': row['Gula_Darah'],
        'Nyeri_Dada': 'Ya' if row['Nyeri_Dada'] == 1 else 'Tidak',
        'Sesak_Napas': 'Ya' if row['Sesak_Napas'] == 1 else 'Tidak',
        'Kelelahan': 'Ya' if row['Kelelahan'] == 1 else 'Tidak',
        'Denyut_Jantung': row['Denyut_Jantung'],
        'Aktual': row[target],
        'Prediksi': pred
    }
    results.append(result_dict)

# Routes for HTML Pages
@app.route('/')
def dashboard():
    return render_template('index.html', page='dashboard')

@app.route('/training_data')
def training_data_page():
    return render_template('training_data.html', page='training_data')

@app.route('/test_data')
def test_data_page():
    return render_template('testing_data.html', page='test_data')

@app.route('/training')
def training_page():
    return render_template('training_model.html', page='training')

@app.route('/tree_visualization')
def tree_visualization_page():
    return render_template('d_tree.html', page='tree_visualization')

@app.route('/rules_visualization')
def rules_visualization_page():
    return render_template('rules.html', page='rules_visualization')

@app.route('/results')
def results_page():
    return render_template('result.html', page='results')

# API Endpoints


@app.route('/api/predictions')
def get_predictions():
    # Convert results to DataFrame for easier manipulation
    results_df = pd.DataFrame(results)

    # Calculate total predictions
    total_predictions = len(results_df)

    # Calculate correct predictions
    correct_predictions = len(results_df[results_df['Aktual'] == results_df['Prediksi']])

    # Calculate overall accuracy
    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

    # Build confusion matrix
    # For simplicity, let's classify predictions into binary classes: "Tidak Ada" (No Disease) and "Ya" (Disease)
    results_df['Aktual_Binary'] = results_df['Aktual'].apply(lambda x: 'Tidak Ada' if x == 'Tidak Ada' else 'Ya')
    results_df['Prediksi_Binary'] = results_df['Prediksi'].apply(lambda x: 'Tidak Ada' if x == 'Tidak Ada' else 'Ya')

    confusion_matrix = {
        'Tidak Ada': {'Tidak Ada': 0, 'Ya': 0},
        'Ya': {'Tidak Ada': 0, 'Ya': 0}
    }

    for _, row in results_df.iterrows():
        actual = row['Aktual_Binary']
        predicted = row['Prediksi_Binary']
        confusion_matrix[actual][predicted] += 1

    # Calculate class-wise accuracy
    class_accuracy = []
    for cls in ['Tidak Ada', 'Ya']:
        total_cls = len(results_df[results_df['Aktual_Binary'] == cls])
        correct_cls = confusion_matrix[cls][cls]
        cls_accuracy = (correct_cls / total_cls * 100) if total_cls > 0 else 0
        performance = (
            'Excellent' if cls_accuracy >= 80 else
            'Good' if cls_accuracy >= 60 else
            'Fair' if cls_accuracy >= 40 else
            'Poor'
        )
        class_accuracy.append({
            'class': cls,
            'correct': correct_cls,
            'total': total_cls,
            'accuracy': cls_accuracy,
            'performance': performance
        })

    # Calculate Precision, Recall, and F1 Score
    # For binary classification (Tidak Ada vs Ya)
    tp = confusion_matrix['Ya']['Ya']
    fp = confusion_matrix['Tidak Ada']['Ya']
    fn = confusion_matrix['Ya']['Tidak Ada']
    tn = confusion_matrix['Tidak Ada']['Tidak Ada']

    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    # Prepare the response
    response = {
        'total': total_predictions,
        'correct': correct_predictions,
        'accuracy': accuracy,
        'confusionMatrix': confusion_matrix,
        'classAccuracy': class_accuracy,
        'precision': precision,
        'recall': recall,
        'f1Score': f1_score,
        'data': results  # Include raw data for filtering on the frontend
    }

    return jsonify(response)

@app.route('/api/training_data')
def get_training_data():
    page = int(request.args.get('page', 1))
    per_page = 10
    start = (page - 1) * per_page
    end = start + per_page
    
    # Convert train_data to list of dictionaries
    train_list = train_data.to_dict('records')
    
    # Calculate statistics for each disease class
    total_records = len(train_list)
    disease_counts = {}
    for record in train_list:
        disease = record['Penyakit_Jantung']
        disease_counts[disease] = disease_counts.get(disease, 0) + 1
    
    # Calculate percentages for each disease
    disease_stats = [
        {
            'disease': disease,
            'count': count,
            'percent': round((count / total_records * 100) if total_records > 0 else 0, 2)
        }
        for disease, count in disease_counts.items()
    ]
    
    # Format data for frontend
    formatted_data = [
        {
            'id': idx + 1,  # Generate ID dynamically
            'usia': record['Usia'],
            'jenis_kelamin': 'Laki-laki' if record['Jenis_Kelamin'] == 0 else 'Perempuan',
            'sistolik': record['Sistolik'],
            'diastolik': record['Diastolik'],
            'kolesterol': 'Tinggi' if record['Kolesterol'] > 200 else 'Normal',
            'gula_darah': 'Tinggi' if record['Gula_Darah'] > 120 else 'Normal',
            'nyeri_dada': 'Ya' if record['Nyeri_Dada'] == 1 else 'Tidak',
            'sesak_napas': 'Ya' if record['Sesak_Napas'] == 1 else 'Tidak',
            'kelelahan': 'Ya' if record['Kelelahan'] == 1 else 'Tidak',
            'denyut_jantung': record['Denyut_Jantung'],
            'penyakit_jantung': record['Penyakit_Jantung']
        }
        for idx, record in enumerate(train_list)
    ]
    
    # Paginate data
    paginated_data = formatted_data[start:end]
    
    # Return response
    return jsonify({
        'data': paginated_data,
        'total': total_records,
        'stats': {
            'total_records': total_records,
            'disease_stats': disease_stats
        },
        'page': page,
        'per_page': per_page
    })
@app.route('/api/results')
def get_results():
    page = int(request.args.get('page', 1))
    per_page = 10
    start = (page - 1) * per_page
    end = start + per_page
    
    # Convert results to DataFrame for easier manipulation
    results_df = pd.DataFrame(results)
    
    # Calculate statistics
    total_predictions = len(results_df)
    correct_predictions = len(results_df[results_df['Aktual'] == results_df['Prediksi']])
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    
    # Build confusion matrix (binary: Tidak Ada vs Ya)
    results_df['Aktual_Binary'] = results_df['Aktual'].apply(lambda x: 'Tidak' if x == 'Tidak Ada' else 'Ya')
    results_df['Prediksi_Binary'] = results_df['Prediksi'].apply(lambda x: 'Tidak' if x == 'Tidak Ada' else 'Ya')
    
    confusion_matrix = {
        'Tidak': {'Tidak': 0, 'Ya': 0},
        'Ya': {'Tidak': 0, 'Ya': 0}
    }
    
    for _, row in results_df.iterrows():
        actual = row['Aktual_Binary']
        predicted = row['Prediksi_Binary']
        confusion_matrix[actual][predicted] += 1
    
    # Calculate breakdown by class
    classes = ['Tidak', 'Ya']
    breakdown = []
    for cls in classes:
        class_data = results_df[results_df['Aktual_Binary'] == cls]
        correct_count = len(class_data[class_data['Prediksi_Binary'] == cls])
        incorrect_count = len(class_data) - correct_count
        breakdown.append({
            'class': cls,
            'correct': correct_count,
            'incorrect': incorrect_count
        })
    
    # Prepare paginated data
    paginated_results = results[start:end]
    
    # Format data to match result.html expectations
    formatted_data = [
        {
            'usia': item['Usia'],
            'jenis_kelamin': item['Jenis_Kelamin'],
            'sistolik': item['Sistolik'],
            'diastolik': item['Diastolik'],
            'kolesterol': item['Kolesterol'],  # Assuming numerical value as in app.py
            'gula_darah': item['Gula_Darah'],
            'nyeri_dada': item['Nyeri_Dada'],
            'sesak_napas': item['Sesak_Napas'],
            'kelelahan': item['Kelelahan'],
            'denyut_jantung': item['Denyut_Jantung'],
            'aktual': item['Aktual'],
            'prediksi': item['Prediksi'],
            'status': 'Correct' if item['Aktual'] == item['Prediksi'] else 'Incorrect'
        }
        for item in paginated_results
    ]
    
    # Return response
    return jsonify({
        'data': formatted_data,
        'total': total_predictions,
        'correct': correct_predictions,
        'accuracy': round(accuracy, 2),
        'breakdown': breakdown,
        'confusionMatrix': confusion_matrix,
        'page': page,
        'per_page': per_page
    })
@app.route('/api/tree_visualization')
def get_tree_visualization():
    global tree, is_boosted, boosted_trees # Access global model state

    # Determine which tree to visualize
    # If boosted, visualize the first tree as a representative
    tree_to_visualize = tree if not is_boosted else (boosted_trees[0] if boosted_trees else None)

    if not tree_to_visualize:
        return jsonify({'error': 'No trained model available to visualize'}), 400

    # Use the same tree_to_dict logic as in /train_model for consistency
    # (Define it locally here or ensure it's accessible globally/imported)
    def tree_to_dict(node):
        if node is None: return None
        if node.value is not None:
            # Simplified version for API: less info needed than in training summary
            # Include 'samples' if available/easily calculable, otherwise omit for API speed
            return {'name': f"Predict: {node.value}", 'value': 'Leaf', 'class_value': node.value, 'samples': 0, 'impurity': 0}

        node_name = f"{node.feature}"
        if node.threshold is not None:
             # Format threshold nicely, especially for floats
             threshold_str = f"{node.threshold:.2f}" if isinstance(node.threshold, float) else str(node.threshold)
             # Use a more descriptive format for the node name
             if node.threshold == 0.5 and '_' in node.feature: # Heuristic for dummy categorical
                 # Try to make dummy name more readable, e.g., "Nyeri_Dada_Ya <= 0.5" -> "Nyeri_Dada is Ya"
                 # This is still a guess based on naming convention
                 parts = node.feature.split('_')
                 # Check if the last part could be a category value (needs better check)
                 # A more robust approach is needed in build_tree to store this info
                 if len(parts) > 1:
                      # Attempt to reconstruct: assumes format OriginalFeature_CategoryValue
                      original_feature = parts[0]
                      category_value = '_'.join(parts[1:]) # Handle category names with underscores
                      node_name = f"{original_feature} is {category_value}"
                 else: # Fallback if parsing fails
                      node_name = f"{node.feature} <= {threshold_str}"
             else: # Numerical split
                 node_name = f"{node.feature} <= {threshold_str}"
        else: # Should not happen for decision nodes in this implementation
             node_name = f"Split on {node.feature}"


        # Recursively call for children, checking existence
        left_child_dict = tree_to_dict(node.left) if hasattr(node, 'left') else None
        right_child_dict = tree_to_dict(node.right) if hasattr(node, 'right') else None
        
        # Filter out None children before returning
        valid_children = [child for child in [left_child_dict, right_child_dict] if child is not None]

        return {
            'name': node_name,
            'value': 'Decision',
            # Ensure children is always a list, even if empty
            'children': valid_children
        }

    # Convert the selected tree structure to JSON
    try:
        tree_structure_json = tree_to_dict(tree_to_visualize)
        if tree_structure_json is None: # Handle case where root itself is None
             return jsonify({'error': 'Model structure is empty or invalid'}), 500
        return jsonify(tree_structure_json)
    except Exception as e:
        print(f"Error generating tree visualization JSON: {e}")
        # Consider logging the full traceback for debugging
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to generate tree structure: {str(e)}'}), 500

@app.route('/api/rules')
def get_rules():
    rules = []
    def extract_rules(node, rule_conditions=[], path_data=train_data):
        if node.value is not None:
            # Hitung confidence dan coverage berdasarkan data
            subset = path_data
            if not subset.empty:
                correct = len(subset[subset[target] == node.value])
                total = len(subset)
                confidence = correct / total if total > 0 else 0
                coverage = total / len(train_data)
            else:
                confidence = 0
                coverage = 0
            rules.append({
                'id': len(rules) + 1,
                'conditions': rule_conditions,  # List of conditions
                'prediction': node.value,
                'confidence': round(confidence, 2),
                'coverage': round(coverage, 2),
                'samples': total
            })
            return
        if node.feature and node.threshold:
            left_data = path_data[path_data[node.feature] <= node.threshold]
            right_data = path_data[path_data[node.feature] > node.threshold]
            extract_rules(node.left, rule_conditions + [f"{node.feature} <= {node.threshold}"], left_data)
            extract_rules(node.right, rule_conditions + [f"{node.feature} > {node.threshold}"], right_data)
    
    extract_rules(tree)
    
    # Hitung statistik
    total_rules = len(rules)
    prediction_counts = {}
    total_conditions = 0
    for rule in rules:
        pred = rule['prediction']
        prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
        total_conditions += len(rule['conditions'])
    avg_conditions = round(total_conditions / total_rules, 1) if total_rules > 0 else 0
    
    stats = {
        'total_rules': total_rules,
        'prediction_counts': prediction_counts,
        'avg_conditions': avg_conditions
    }
    
    # Paginasi
    page = int(request.args.get('page', 1))
    per_page = 10
    start = (page - 1) * per_page
    end = start + per_page
    paginated_rules = rules[start:end]
    
    return jsonify({
        'data': paginated_rules,
        'total': len(rules),
        'stats': stats,
        'page': page,
        'per_page': per_page
    })

# File Upload Endpoints
@app.route('/upload_training_data', methods=['POST'])
def upload_training_data():
    file = request.files['file']
    if file and file.filename.endswith('.csv'):
        global train_data
        train_data = pd.read_csv(file)
        return jsonify({'status': 'success', 'message': 'Data uploaded successfully'})
    return jsonify({'status': 'error', 'message': 'Invalid file'})

@app.route('/upload_test_data', methods=['POST'])
def upload_test_data():
    global test_data
    file = request.files.get('file')
    
    # Validasi file
    if not file:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
    if not file.filename.endswith('.csv'):
        return jsonify({'status': 'error', 'message': 'File must be a CSV'}), 400
    
    try:
        # Baca file CSV
        new_test_data = pd.read_csv(file)
        
        # Validasi kolom yang diperlukan
        required_columns = ['Usia', 'Jenis_Kelamin', 'Tekanan_Darah', 'Kolesterol', 'Gula_Darah', 
                           'Nyeri_Dada', 'Sesak_Napas', 'Kelelahan', 'Denyut_Jantung', 'Penyakit_Jantung']
        if not all(col in new_test_data.columns for col in required_columns):
            return jsonify({'status': 'error', 'message': 'Missing required columns in CSV'}), 400
        
        # Transformasi data
        # Pisahkan Tekanan_Darah menjadi Sistolik dan Diastolik
        new_test_data[['Sistolik', 'Diastolik']] = new_test_data['Tekanan_Darah'].str.split('/', expand=True).astype(int)
        new_test_data = new_test_data.drop('Tekanan_Darah', axis=1)
        
        # Konversi kolom kategorikal
        new_test_data['Jenis_Kelamin'] = new_test_data['Jenis_Kelamin'].map({'L': 0, 'P': 1, 'Laki-laki': 0, 'Perempuan': 1}).fillna(new_test_data['Jenis_Kelamin'])
        new_test_data['Nyeri_Dada'] = new_test_data['Nyeri_Dada'].map({'Tidak': 0, 'Ya': 1, 0: 0, 1: 1}).fillna(new_test_data['Nyeri_Dada'])
        new_test_data['Sesak_Napas'] = new_test_data['Sesak_Napas'].map({'Tidak': 0, 'Ya': 1, 0: 0, 1: 1}).fillna(new_test_data['Sesak_Napas'])
        new_test_data['Kelelahan'] = new_test_data['Kelelahan'].map({'Tidak': 0, 'Ya': 1, 0: 0, 1: 1}).fillna(new_test_data['Kelelahan'])
        
        # Validasi nilai setelah konversi
        if new_test_data[['Jenis_Kelamin', 'Nyeri_Dada', 'Sesak_Napas', 'Kelelahan']].isnull().any().any():
            return jsonify({'status': 'error', 'message': 'Invalid values in categorical columns'}), 400
        
        # Simpan data ke variabel global
        test_data = new_test_data
        
        return jsonify({
            'status': 'success',
            'message': f'Successfully uploaded {len(new_test_data)} records',
            'records': len(new_test_data)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error processing file: {str(e)}'}), 500
# Model Training Endpoint
@app.route('/train_model', methods=['POST'])
def train_model():
    # Access global variables to store the model
    global tree, results, train_data, target, features
    global boosted_trees, boosted_alphas, is_boosted

    # Validate training data
    if train_data.empty:
        return jsonify({'status': 'error', 'message': 'No training data available'}), 400
    if target not in train_data.columns:
         return jsonify({'status': 'error', 'message': f'Target column "{target}" not found in training data'}), 400
    
    # Ensure features are correctly defined (excluding target)
    current_features = [col for col in train_data.columns if col != target]
    if not current_features:
         return jsonify({'status': 'error', 'message': 'No features found in training data (excluding target)'}), 400
    features = current_features # Update global features list based on current train_data

    # Get parameters from form
    try:
        max_depth = int(request.form.get('max_depth', 3))
        min_samples_split = int(request.form.get('min_samples_split', 2))
        # Confidence is for pruning - not implemented in this manual build_tree
        # confidence = float(request.form.get('confidence', 0.25))
        boosting = request.form.get('boosting', 'false').lower() == 'true'
        n_estimators = int(request.form.get('n_estimators', 10)) if boosting else 1
    except ValueError:
         return jsonify({'status': 'error', 'message': 'Invalid training parameters (must be numbers)'}), 400


    # --- Model Training ---
    boosted_trees = [] # Clear previous boosted model
    boosted_alphas = []
    is_boosted = False # Reset flag

    if boosting:
        print(f"Starting Boosting with {n_estimators} estimators...")
        is_boosted = True
        sample_weights = np.ones(len(train_data)) / len(train_data) # Initialize weights

        for i in range(n_estimators):
            print(f"Building tree {i+1}/{n_estimators}...")
            # Build tree using current sample weights
            # Make sure 'features' used here is the updated list
            current_tree = build_tree(train_data, target, features,
                                      max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                      sample_weights=sample_weights)

            if current_tree is None: # Handle case where tree building fails
                 print(f"Warning: Tree {i+1} could not be built.")
                 # Option: Stop boosting, or skip this estimator? Let's skip.
                 continue

            # Predict on training data using the *single* current tree
            predictions = [predict_single_tree(current_tree, row) for _, row in train_data.iterrows()]

            # Calculate weighted error rate
            actuals = train_data[target].values
            incorrect = np.array([1 if pred != true else 0 for pred, true in zip(predictions, actuals)])
            
            # Mask weights where prediction failed (if predict_single_tree returns None)
            valid_preds_mask = np.array([p is not None for p in predictions])
            masked_weights = sample_weights[valid_preds_mask]
            masked_incorrect = incorrect[valid_preds_mask]
            
            total_masked_weight = np.sum(masked_weights)

            if total_masked_weight == 0: # Avoid division by zero if all predictions failed or weights are zero
                 error = 0.5 # Assign neutral error to avoid issues, maybe stop boosting?
                 print(f"Warning: Total weight for error calculation is zero in tree {i+1}. Assigning neutral error.")
            else:
                 error = np.dot(masked_weights, masked_incorrect) / total_masked_weight

            # Prevent error from being exactly 0 or 1 for alpha calculation stability
            error = np.clip(error, 1e-10, 1 - 1e-10)

            # Calculate estimator weight (alpha)
            alpha = 0.5 * np.log((1 - error) / error)

            # Update sample weights: increase weight for misclassified samples
            # weight_update_factors = np.exp(alpha * (2 * incorrect - 1)) # Maps 1 -> alpha, 0 -> -alpha
            weight_update_factors = np.array([np.exp(alpha) if inc == 1 else np.exp(-alpha) for inc in incorrect])
            
            sample_weights *= weight_update_factors
            
            # Normalize weights
            total_weight = np.sum(sample_weights)
            if total_weight == 0 or not np.isfinite(total_weight):
                 print("Warning: Sample weights became zero or non-finite. Resetting weights.")
                 sample_weights = np.ones(len(train_data)) / len(train_data) # Reset if weights explode/vanish
            else:
                 sample_weights /= total_weight

            # Store the tree and its weight
            boosted_trees.append(current_tree)
            boosted_alphas.append(alpha)
            print(f"Tree {i+1}: Error={error:.4f}, Alpha={alpha:.4f}")

        if not boosted_trees: # If all trees failed to build
             return jsonify({'status': 'error', 'message': 'Boosting failed: No trees were built.'}), 500
        # Use the first tree for single-tree metrics/visualization (optional)
        tree = boosted_trees[0] # Assign the first tree to the global 'tree' for compatibility
        print("Boosting complete.")

    else: # --- Single Tree Training ---
        print("Starting Single Tree Training...")
        is_boosted = False
        # Make sure 'features' used here is the updated list
        tree = build_tree(train_data, target, features,
                          max_depth=max_depth,
                          min_samples_split=min_samples_split)
        boosted_trees = [] # Ensure these are empty
        boosted_alphas = []
        print("Single tree training complete.")


    # --- Post-Training Steps ---
    if tree is None and not is_boosted: # Check if single tree training failed
         return jsonify({'status': 'error', 'message': 'Model training failed: Could not build tree.'}), 500
    elif is_boosted and not boosted_trees: # Check if boosting failed
         return jsonify({'status': 'error', 'message': 'Model training failed: Could not build boosted ensemble.'}), 500

    # Calculate metrics based on the primary tree (first tree if boosted, or the single tree)
    display_tree = tree # Use the globally assigned tree (either single or first boosted)
    tree_metrics = {'depth': 0, 'nodes': 0, 'leaves': 0}
    tree_structure = {}
    if display_tree:
         try:
             # Define metric calculation functions locally or ensure they are accessible
             # These need to handle None nodes potentially returned by build_tree
             def calculate_tree_depth(node):
                 if node is None or node.value is not None: return 0
                 # Check if children exist before recursing
                 left_depth = calculate_tree_depth(node.left) if hasattr(node, 'left') else 0
                 right_depth = calculate_tree_depth(node.right) if hasattr(node, 'right') else 0
                 return max(left_depth, right_depth) + 1

             def calculate_total_nodes(node):
                 if node is None: return 0
                 left_nodes = calculate_total_nodes(node.left) if hasattr(node, 'left') else 0
                 right_nodes = calculate_total_nodes(node.right) if hasattr(node, 'right') else 0
                 return 1 + left_nodes + right_nodes

             def calculate_leaf_nodes(node):
                 if node is None: return 0
                 if node.value is not None: return 1
                 left_leaves = calculate_leaf_nodes(node.left) if hasattr(node, 'left') else 0
                 right_leaves = calculate_leaf_nodes(node.right) if hasattr(node, 'right') else 0
                 return left_leaves + right_leaves

             tree_metrics = {
                 'depth': calculate_tree_depth(display_tree),
                 'nodes': calculate_total_nodes(display_tree),
                 'leaves': calculate_leaf_nodes(display_tree)
             }

             # Define tree_to_dict locally or ensure accessible
             def tree_to_dict(node):
                 if node is None: return None
                 if node.value is not None:
                     return {'name': f"Predict: {node.value}", 'value': 'Leaf'}
                 
                 node_name = f"{node.feature}"
                 if node.threshold is not None:
                      # Format threshold nicely, especially for floats
                      threshold_str = f"{node.threshold:.2f}" if isinstance(node.threshold, float) else str(node.threshold)
                      node_name += f" <= {threshold_str}"

                 # Recursively call for children, checking existence
                 left_child_dict = tree_to_dict(node.left) if hasattr(node, 'left') else None
                 right_child_dict = tree_to_dict(node.right) if hasattr(node, 'right') else None
                 
                 return {
                     'name': node_name,
                     'value': 'Decision',
                     # Ensure children list doesn't contain None if a branch is missing
                     'children': [child for child in [left_child_dict, right_child_dict] if child is not None]
                 }
             tree_structure = tree_to_dict(display_tree)
         except Exception as e:
              print(f"Error calculating metrics or tree structure: {e}")
              # Continue without metrics/structure if calculation fails

    # --- Update Predictions using the newly trained model ---
    print(f"Generating predictions using {'boosted' if is_boosted else 'single'} model...")
    results = []
    # Use the full dataframe 'df' for generating results as before
    for idx, row in df.iterrows():
        # Pass the row (which is a pandas Series) to predict
        # The predict function now uses the global is_boosted flag internally
        pred = predict(row, use_boosting=is_boosted)
        
        # Map numerical back to categorical for display
        jenis_kelamin_display = 'Laki-laki' if row['Jenis_Kelamin'] == 0 else 'Perempuan'
        nyeri_dada_display = 'Ya' if row['Nyeri_Dada'] == 1 else 'Tidak'
        sesak_napas_display = 'Ya' if row['Sesak_Napas'] == 1 else 'Tidak'
        kelelahan_display = 'Ya' if row['Kelelahan'] == 1 else 'Tidak'

        result_dict = {
            'Usia': row['Usia'],
            'Jenis_Kelamin': jenis_kelamin_display,
            'Sistolik': row['Sistolik'],
            'Diastolik': row['Diastolik'],
            'Kolesterol': row['Kolesterol'],
            'Gula_Darah': row['Gula_Darah'],
            'Nyeri_Dada': nyeri_dada_display,
            'Sesak_Napas': sesak_napas_display,
            'Kelelahan': kelelahan_display,
            'Denyut_Jantung': row['Denyut_Jantung'],
            'Aktual': row[target],
            'Prediksi': pred if pred is not None else "N/A" # Handle None prediction
        }
        results.append(result_dict)
    print("Prediction generation complete.")

    # Calculate feature importance (placeholder - replace with actual calculation)
    feature_importance = {}
    for feature in features:
        feature_importance[feature] = 0.1  # Assign a default value (replace with actual importance)

    # Return success response
    return jsonify({
        'status': 'success',
        'message': f"Model trained successfully ({'Boosting enabled' if is_boosted else 'Single tree'}).",
        'metrics': tree_metrics,
        'tree_structure': tree_structure, # Send structure for visualization
        'feature_importance': feature_importance
    })

@app.route('/api/test_data')
def get_test_data():
    page = int(request.args.get('page', 1))
    per_page = 10
    start = (page - 1) * per_page
    end = start + per_page
    
    # Convert test_data to list of dictionaries
    test_list = test_data.to_dict('records')
    
    # Calculate statistics for each disease class
    total_records = len(test_list)
    disease_counts = {}
    for record in test_list:
        disease = record['Penyakit_Jantung']
        disease_counts[disease] = disease_counts.get(disease, 0) + 1
    
    # Calculate percentages for each disease
    disease_stats = [
        {
            'disease': disease,
            'count': count,
            'percent': round((count / total_records * 100) if total_records > 0 else 0, 2)
        }
        for disease, count in disease_counts.items()
    ]
    
    # Format data for frontend
    formatted_data = [
        {
            'id': idx + 1,  # Generate ID dynamically
            'usia': record['Usia'],
            'jenis_kelamin': 'Laki-laki' if record['Jenis_Kelamin'] == 0 else 'Perempuan',
            'sistolik': record['Sistolik'],
            'diastolik': record['Diastolik'],
            'kolesterol': 'Tinggi' if record['Kolesterol'] > 200 else 'Normal',
            'gula_darah': 'Tinggi' if record['Gula_Darah'] > 120 else 'Normal',
            'nyeri_dada': 'Ya' if record['Nyeri_Dada'] == 1 else 'Tidak',
            'sesak_napas': 'Ya' if record['Sesak_Napas'] == 1 else 'Tidak',
            'kelelahan': 'Ya' if record['Kelelahan'] == 1 else 'Tidak',
            'denyut_jantung': record['Denyut_Jantung'],
            'penyakit_jantung': record['Penyakit_Jantung']
        }
        for idx, record in enumerate(test_list)
    ]
    
    # Paginate data
    paginated_data = formatted_data[start:end]
    
    # Return response
    return jsonify({
        'data': paginated_data,
        'total': total_records,
        'stats': {
            'total_records': total_records,
            'disease_stats': disease_stats
        },
        'page': page,
        'per_page': per_page
    })
# Export Endpoints
@app.route('/download_results')
def download_results():
    results_df = pd.DataFrame(results)
    output = BytesIO()
    results_df.to_excel(output, index=False)
    output.seek(0)
    return send_file(output, download_name='hasil_prediksi_manual.xlsx', as_attachment=True)

@app.route('/download_rules')
def download_rules():
    rules = []
    def extract_rules(node, rule_conditions=[]):
        if node.value is not None:
            rules.append({
                'Rule': " AND ".join(rule_conditions),
                'Prediction': node.value,
                'Confidence': 0.9,
                'Coverage': 0.3
            })
            return
        if node.feature and node.threshold:
            extract_rules(node.left, rule_conditions + [f"{node.feature} <= {node.threshold}"])
            extract_rules(node.right, rule_conditions + [f"{node.feature} > {node.threshold}"])
    extract_rules(tree)
    rules_df = pd.DataFrame(rules)
    output = BytesIO()
    rules_df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, download_name='decision_rules.csv', as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
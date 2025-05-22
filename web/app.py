import pandas as pd
import pandas as pd
import numpy as np
import math
import traceback
from collections import Counter
from flask import Flask, render_template, request, jsonify, send_file
from io import BytesIO
import json
import os
# Import the custom C5.0 implementation
from c5_algorithm import (build_c5_tree, predict_c5_instance, extract_rules_from_c5_tree, 
                         format_rules_for_display, calculate_accuracy, C5Booster, preprocess_data)
try:
    from sklearn.model_selection import train_test_split as sk_train_test_split
except ImportError:
    sk_train_test_split = None

app = Flask(__name__)

extracted_rules = []

# Load and clean the training data
df = pd.read_csv('data_latih_2.csv', sep=';')

# Fix duplicated columns issue in data_latih_2.csv
if len(df.columns) > 13 and 'Nama.1' in df.columns:
    print("Fixing duplicated columns in dataset...")
    original_columns = ['#', 'Nama', 'Usia', 'Jenis_Kelamin', 'Tekanan_Darah', 
                        'Kolesterol', 'Gula_Darah', 'Nyeri_Dada', 'Sesak_Napas', 
                        'Kelelahan', 'Denyut_Jantung', 'Penyakit_Jantung']
    df = df[original_columns]

# Handle specific issues in the data
if '#' in df.columns:
    df.drop('#', axis=1, inplace=True)  # Drop the index column
    
# Fix any whitespace in the Jenis_Kelamin column
if 'Jenis_Kelamin' in df.columns:
    df['Jenis_Kelamin'] = df['Jenis_Kelamin'].str.strip()

# Split the blood pressure into systolic and diastolic
df[['Sistolik', 'Diastolik']] = df['Tekanan_Darah'].str.split('/', expand=True).astype(int)
df = df.drop('Tekanan_Darah', axis=1)

# Encode categorical variables for the old model
df_encoded = df.copy()
df_encoded['Jenis_Kelamin'] = df_encoded['Jenis_Kelamin'].map({'L': 0, 'P': 1})
df_encoded['Nyeri_Dada'] = df_encoded['Nyeri_Dada'].map({'Tidak': 0, 'Ya': 1})
df_encoded['Sesak_Napas'] = df_encoded['Sesak_Napas'].map({'Tidak': 0, 'Ya': 1})
df_encoded['Kelelahan'] = df_encoded['Kelelahan'].map({'Tidak': 0, 'Ya': 1})

# Initial split removed, will be done in train_model route
train_data = pd.DataFrame() # Initialize as empty DataFrame
test_data = pd.DataFrame()  # Initialize as empty DataFrame

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
        return 0

    values = data[feature].unique()
    split_entropy = 0

    for value in values:
        subset_indices = data[feature] == value
        subset_weight = np.sum(sample_weights[subset_indices])

        if subset_weight > 0:
            probability = subset_weight / total_weight
            split_entropy -= probability * math.log2(probability)

    return split_entropy if split_entropy > 1e-9 else 1e-9

def gain_ratio(data, feature, target, sample_weights=None):
    ig = information_gain(data, feature, target, sample_weights)
    si = split_info(data, feature, sample_weights)
    return ig / si

class DecisionNode:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right

def build_tree(data, target, features, depth=0, max_depth=3, min_samples_split=2, sample_weights=None):
    if sample_weights is None:
        sample_weights = np.ones(len(data))

    unique_classes = data[target].unique()
    if len(unique_classes) == 1:
        return DecisionNode(value=unique_classes[0])

    if depth == max_depth or len(data) < min_samples_split or np.sum(sample_weights) < min_samples_split:
        weighted_counts = {}
        for i, label in enumerate(data[target]):
            weighted_counts[label] = weighted_counts.get(label, 0) + sample_weights[i]
        majority_class = max(weighted_counts, key=weighted_counts.get) if weighted_counts else None
        return DecisionNode(value=majority_class)

    if not features: # Check if the list is empty
        weighted_counts = {}
        for i, label in enumerate(data[target]):
            weighted_counts[label] = weighted_counts.get(label, 0) + sample_weights[i]
        majority_class = max(weighted_counts, key=weighted_counts.get) if weighted_counts else None
        return DecisionNode(value=majority_class)

    best_gain_ratio = -1
    best_feature = None
    best_threshold = None
    best_split_type = None

    current_features = list(features)

    for feature in current_features:
        unique_values = data[feature].unique()

        if len(unique_values) <= 1:
            continue

        if pd.api.types.is_numeric_dtype(data[feature]):
            sorted_values = sorted(unique_values)
            for i in range(len(sorted_values) - 1):
                threshold = (sorted_values[i] + sorted_values[i+1]) / 2

                temp_feature_name = f"{feature}_le_{threshold}"
                data[temp_feature_name] = (data[feature] <= threshold).astype(int)

                gain = gain_ratio(data, temp_feature_name, target, sample_weights)

                del data[temp_feature_name]

                if gain > best_gain_ratio:
                    best_gain_ratio = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_split_type = 'numerical'

        else:
            gain = gain_ratio(data, feature, target, sample_weights)
            if gain > best_gain_ratio:
                best_gain_ratio = gain
                best_feature = feature
                best_threshold = None
                best_split_type = 'categorical'

    if best_gain_ratio <= 0:
        weighted_counts = {}
        for i, label in enumerate(data[target]):
            weighted_counts[label] = weighted_counts.get(label, 0) + sample_weights[i]
        majority_class = max(weighted_counts, key=weighted_counts.get) if weighted_counts else None
        return DecisionNode(value=majority_class)

    remaining_features = [f for f in current_features if f != best_feature]

    if best_split_type == 'numerical':
        left_indices = data[best_feature] <= best_threshold
        right_indices = data[best_feature] > best_threshold

        left_data = data[left_indices]
        right_data = data[right_indices]
        left_weights = sample_weights[left_indices]
        right_weights = sample_weights[right_indices]

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
         dummies = pd.get_dummies(data[best_feature], prefix=best_feature, drop_first=True)
         best_dummy_feature = None
         best_dummy_gain = -1

         for dummy in dummies.columns:
             gain = gain_ratio(data.assign(**{dummy: dummies[dummy]}), dummy, target, sample_weights)
             if gain > best_dummy_gain:
                 best_dummy_gain = gain
                 best_dummy_feature = dummy

         if best_dummy_feature is None:
            weighted_counts = {}
            for i, label in enumerate(data[target]):
                weighted_counts[label] = weighted_counts.get(label, 0) + sample_weights[i]
            majority_class = max(weighted_counts, key=weighted_counts.get) if weighted_counts else None
            return DecisionNode(value=majority_class)

         temp_data = data.assign(**{best_dummy_feature: dummies[best_dummy_feature]})
         left_indices = temp_data[best_dummy_feature] == 0
         right_indices = temp_data[best_dummy_feature] == 1

         left_data = data[left_indices]
         right_data = data[right_indices]
         left_weights = sample_weights[left_indices]
         right_weights = sample_weights[right_indices]

         if len(left_data) == 0 or len(right_data) == 0 or np.sum(left_weights) == 0 or np.sum(right_weights) == 0:
             weighted_counts = {}
             for i, label in enumerate(data[target]):
                 weighted_counts[label] = weighted_counts.get(label, 0) + sample_weights[i]
             majority_class = max(weighted_counts, key=weighted_counts.get) if weighted_counts else None
             return DecisionNode(value=majority_class)

         left_child = build_tree(left_data, target, remaining_features, depth + 1, max_depth, min_samples_split, left_weights)
         right_child = build_tree(right_data, target, remaining_features, depth + 1, max_depth, min_samples_split, right_weights)

         return DecisionNode(feature=best_dummy_feature, threshold=0.5, left=left_child, right=right_child)

    else:
        weighted_counts = {}
        for i, label in enumerate(data[target]):
            weighted_counts[label] = weighted_counts.get(label, 0) + sample_weights[i]
        majority_class = max(weighted_counts, key=weighted_counts.get) if weighted_counts else None
        return DecisionNode(value=majority_class)

target = 'Penyakit_Jantung'
features = df.columns.drop(target) # Initial features based on df
tree = None # Initialize tree as None, build it inside train_model route

boosted_trees = []
boosted_alphas = []
is_boosted = False

def predict_single_tree(node, sample):
    if node is None:
        return None

    if node.value is not None:
        return node.value

    feature = node.feature

    if node.threshold is not None and feature in sample.index and pd.api.types.is_numeric_dtype(sample[feature]):
         try:
             sample_value = sample[feature]
             if sample_value <= node.threshold:
                 return predict_single_tree(node.left, sample)
             else:
                 return predict_single_tree(node.right, sample)
         except TypeError:
              return predict_single_tree(node.left, sample)

    elif node.threshold == 0.5 and feature in sample.index:
         return predict_single_tree(node.left, sample)

    else:
        return predict_single_tree(node.left, sample)

import os
import json

def get_majority_class(data, target_col):
    """Ambil kelas mayoritas dari data latih. Jika ada lebih dari satu kelas mayoritas, pilih secara acak."""
    if data is not None and not data.empty and target_col in data.columns:
        mode_vals = data[target_col].mode()
        if len(mode_vals) == 1:
            return mode_vals[0]
        elif len(mode_vals) > 1:
            import random
            return random.choice(mode_vals.tolist())
    return "N/A"

def predict_with_confidence(sample, use_rules=True, use_boosting=False):
    """
    Predict with confidence score. Returns a tuple of (prediction, confidence_score).
    Confidence score is between 0 and 1, with higher values indicating more confidence.
    """
    global extracted_rules, train_data, target, boosted_trees, boosted_alphas, is_boosted, tree

    # Fallback: majority class from training data
    fallback_majority = get_majority_class(train_data, target)

    if use_rules:
        # Initialize variables for tracking best rule matches
        best_full_match_rule = None
        best_partial_match_rule = None
        max_conditions_met = 0
        max_confidence = -1
        default_rule = None

        if not extracted_rules:
            return (fallback_majority, 0.5)  # Return with medium confidence

        # Sort rules by confidence for better prediction
        sorted_rules = sorted(extracted_rules, key=lambda x: x.get('confidence', 0), reverse=True)

        # First pass: find the default rule and rules with conditions
        rules_with_conditions = []
        for rule in sorted_rules:
            if 'machine_rule' not in rule or 'conditions' not in rule['machine_rule'] or 'prediction' not in rule['machine_rule']:
                continue
                
            conditions = rule['machine_rule']['conditions']
            prediction = rule['machine_rule']['prediction']
            confidence = rule.get('confidence', 0)
            
            if not conditions:
                # Rule without conditions is considered a default rule
                if default_rule is None or confidence > default_rule.get('confidence', 0):
                    default_rule = rule
            else:
                rules_with_conditions.append(rule)
        
        # Second pass: evaluate each rule with conditions
        for rule in rules_with_conditions:
            conditions = rule['machine_rule']['conditions']
            prediction = rule['machine_rule']['prediction']
            confidence = rule.get('confidence', 0)
            
            num_conditions_in_rule = len(conditions)
            conditions_met = 0
            all_met = True
            
            for condition_str in conditions:
                try:
                    parts = condition_str.split()
                    if len(parts) < 3:
                        all_met = False
                        continue
                        
                    feature = parts[0]
                    operator = parts[1]
                    value_from_rule = float(parts[2])
                    
                    if feature not in sample.index:
                        all_met = False
                        continue
                        
                    sample_value = sample[feature]
                    
                    condition_holds = False
                    if operator == '<=':
                        condition_holds = sample_value <= value_from_rule
                    elif operator == '>':
                        condition_holds = sample_value > value_from_rule
                    
                    if condition_holds:
                        conditions_met += 1
                    else:
                        all_met = False
                        
                except (IndexError, ValueError):
                    all_met = False
                    continue
            
            # Calculate match ratio for partial matches
            match_ratio = conditions_met / num_conditions_in_rule if num_conditions_in_rule > 0 else 0
            
            # Prioritize rules based on match type, number of conditions met, and confidence
            if all_met and num_conditions_in_rule > 0:
                if best_full_match_rule is None or confidence > max_confidence:
                    best_full_match_rule = rule
                    max_confidence = confidence
                    break  # Found a full match, no need to check other rules
            elif conditions_met > max_conditions_met:
                best_partial_match_rule = rule
                max_conditions_met = conditions_met
                max_confidence = confidence * match_ratio  # Adjust confidence for partial matches
            elif conditions_met == max_conditions_met and confidence > max_confidence:
                best_partial_match_rule = rule
                max_confidence = confidence * match_ratio  # Adjust confidence for partial matches
        
        # Return prediction based on the best matching rule
        if best_full_match_rule is not None:
            return (best_full_match_rule['machine_rule']['prediction'], best_full_match_rule.get('confidence', 0.5))
        elif best_partial_match_rule is not None and max_conditions_met > 0:
            # Calculate adjusted confidence based on partial match
            adjusted_confidence = best_partial_match_rule.get('confidence', 0.5) * (max_conditions_met / len(best_partial_match_rule['machine_rule']['conditions']))
            return (best_partial_match_rule['machine_rule']['prediction'], adjusted_confidence)
        elif default_rule is not None:
            return (default_rule['machine_rule']['prediction'], default_rule.get('confidence', 0.5))
        else:
            return (fallback_majority, 0.5)  # Return with medium confidence

    else:  # This is the tree-based prediction (single or boosted)
        current_model_is_boosted = is_boosted

        if use_boosting and current_model_is_boosted and boosted_trees and boosted_alphas:
            weighted_votes = {}
            total_weight = sum(boosted_alphas)

            for b_tree, alpha in zip(boosted_trees, boosted_alphas):
                pred = predict_single_tree(b_tree, sample)
                if pred is not None:
                    weighted_votes[pred] = weighted_votes.get(pred, 0) + alpha

            if not weighted_votes:
                return (None, 0.0)

            # Find the prediction with highest weighted vote
            best_pred = max(weighted_votes, key=weighted_votes.get)
            # Calculate confidence as the proportion of total weight
            confidence = weighted_votes[best_pred] / total_weight if total_weight > 0 else 0.5
            return (best_pred, confidence)

        elif not use_boosting and tree:
            # For single tree, use a simple confidence calculation based on tree structure
            prediction, node_depth = predict_single_tree_with_depth(tree, sample)
            if prediction is None:
                return (None, 0.0)
            
            # Calculate confidence based on tree depth: deeper nodes are typically more confident
            confidence = min(0.5 + (node_depth * 0.1), 0.95)  # Caps at 0.95 confidence
            return (prediction, confidence)
        
        elif use_boosting and not current_model_is_boosted and tree:
            prediction, node_depth = predict_single_tree_with_depth(tree, sample)
            if prediction is None:
                return (None, 0.0)
            
            # Calculate confidence based on tree depth
            confidence = min(0.5 + (node_depth * 0.1), 0.95)
            return (prediction, confidence)
        
        else:
            return (None, 0.0)

def predict_single_tree_with_depth(node, sample, depth=0):
    """Predicts using a single tree and returns both the prediction and the depth at which the prediction was made"""
    if node is None:
        return None, depth
        
    if node.value is not None:
        return node.value, depth
        
    feature = node.feature
    
    if feature not in sample.index:
        # If feature is missing in the sample, make a best guess
        left_result, left_depth = predict_single_tree_with_depth(node.left, sample, depth + 1) if hasattr(node, 'left') else (None, depth)
        right_result, right_depth = predict_single_tree_with_depth(node.right, sample, depth + 1) if hasattr(node, 'right') else (None, depth)
        
        if left_result is not None and right_result is not None:
            # Choose the deeper path which typically has more specific rules
            if left_depth >= right_depth:
                return left_result, left_depth
            else:
                return right_result, right_depth
        elif left_result is not None:
            return left_result, left_depth
        elif right_result is not None:
            return right_result, right_depth
        else:
            return None, depth
    
    if node.threshold is not None and pd.api.types.is_numeric_dtype(sample[feature]):
        if sample[feature] <= node.threshold:
            return predict_single_tree_with_depth(node.left, sample, depth + 1)
        else:
            return predict_single_tree_with_depth(node.right, sample, depth + 1)
    elif node.threshold == 0.5:  # For binary features, typically encoded as 0 or 1
        if sample[feature] <= 0.5:  # Treat as binary decision
            return predict_single_tree_with_depth(node.left, sample, depth + 1)
        else:
            return predict_single_tree_with_depth(node.right, sample, depth + 1)
    else:
        # For other cases (e.g., categorical features), make best guess
        left_result, left_depth = predict_single_tree_with_depth(node.left, sample, depth + 1) if hasattr(node, 'left') else (None, depth)
        right_result, right_depth = predict_single_tree_with_depth(node.right, sample, depth + 1) if hasattr(node, 'right') else (None, depth)
        
        if left_result is not None and right_result is not None:
            if left_depth >= right_depth:
                return left_result, left_depth
            else:
                return right_result, right_depth
        elif left_result is not None:
            return left_result, left_depth
        elif right_result is not None:
            return right_result, right_depth
        else:
            return None, depth

def predict(sample, use_rules=True, use_boosting=False):
    """
    Original predict function - now updated to use predict_with_confidence
    and return just the prediction for backward compatibility
    """
    result = predict_with_confidence(sample, use_rules, use_boosting)
    if isinstance(result, tuple) and len(result) == 2:
        return result[0]  # Return just the prediction
    return result  # In case of error, return the result as is

    # Fallback: majority class from training data
    fallback_majority = get_majority_class(train_data, target)

    if use_rules:
        # Initialize variables for tracking best rule matches
        best_full_match_rule = None
        best_partial_match_rule = None
        max_conditions_met = 0
        max_confidence = -1
        default_rule = None

        if not extracted_rules:
            return fallback_majority

        # Sort rules by confidence for better prediction
        sorted_rules = sorted(extracted_rules, key=lambda x: x.get('confidence', 0), reverse=True)

        # First pass: find the default rule and rules with conditions
        rules_with_conditions = []
        for rule in sorted_rules:
            if 'machine_rule' not in rule or 'conditions' not in rule['machine_rule'] or 'prediction' not in rule['machine_rule']:
                continue
                
            conditions = rule['machine_rule']['conditions']
            prediction = rule['machine_rule']['prediction']
            confidence = rule.get('confidence', 0)
            
            if not conditions:
                # Rule without conditions is considered a default rule
                if default_rule is None or confidence > default_rule.get('confidence', 0):
                    default_rule = rule
            else:
                rules_with_conditions.append(rule)
        
        # Second pass: evaluate each rule with conditions
        for rule in rules_with_conditions:
            conditions = rule['machine_rule']['conditions']
            prediction = rule['machine_rule']['prediction']
            confidence = rule.get('confidence', 0)
            
            num_conditions_in_rule = len(conditions)
            conditions_met = 0
            all_met = True
            
            for condition_str in conditions:
                try:
                    parts = condition_str.split()
                    if len(parts) < 3:
                        all_met = False
                        continue
                        
                    feature = parts[0]
                    operator = parts[1]
                    value_from_rule = float(parts[2])
                    
                    if feature not in sample.index:
                        all_met = False
                        continue
                        
                    sample_value = sample[feature]
                    
                    condition_holds = False
                    if operator == '<=':
                        condition_holds = sample_value <= value_from_rule
                    elif operator == '>':
                        condition_holds = sample_value > value_from_rule
                    
                    if condition_holds:
                        conditions_met += 1
                    else:
                        all_met = False
                        
                except (IndexError, ValueError):
                    all_met = False
                    continue
            
            # Prioritize rules based on match type, number of conditions met, and confidence
            if all_met and num_conditions_in_rule > 0:
                if best_full_match_rule is None or confidence > max_confidence:
                    best_full_match_rule = rule
                    max_confidence = confidence
                    break  # Found a full match, no need to check other rules
            elif conditions_met > max_conditions_met:
                best_partial_match_rule = rule
                max_conditions_met = conditions_met
            elif conditions_met == max_conditions_met and confidence > max_confidence:
                best_partial_match_rule = rule
                max_confidence = confidence
        
        # Return prediction based on the best matching rule
        if best_full_match_rule is not None:
            return best_full_match_rule['machine_rule']['prediction']
        elif best_partial_match_rule is not None and max_conditions_met > 0:
            return best_partial_match_rule['machine_rule']['prediction']
        elif default_rule is not None:
            return default_rule['machine_rule']['prediction']
        else:
            return fallback_majority

    else: # This is the tree-based prediction (single or boosted)
        global boosted_trees, boosted_alphas, is_boosted, tree

        current_model_is_boosted = is_boosted

        if use_boosting and current_model_is_boosted and boosted_trees and boosted_alphas:
            weighted_votes = {}

            for b_tree, alpha in zip(boosted_trees, boosted_alphas):
                pred = predict_single_tree(b_tree, sample)

                if pred is not None:
                     weighted_votes[pred] = weighted_votes.get(pred, 0) + alpha

            if not weighted_votes:
                 return None

            return max(weighted_votes, key=weighted_votes.get)

        elif not use_boosting and tree:
             return predict_single_tree(tree, sample)
        elif use_boosting and not current_model_is_boosted and tree:
             return predict_single_tree(tree, sample)
        else:
             return None

@app.route('/')
def dashboard():
    return render_template('index.html', page='dashboard')

@app.route('/training_data')
def training_data_page():
    return render_template('training_data.html', page='training_data')

@app.route('/test_data')
def test_data_page():
    return render_template('test_data.html', page='test_data')

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

@app.route('/api/all_predictions')
def get_all_predictions():
    global df, tree, target
    results = []
    for idx, row in df.iterrows():
        # Get prediction with confidence
        pred_result = predict_with_confidence(row, use_rules=True, use_boosting=is_boosted)
        
        if isinstance(pred_result, tuple) and len(pred_result) == 2:
            pred, confidence = pred_result
        else:
            pred = pred_result
            confidence = 0.0
            
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
            'Prediksi': pred if pred is not None else "N/A",
            'Confidence': round(confidence, 2) if confidence is not None else 0.0
        }
        results.append(result_dict)
    return jsonify(results)

@app.route('/api/predictions')
def get_predictions():
    response = requests.get(request.url_root + 'api/all_predictions')
    results = response.json()
    results_df = pd.DataFrame(results)

    total_predictions = len(results_df)

    correct_predictions = len(results_df[results_df['Aktual'] == results_df['Prediksi']])

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

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

    tp = confusion_matrix['Ya']['Ya']
    fp = confusion_matrix['Tidak Ada']['Ya']
    fn = confusion_matrix['Ya']['Tidak Ada']
    tn = confusion_matrix['Tidak Ada']['Tidak Ada']

    precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    response = {
        'total': total_predictions,
        'correct': correct_predictions,
        'accuracy': accuracy,
        'confusionMatrix': confusion_matrix,
        'classAccuracy': class_accuracy,
        'precision': precision,
        'recall': recall,
        'f1Score': f1_score,
        'data': results
    }

    return jsonify(response)

@app.route('/api/test_data')
def get_test_data():
    page = int(request.args.get('page', 1))
    per_page = 10
    start = (page - 1) * per_page
    end = start + per_page

    df_test = pd.read_csv('static/test_data_original.csv')

    # Convert DataFrame to list of dictionaries
    test_list = df_test.to_dict('records')

    # Explicitly handle NaN/None values in the list of dictionaries
    cleaned_test_list = []
    for record in test_list:
        cleaned_record = {}
        for key, value in record.items():
            # Replace pandas NaN and Python None with empty string
            if pd.isna(value) or value is None:
                cleaned_record[key] = ''
            else:
                cleaned_record[key] = value
        cleaned_test_list.append(cleaned_record)

    total_records = len(cleaned_test_list)
    disease_counts = {}
    for record in cleaned_test_list:
        disease = record.get('Penyakit_Jantung', 'Unknown') # Use .get with default for safety
        disease_counts[disease] = disease_counts.get(disease, 0) + 1

    disease_stats = [
        {
            'disease': disease,
            'count': count,
            'percent': round((count / total_records * 100) if total_records > 0 else 0, 2)
        }
        for disease, count in disease_counts.items()
    ]

    # The frontend expects specific keys, ensure they are present even if empty
    formatted_data = []
    for idx, record in enumerate(cleaned_test_list):
         formatted_record = {
            'id': idx + 1,
            'Nama': record.get('Nama', ''),
            'Usia': record.get('Usia', ''),
            'Jenis_Kelamin': record.get('Jenis_Kelamin', ''),
            'Sistolik': record.get('Sistolik', ''),
            'Diastolik': record.get('Diastolik', ''),
            'Kolesterol': record.get('Kolesterol', ''),
            'Gula_Darah': record.get('Gula_Darah', ''),
            'Nyeri_Dada': record.get('Nyeri_Dada', ''),
            'Sesak_Napas': record.get('Sesak_Napas', ''),
            'Kelelahan': record.get('Kelelahan', ''),
            'Denyut_Jantung': record.get('Denyut_Jantung', ''),
            'Penyakit_Jantung': record.get('Penyakit_Jantung', '')
         }
         formatted_data.append(formatted_record)


    paginated_data = formatted_data[start:end]

    return jsonify({
        'data': paginated_data,
        'total': len(cleaned_test_list),
        'stats': {
            'total_records': total_records,
            'disease_stats': disease_stats
        },
        'page': page,
        'per_page': per_page
    })

@app.route('/api/training_data')
def get_training_data():
    page = int(request.args.get('page', 1))
    per_page = 10
    start = (page - 1) * per_page
    end = start + per_page

    df = pd.read_csv('data_latih_2.csv', sep=';')

    df[['Sistolik', 'Diastolik']] = df['Tekanan_Darah'].str.split('/', expand=True).astype(int)
    df = df.drop('Tekanan_Darah', axis=1)
    df['Jenis_Kelamin'] = df['Jenis_Kelamin'].map({'L': 0, 'P': 1})
    df['Nyeri_Dada'] = df['Nyeri_Dada'].map({'Tidak': 0, 'Ya': 1})
    df['Sesak_Napas'] = df['Sesak_Napas'].map({'Tidak': 0, 'Ya': 1})
    df['Kelelahan'] = df['Kelelahan'].map({'Tidak': 0, 'Ya': 1})

    train_list = df.to_dict('records')

    # Map categorical features to numerical values
    for record in train_list:
        record['Jenis_Kelamin'] = 0 if record['Jenis_Kelamin'] == 'L' or record['Jenis_Kelamin'] == 'Laki-laki' else 1
        record['Nyeri_Dada'] = 1 if record['Nyeri_Dada'] == 'Ya' else 0
        record['Sesak_Napas'] = 1 if record['Sesak_Napas'] == 'Ya' else 0
        record['Kelelahan'] = 1 if record['Kelelahan'] == 'Ya' else 0

    total_records = len(train_list)
    disease_counts = {}
    for record in train_list:
        disease = record['Penyakit_Jantung']
        disease_counts[disease] = disease_counts.get(disease, 0) + 1

    disease_stats = [
        {
            'disease': disease,
            'count': count,
            'percent': round((count / total_records * 100) if total_records > 0 else 0, 2)
        }
        for disease, count in disease_counts.items()
    ]

    formatted_data = [
        {
            'id': idx + 1,
            'Nama': record['Nama'],
            'Usia': record['Usia'],
            'Jenis_Kelamin': 'Laki-laki' if record['Jenis_Kelamin'] == 0 else 'Perempuan',
            'Sistolik': record['Sistolik'],
            'Diastolik': record['Diastolik'],
            'Tekanan_Darah': f"{record['Sistolik']}/{record['Diastolik']}",
            'Kolesterol': record['Kolesterol'],
            'Gula_Darah': record['Gula_Darah'],
            'Nyeri_Dada': 'Ya' if record['Nyeri_Dada'] == 1 else 'Tidak',
            'Sesak_Napas': 'Ya' if record['Sesak_Napas'] == 1 else 'Tidak',
            'Kelelahan': 'Ya' if record['Kelelahan'] == 1 else 'Tidak',
            'Denyut_Jantung': record['Denyut_Jantung'],
            'Penyakit_Jantung': record['Penyakit_Jantung']
        }
        for idx, record in enumerate(train_list)
    ]

    paginated_data = formatted_data[start:end]

    return jsonify({
        'data': paginated_data,
        'total': len(train_list),
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

    response = requests.get(request.url_root + 'api/all_predictions')
    results = response.json()
    results_df = pd.DataFrame(results)

    total_predictions = len(results_df)
    correct_predictions = len(results_df[results_df['Aktual'] == results_df['Prediksi']])
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0

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

    paginated_results = results[start:end]

    formatted_data = [
        {
            'usia': item['Usia'],
            'jenis_kelamin': item['Jenis_Kelamin'],
            'sistolik': item['Sistolik'],
            'diastolik': item['Diastolik'],
            'kolesterol': item['Kolesterol'],
            'gula_darah': item['Gula_Darah'],
            'nyeri_dada': item['Nyeri_Dada'],
            'sesak_napas': item['Sesak_Napas'],
            'kelelahan': item['Kelelahan'],
            'denyut_jantung': item['Denyut_Jantung'],
            'aktual': item['Aktual'],
            'prediksi': item['Prediksi'],
            'confidence': item.get('Confidence', 0.0),  # Include confidence value
            'status': 'Correct' if item['Aktual'] == item['Prediksi'] else 'Incorrect'
        }
        for item in paginated_results
    ]

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
    global tree, is_boosted, boosted_trees

    tree_to_visualize = tree if not is_boosted else (boosted_trees[0] if boosted_trees else None)

    if not tree_to_visualize:
        return jsonify({'error': 'No trained model available to visualize'}), 400

    def tree_to_dict(node):
        if node is None: return None
        if node.value is not None:
            return {'name': f"Predict: {node.value}", 'value': 'Leaf', 'class_value': node.value, 'samples': 0, 'impurity': 0}

        node_name = f"{node.feature}"
        if node.threshold is not None:
             threshold_str = f"{node.threshold:.2f}" if isinstance(node.threshold, float) else str(node.threshold)
             if node.threshold == 0.5 and '_' in node.feature:
                 parts = node.feature.split('_')
                 if len(parts) > 1:
                      original_feature = parts[0]
                      category_value = '_'.join(parts[1:])
                      node_name = f"{original_feature} is {category_value}"
                 else:
                      node_name = f"{node.feature} <= {threshold_str}"
             else:
                 node_name = f"{node.feature} <= {threshold_str}"

        left_child_dict = tree_to_dict(node.left) if hasattr(node, 'left') else None
        right_child_dict = tree_to_dict(node.right) if hasattr(node, 'right') else None

        valid_children = [child for child in [left_child_dict, right_child_dict] if child is not None]

        return {
            'name': node_name,
            'value': 'Decision',
            'children': valid_children
        }

    try:
        tree_structure_json = tree_to_dict(tree_to_visualize)
        if tree_structure_json is None:
             return jsonify({'error': 'Model structure is empty or invalid'}), 500
        return jsonify(tree_structure_json)
    except Exception as e:
        print(f"Error generating tree visualization JSON: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to generate tree structure: {str(e)}'}), 500

@app.route('/api/rules')
def get_rules():
    try:
        with open('static/rules.json', 'r') as f:
            rules = json.load(f)
    except FileNotFoundError:
        return jsonify({'data': [], 'total': 0, 'stats': {'total_rules': 0, 'prediction_counts': {}, 'avg_conditions': 0}, 'page': 1, 'per_page': 10}), 404
    except json.JSONDecodeError:
        return jsonify({'error': 'Error decoding rules.json'}), 500

    page = int(request.args.get('page', 1))
    per_page = 10
    start = (page - 1) * per_page
    end = start + per_page
    paginated_rules = rules[start:end]

    total_rules = len(rules)
    prediction_counts = {}
    total_conditions = 0
    for rule in rules:
        if 'machine_rule' in rule and 'prediction' in rule['machine_rule']:
            pred = rule['machine_rule']['prediction']
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
        if 'machine_rule' in rule and 'conditions' in rule['machine_rule']:
             total_conditions += len(rule['machine_rule']['conditions'])

    avg_conditions = round(total_conditions / total_rules, 1) if total_rules > 0 else 0

    stats = {
        'total_rules': total_rules,
        'prediction_counts': prediction_counts,
        'avg_conditions': avg_conditions
    }

    return jsonify({
        'data': paginated_rules,
        'total': total_rules,
        'stats': stats,
        'page': page,
        'per_page': per_page
    })

@app.route('/upload_training_data', methods=['POST'])
def upload_training_data():
    file = request.files['file']
    if file and file.filename.endswith('.csv'):
        global train_data
        try:
            train_data = pd.read_csv(file, sep=';')
            train_data[['Sistolik', 'Diastolik']] = train_data['Tekanan_Darah'].str.split('/', expand=True).astype(int)
            train_data = train_data.drop('Tekanan_Darah', axis=1)
            train_data['Jenis_Kelamin'] = train_data['Jenis_Kelamin'].map({'L': 0, 'P': 1})
            train_data['Nyeri_Dada'] = train_data['Nyeri_Dada'].map({'Tidak': 0, 'Ya': 1})
            train_data['Sesak_Napas'] = train_data['Sesak_Napas'].map({'Tidak': 0, 'Ya': 1})
            train_data['Kelelahan'] = train_data['Kelelahan'].map({'Tidak': 0, 'Ya': 1})
            if 'Nama' not in train_data.columns:
                train_data['Nama'] = [f'Pasien {i+1}' for i in range(len(train_data))]
        except Exception as e:
            return jsonify({'status': 'error', 'message': f'Error processing file: {str(e)}'}), 500
        return jsonify({'status': 'success', 'message': 'Data uploaded successfully'})
    return jsonify({'status': 'error', 'message': 'Invalid file'})

@app.route('/upload_test_data', methods=['POST'])
def upload_test_data():
    global test_data
    file = request.files.get('file')

    if not file:
        return jsonify({'status': 'error', 'message': 'No file uploaded'}), 400
    if not file.filename.endswith('.csv'):
        return jsonify({'status': 'error', 'message': 'File must be a CSV'}), 400

    try:
        new_test_data = pd.read_csv(file)

        required_columns = ['Usia', 'Jenis_Kelamin', 'Tekanan_Darah', 'Kolesterol', 'Gula_Darah',
                           'Nyeri_Dada', 'Sesak_Napas', 'Kelelahan', 'Denyut_Jantung', 'Penyakit_Jantung']
        if not all(col in new_test_data.columns for col in required_columns):
            return jsonify({'status': 'error', 'message': 'Missing required columns in CSV'}), 400

        new_test_data[['Sistolik', 'Diastolik']] = new_test_data['Tekanan_Darah'].str.split('/', expand=True).astype(int)
        new_test_data = new_test_data.drop('Tekanan_Darah', axis=1)

        new_test_data['Jenis_Kelamin'] = new_test_data['Jenis_Kelamin'].map({'L': 0, 'P': 1, 'Laki-laki': 0, 'Perempuan': 1}).fillna(new_test_data['Jenis_Kelamin'])
        new_test_data['Nyeri_Dada'] = new_test_data['Nyeri_Dada'].map({'Tidak': 0, 'Ya': 1, 0: 0, 1: 1}).fillna(new_test_data['Nyeri_Dada'])
        new_test_data['Sesak_Napas'] = new_test_data['Sesak_Napas'].map({'Tidak': 0, 'Ya': 1, 0: 0, 1: 1}).fillna(new_test_data['Sesak_Napas'])
        new_test_data['Kelelahan'] = new_test_data['Kelelahan'].map({'Tidak': 0, 'Ya': 1, 0: 0, 1: 1}).fillna(new_test_data['Kelelahan'])

        if new_test_data[['Jenis_Kelamin', 'Nyeri_Dada', 'Sesak_Napas', 'Kelelahan']].isnull().any().any():
            return jsonify({'status': 'error', 'message': 'Invalid values in categorical columns'}), 400

        test_data = new_test_data

        return jsonify({
            'status': 'success',
            'message': f'Successfully uploaded {len(new_test_data)} records',
            'records': len(new_test_data)
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error processing file: {str(e)}'}), 500
@app.route('/train_model', methods=['POST'])
def train_model():
    global tree, results, train_data, test_data, target, features, df # Added test_data and df
    global boosted_trees, boosted_alphas, is_boosted, extracted_rules

    # --- 1. Get Parameters and Split Data ---
    try:
        max_depth = int(request.form.get('max_depth', 3))
        min_samples_split = int(request.form.get('min_samples_split', 2))
        boosting = request.form.get('boosting', 'false').lower() == 'true'
        n_estimators = int(request.form.get('n_estimators', 10)) if boosting else 1
        train_test_split_ratio = float(request.form.get('train_test_split', 0.8))
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid training parameters (must be numbers)'}), 400

    # Perform data split with stratification to maintain class distributions
    if df.empty:
        return jsonify({'status': 'error', 'message': 'Original dataset (df) is empty. Cannot split.'}), 400
    
    # Try to stratify if possible to maintain class distributions
    try:
        if sk_train_test_split is not None:
            train_idx, test_idx = sk_train_test_split(
                df.index, 
                test_size=(1-train_test_split_ratio),
                random_state=42,
                stratify=df[target] if len(df[target].unique()) > 1 else None
            )
            train_data = df.loc[train_idx]
            test_data = df.loc[test_idx]
            print("Using stratified train-test split")
        else:
            # Fallback to pandas sampling if sklearn is not available
            train_data = df.sample(frac=train_test_split_ratio, random_state=42)
            test_data = df.drop(train_data.index)
            print("Using random train-test split (stratified split not available)")
    except Exception as e:
        # Fallback to pandas sampling if stratification fails
        train_data = df.sample(frac=train_test_split_ratio, random_state=42)
        test_data = df.drop(train_data.index)
        print(f"Using random train-test split (stratified split failed: {str(e)})")
    
    print(f"Data split: {len(train_data)} training samples, {len(test_data)} testing samples (Ratio: {train_test_split_ratio*100:.0f}%)")

    # Save original test data to CSV
    os.makedirs('static', exist_ok=True)
    test_data.to_csv('static/test_data_original.csv', index=False)

    # --- 2. Validate Training Data and Features ---
    if train_data.empty:
        return jsonify({'status': 'error', 'message': 'Training data is empty after split'}), 400
    if target not in train_data.columns:
         return jsonify({'status': 'error', 'message': f'Target column "{target}" not found in training data'}), 400

    # Parse feature types for the C5.0 algorithm
    feature_types = {}
    for col in train_data.columns:
        if col == target or col == 'Nama':
            continue
        
        if col in ['Jenis_Kelamin', 'Nyeri_Dada', 'Sesak_Napas', 'Kelelahan']:
            feature_types[col] = 'categorical'
        elif pd.api.types.is_numeric_dtype(train_data[col]):
            feature_types[col] = 'numeric'
        else:
            feature_types[col] = 'categorical'
    
    # Use our original non-encoded data for training
    c5_train_data = train_data.copy()
    
    # --- 3. Train Model (Boosting or Single Tree) ---
    boosted_trees = []
    boosted_alphas = []
    is_boosted = False # Reset boosting status for each training run
    
    # Prepare data for C5.0 algorithm
    feature_columns = [col for col in c5_train_data.columns if col != target and col != 'Nama']
    X_train = c5_train_data[feature_columns]
    y_train = c5_train_data[target]

    if boosting:
        print(f"Starting C5.0 Boosting with {n_estimators} estimators...")
        is_boosted = True
        
        booster = C5Booster(
            max_depth=max_depth, 
            min_samples_split=min_samples_split,
            n_estimators=n_estimators
        )
        
        booster.fit(X_train, y_train, feature_types=feature_types)
        boosted_trees = booster.trees
        boosted_alphas = booster.weights
        
        if not boosted_trees:
             return jsonify({'status': 'error', 'message': 'C5.0 Boosting failed: No trees were built.'}), 500
        tree = boosted_trees[0]  # Use the first tree for rule extraction
        print("C5.0 Boosting complete.")
    else:
        print("Starting Single C5.0 Tree Training...")
        is_boosted = False
        
        # Get feature list excluding target and Nama
        features = [col for col in c5_train_data.columns if col != target and col != 'Nama']
        
        # Build a C5.0 decision tree
        tree = build_c5_tree(
            X_train,
            y_train, 
            features, 
            feature_types,
            max_depth=max_depth,
            min_samples_split=min_samples_split
        )
        
        boosted_trees = [tree]
        boosted_alphas = [1.0]
        print("Single C5.0 tree training complete.")

    global extracted_rules
    extracted_rules = extract_rules(tree, path_data=train_data)

    # Save extracted rules to /static/rules.json
    os.makedirs('static', exist_ok=True)
    with open('static/rules.json', 'w') as f:
        json.dump(extracted_rules, f, indent=4)
    
    # --- 5. Generate predictions for test data ---
        df_for_csv_output = test_data.copy()        # Process test data the same way as training data
        categorical_features = [col for col in df_for_csv_output.columns if 
                              col in ['Jenis_Kelamin', 'Nyeri_Dada', 'Sesak_Napas', 'Kelelahan']]
        temp_test_processed = pd.get_dummies(df_for_csv_output, columns=categorical_features, drop_first=True)
    
        # Add missing columns that might be in the training features but not in test
        for feature_col in features:
            if feature_col not in temp_test_processed.columns:
                temp_test_processed[feature_col] = 0
    
        # Make predictions using both rule-based and direct tree-based methods
        rule_predictions = []
        tree_predictions = []
    
        for _, row in temp_test_processed.iterrows():
            rule_pred = predict(row, use_rules=True, use_boosting=False)
            tree_pred = predict(row, use_rules=False, use_boosting=is_boosted)
            rule_predictions.append(rule_pred if rule_pred is not None else "N/A")
            tree_predictions.append(tree_pred if tree_pred is not None else "N/A")
    
        # Use the better of the two prediction methods
        df_for_csv_output['Prediksi_Rule'] = rule_predictions
        df_for_csv_output['Prediksi_Tree'] = tree_predictions
    
        # Compare prediction accuracy for both methods on the test set
        rule_correct = sum(1 for i, row in df_for_csv_output.iterrows() 
                           if row['Prediksi_Rule'] == row[target] and row['Prediksi_Rule'] != "N/A")
        tree_correct = sum(1 for i, row in df_for_csv_output.iterrows() 
                           if row['Prediksi_Tree'] == row[target] and row['Prediksi_Tree'] != "N/A")
                      
        rule_accuracy = rule_correct / len(df_for_csv_output) if len(df_for_csv_output) > 0 else 0
        tree_accuracy = tree_correct / len(df_for_csv_output) if len(df_for_csv_output) > 0 else 0
    
        print(f"Rule-based accuracy: {rule_accuracy:.4f}, Tree-based accuracy: {tree_accuracy:.4f}")
    
        # Use the better method for the final prediction
        if rule_accuracy >= tree_accuracy:
            df_for_csv_output['Prediksi'] = df_for_csv_output['Prediksi_Rule']
            print("Using rule-based predictions as they're more accurate")
        else:
            df_for_csv_output['Prediksi'] = df_for_csv_output['Prediksi_Tree'] 
            print("Using tree-based predictions as they're more accurate")
    
        # Ensure N/A values are filled appropriately
        df_for_csv_output['Prediksi'] = df_for_csv_output['Prediksi'].fillna("N/A")
    
        # Drop the intermediate columns
        df_for_csv_output.drop(['Prediksi_Rule', 'Prediksi_Tree'], axis=1, inplace=True)
    
        # Save the predictions
        df_for_csv_output.to_csv('static/hasil_prediksi.csv', index=False)
    
        if tree is None and not is_boosted:
            return jsonify({'status': 'error', 'message': 'Model training failed: Could not build tree.'}), 500
        elif is_boosted and not boosted_trees:
            return jsonify({'status': 'error', 'message': 'Model training failed: Could not build boosted ensemble.'}), 500

    display_tree = tree
    tree_metrics = {'depth': 0, 'nodes': 0, 'leaves': 0}
    tree_structure = {}
    if display_tree:
         try:
             def calculate_tree_depth(node):
                 if node is None or node.value is not None: return 0
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

             def tree_to_dict(node):
                 if node is None: return None
                 if node.value is not None:
                     return {'name': f"Predict: {node.value}", 'value': 'Leaf'}

                 node_name = f"{node.feature}"
                 if node.threshold is not None:
                      threshold_str = f"{node.threshold:.2f}" if isinstance(node.threshold, float) else str(node.threshold)
                      node_name += f" <= {threshold_str}"

                 left_child_dict = tree_to_dict(node.left) if hasattr(node, 'left') else None
                 right_child_dict = tree_to_dict(node.right) if hasattr(node, 'right') else None

                 return {
                     'name': node_name,
                     'value': 'Decision',
                     'children': [child for child in [left_child_dict, right_child_dict] if child is not None]                 }
             tree_structure = tree_to_dict(display_tree)
         except Exception as e:
              print(f"Error calculating metrics or tree structure: {e}")
    print(f"Generating predictions using {'boosted' if is_boosted else 'single'} model...")
    results = []
    for idx, row in df.iterrows():
        # Get prediction and confidence
        pred_result = predict_with_confidence(row, use_boosting=is_boosted)
        
        if isinstance(pred_result, tuple) and len(pred_result) == 2:
            pred, confidence = pred_result
        else:
            pred = pred_result
            confidence = 0.0

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
            'Prediksi': pred if pred is not None else "N/A",
            'Confidence': round(confidence, 2) if confidence is not None else 0.0
         }
        results.append(result_dict)
    print("Prediction generation complete.")    # Calculate feature importance
    from feature_importance import calculate_feature_importance
    feature_importance = calculate_feature_importance(tree, features)

    return jsonify({
        'status': 'success',
        'message': f"Model trained successfully ({'Boosting enabled' if is_boosted else 'Single tree'}).",
        'metrics': tree_metrics,
        'tree_structure': tree_structure,
        'feature_importance': feature_importance,
        'rule_accuracy': round(rule_accuracy * 100, 2),
        'tree_accuracy': round(tree_accuracy * 100, 2)
    })

@app.route('/download_results')
def download_results():
    results_df = pd.DataFrame(results)
    output = BytesIO()
    results_df.to_excel(output, index=False)
    output.seek(0)
    return send_file(output, download_name='hasil_prediksi_manual.xlsx', as_attachment=True)

@app.route('/download_rules')
def download_rules():
    rules_df = pd.DataFrame(extracted_rules)
    output = BytesIO()
    rules_df.to_csv(output, index=False)
    output.seek(0)
    return send_file(output, download_name='decision_rules.csv', as_attachment=True)

def extract_rules(node, rule_conditions=[], path_data=None):
    global train_data, target
    
    rules = []
    
    # Use the full train_data as default if path_data is None
    if path_data is None:
        path_data = train_data.copy()
    
    if node.value is not None:
        # Calculate rule confidence and coverage only if we have valid path data
        if not path_data.empty:
            correct = len(path_data[path_data[target] == node.value])
            total = len(path_data)
            confidence = correct / total if total > 0 else 0
            coverage = total / len(train_data) if len(train_data) > 0 else 0
            
            # Calculate class distribution for this rule
            class_distribution = path_data[target].value_counts().to_dict()
            class_distribution_str = ", ".join([f"{cls}: {count} ({count/total*100:.1f}%)" 
                                              for cls, count in class_distribution.items()])
        else:
            confidence = 0
            coverage = 0
            class_distribution = {}
            class_distribution_str = "No data"

        machine_rule = {
            'conditions': rule_conditions,
            'prediction': node.value,
            'class_distribution': class_distribution_str
        }

        display_rule = "JIKA " + " DAN ".join(
            [cond.replace(" <= ", " LEBIH KECIL SAMA DENGAN ").replace(" > ", " LEBIH BESAR DARI ") for cond in rule_conditions]
        ) + f" MAKA PREDICT {node.value}"

        rules.append({
            'display_rule': display_rule,
            'machine_rule': machine_rule,
            'confidence': round(confidence, 2),
            'coverage': round(coverage, 2),
            'class_distribution': class_distribution_str
        })
        return rules

    if node.feature and node.threshold is not None:
        # Create proper path data subsets for left and right branches
        left_mask = path_data[node.feature] <= node.threshold
        right_mask = path_data[node.feature] > node.threshold
        
        left_path_data = path_data[left_mask].copy() if not path_data.empty else pd.DataFrame()
        right_path_data = path_data[right_mask].copy() if not path_data.empty else pd.DataFrame()
        
        condition = f"{node.feature} <= {node.threshold}"
        left_rules = extract_rules(node.left, rule_conditions + [condition], left_path_data)
        right_rules = extract_rules(node.right, rule_conditions + [condition.replace('<=', '>')], right_path_data)
        return left_rules + right_rules
    
    return rules

import requests # type: ignore

if __name__ == '__main__':
    app.run(debug=True)

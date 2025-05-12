import pandas as pd
import numpy as np
import math
from collections import Counter
from flask import Flask, render_template, request, jsonify, send_file
from io import BytesIO
import json
import os

app = Flask(__name__)

extracted_rules = []

df = pd.read_csv('data_latih_2.csv', sep=';')

df[['Sistolik', 'Diastolik']] = df['Tekanan_Darah'].str.split('/', expand=True).astype(int)
df = df.drop('Tekanan_Darah', axis=1)
df['Jenis_Kelamin'] = df['Jenis_Kelamin'].map({'L': 0, 'P': 1})
df['Nyeri_Dada'] = df['Nyeri_Dada'].map({'Tidak': 0, 'Ya': 1})
df['Sesak_Napas'] = df['Sesak_Napas'].map({'Tidak': 0, 'Ya': 1})
df['Kelelahan'] = df['Kelelahan'].map({'Tidak': 0, 'Ya': 1})

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
# tree = build_tree(train_data, target, features, max_depth=3) # Removed initial build

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

def predict(sample, use_rules=True, use_boosting=False):
    global extracted_rules

    if use_rules:
        best_full_match_prediction = None
        best_partial_match_prediction = "N/A"  # Default if no rule matches at all
        max_conditions_partially_met = -1    # Max conditions met for rules *with conditions*
        
        default_rule_prediction = "N/A"      # For rules with no conditions

        if not extracted_rules: # Handle case of no rules
            return "N/A"

        rules_with_conditions = []
        # First pass to identify default rules and collect rules with conditions
        for rule in extracted_rules:
            if 'machine_rule' not in rule or 'conditions' not in rule['machine_rule'] or 'prediction' not in rule['machine_rule']:
                # print(f"Skipping malformed rule: {rule}")
                continue

            conditions = rule['machine_rule']['conditions']
            prediction = rule['machine_rule']['prediction']

            if not conditions:  # This is a default rule
                if default_rule_prediction == "N/A": # Take the first one encountered
                    default_rule_prediction = prediction
                continue
            
            rules_with_conditions.append(rule)

        # Second pass: Process rules with conditions
        for rule in rules_with_conditions:
            # At this point, rule structure is assumed to be valid and conditions list is non-empty
            conditions = rule['machine_rule']['conditions']
            prediction = rule['machine_rule']['prediction']
            
            num_conditions_in_rule = len(conditions) # Should be > 0
            current_conditions_met_count = 0
            all_conditions_met_for_this_rule = True # Assume true until a condition fails

            for condition_str in conditions:
                try:
                    parts = condition_str.split()
                    if len(parts) < 3: # Basic validation for condition string
                        all_conditions_met_for_this_rule = False
                        continue # Move to next condition

                    feature = parts[0]
                    operator = parts[1]
                    value_from_rule = float(parts[2])

                    if feature not in sample.index:
                        all_conditions_met_for_this_rule = False
                        # This condition is not met, don't increment count for it.
                        # Continue to the next condition in this rule to check if others might match.
                        continue

                    sample_value = sample[feature]
                    
                    condition_holds = False
                    if operator == '<=':
                        if sample_value <= value_from_rule:
                            condition_holds = True
                    elif operator == '>':
                        if sample_value > value_from_rule:
                            condition_holds = True
                    # Potentially add other operators like '==' if your rule system supports them
                    
                    if condition_holds:
                        current_conditions_met_count += 1
                    else:
                        all_conditions_met_for_this_rule = False
                
                except (IndexError, ValueError) as e:
                    # Malformed condition string or sample value issue for this specific condition
                    # print(f"Warning: Error processing condition '{condition_str}'. Error: {e}")
                    all_conditions_met_for_this_rule = False
                    # Continue to the next condition in this rule
                    continue
            
            if all_conditions_met_for_this_rule and num_conditions_in_rule > 0:
                best_full_match_prediction = prediction
                break # Found a full match among rules with conditions, prioritize this

            # If not a full match, check for best partial match from rules with conditions
            if current_conditions_met_count > max_conditions_partially_met:
                max_conditions_partially_met = current_conditions_met_count
                best_partial_match_prediction = prediction
        
        # Determine final prediction based on findings
        if best_full_match_prediction is not None:
            return best_full_match_prediction
        elif max_conditions_partially_met > 0: # A partial match from a rule with conditions was found
            return best_partial_match_prediction
        else: # No full or partial match from rules with conditions, try default rule
            return default_rule_prediction # This will be "N/A" if no default rule was found or no rules at all

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
        pred = predict(row, use_rules=True, use_boosting=is_boosted) # Use global is_boosted
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
    global boosted_trees, boosted_alphas, is_boosted

    # --- 1. Get Parameters and Split Data ---
    try:
        max_depth = int(request.form.get('max_depth', 3))
        min_samples_split = int(request.form.get('min_samples_split', 2))
        boosting = request.form.get('boosting', 'false').lower() == 'true'
        n_estimators = int(request.form.get('n_estimators', 10)) if boosting else 1
        train_test_split_ratio = float(request.form.get('train_test_split', 0.8)) # Get split ratio
    except ValueError:
         return jsonify({'status': 'error', 'message': 'Invalid training parameters (must be numbers)'}), 400

    # Perform data split based on the ratio from the request
    if df.empty:
         return jsonify({'status': 'error', 'message': 'Original dataset (df) is empty. Cannot split.'}), 400
    train_data = df.sample(frac=train_test_split_ratio, random_state=42)
    test_data = df.drop(train_data.index)
    print(f"Data split: {len(train_data)} training samples, {len(test_data)} testing samples (Ratio: {train_test_split_ratio*100:.0f}%)")

    # Save original test data to CSV
    os.makedirs('static', exist_ok=True)
    test_data.to_csv('static/test_data_original.csv', index=False)

    # --- 2. Validate Training Data and Features ---
    if train_data.empty:
        return jsonify({'status': 'error', 'message': 'Training data is empty after split'}), 400
    if target not in train_data.columns:
         return jsonify({'status': 'error', 'message': f'Target column "{target}" not found in training data'}), 400

    current_features = [col for col in train_data.columns if col != target]
    if not current_features:
         return jsonify({'status': 'error', 'message': 'No features found in training data (excluding target)'}), 400
    # Create dummy variables for categorical features, excluding 'Nama'
    categorical_features = [col for col in current_features if not pd.api.types.is_numeric_dtype(train_data[col]) and col != 'Nama']
    train_data = pd.get_dummies(train_data, columns=categorical_features, drop_first=True)
    features = [col for col in train_data.columns if col != target and col != 'Nama']  # Update global features based on current train_data, excluding 'Nama'

    # --- 3. Train Model (Boosting or Single Tree) ---
    boosted_trees = []
    boosted_alphas = []
    is_boosted = False # Reset boosting status for each training run

    if boosting:
        print(f"Starting Boosting with {n_estimators} estimators...")
        is_boosted = True
        sample_weights = np.ones(len(train_data)) / len(train_data)

        for i in range(n_estimators):
            print(f"Building tree {i+1}/{n_estimators}...")
            current_tree = build_tree(train_data, target, features,
                                      max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                      sample_weights=sample_weights)

            if current_tree is None:
                 print(f"Warning: Tree {i+1} could not be built.")
                 continue

            predictions = [predict_single_tree(current_tree, row) for _, row in train_data.iterrows()]

            actuals = train_data[target].values
            incorrect = np.array([1 if pred != true else 0 for pred, true in zip(predictions, actuals)])

            valid_preds_mask = np.array([p is not None for p in predictions])
            masked_weights = sample_weights[valid_preds_mask]
            masked_incorrect = incorrect[valid_preds_mask]

            total_masked_weight = np.sum(masked_weights)

            if total_masked_weight == 0:
                 error = 0.5
                 print(f"Warning: Total weight for error calculation is zero in tree {i+1}. Assigning neutral error.")
            else:
                 error = np.dot(masked_weights, masked_incorrect) / total_masked_weight

            error = np.clip(error, 1e-10, 1 - 1e-10)

            alpha = 0.5 * np.log((1 - error) / error)

            weight_update_factors = np.array([np.exp(alpha) if inc == 1 else np.exp(-alpha) for inc in incorrect])

            sample_weights *= weight_update_factors

            total_weight = np.sum(sample_weights)
            if total_weight == 0 or not np.isfinite(total_weight):
                 print("Warning: Sample weights became zero or non-finite. Resetting weights.")
                 sample_weights = np.ones(len(train_data)) / len(train_data)
            else:
                 sample_weights /= total_weight

            boosted_trees.append(current_tree)
            boosted_alphas.append(alpha)
            print(f"Tree {i+1}: Error={error:.4f}, Alpha={alpha:.4f}")

        if not boosted_trees:
             return jsonify({'status': 'error', 'message': 'Boosting failed: No trees were built.'}), 500
        tree = boosted_trees[0]
        print("Boosting complete.")

    else:
        print("Starting Single Tree Training...")
        is_boosted = False
        tree = build_tree(train_data, target, features,
                          max_depth=max_depth,
                          min_samples_split=min_samples_split)
        boosted_trees = []
        boosted_alphas = []
        print("Single tree training complete.")

    global extracted_rules
    extracted_rules = extract_rules(tree, path_data=train_data)

    # Save extracted rules to /static/rules.json
    os.makedirs('static', exist_ok=True)
    with open('static/rules.json', 'w') as f:
        json.dump(extracted_rules, f, indent=4)
    
        # --- Generate hasil_prediksi.csv using the latest rules and properly dummified test_data ---
        # test_data is from the original split (line 835), non-dummified.
        # categorical_features is from line 864.
        # features (global) is from line 866 (columns of dummified train_data).
        
        df_for_csv_output = test_data.copy()
    
        # Dummify a temporary version of test_data for rule prediction
        temp_test_dummified_for_rules = pd.get_dummies(df_for_csv_output, columns=categorical_features, drop_first=True, dtype=int)
    
        # Align columns with the features used for training the tree (from dummified train_data)
        # These are the features that rules will be based on.
        for feature_col in features:
            if feature_col not in temp_test_dummified_for_rules.columns:
                temp_test_dummified_for_rules[feature_col] = 0
        # Ensure only relevant columns (those in `features` plus any others needed by `predict` if not in `features`) are passed,
        # or ensure `predict` handles rows robustly. The current `predict` checks `feature in sample.index`.
        # Reindexing to `features` might be too strict if `predict` can handle a superset of columns.
        # For now, adding missing `features` columns is the key step.
    
        # Make predictions using rules
        rule_predictions_for_csv = temp_test_dummified_for_rules.apply(
            lambda row: predict(row, use_rules=True), axis=1
        ).values
    
        df_for_csv_output['Prediksi'] = rule_predictions_for_csv
        df_for_csv_output['Prediksi'] = df_for_csv_output['Prediksi'].fillna("N/A")
        
        # Ensure 'Jenis_Kelamin' is in 0/1 format for the CSV if it wasn't already
        # This was done to df initially, so test_data should have it as 0/1.
        # If 'Jenis_Kelamin' in df_for_csv_output.columns and df_for_csv_output['Jenis_Kelamin'].dtype == 'object':
        #    df_for_csv_output['Jenis_Kelamin'] = df_for_csv_output['Jenis_Kelamin'].map({'L': 0, 'P': 1, 'Laki-laki': 0, 'Perempuan': 1}).fillna(df_for_csv_output['Jenis_Kelamin'])
    
        df_for_csv_output.to_csv('static/hasil_prediksi.csv', index=False)
        # --- End of hasil_prediksi.csv generation ---
    
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
                     'children': [child for child in [left_child_dict, right_child_dict] if child is not None]
                 }
             tree_structure = tree_to_dict(display_tree)
         except Exception as e:
              print(f"Error calculating metrics or tree structure: {e}")

    print(f"Generating predictions using {'boosted' if is_boosted else 'single'} model...")
    results = []
    for idx, row in df.iterrows():
        pred = predict(row, use_boosting=is_boosted)

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
            'Prediksi': pred if pred is not None else "N/A"
         }
        results.append(result_dict)
    print("Prediction generation complete.")

    feature_importance = {}
    for feature in features:
        feature_importance[feature] = 0.1

    return jsonify({
        'status': 'success',
        'message': f"Model trained successfully ({'Boosting enabled' if is_boosted else 'Single tree'}).",
        'metrics': tree_metrics,
        'tree_structure': tree_structure,
        'feature_importance': feature_importance
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

def extract_rules(node, rule_conditions=[], path_data=train_data):
    rules = []
    if node.value is not None:
        subset = path_data
        if not subset.empty:
            correct = len(subset[subset[target] == node.value])
            total = len(subset)
            confidence = correct / total if total > 0 else 0
            coverage = total / len(train_data)
        else:
            confidence = 0
            coverage = 0

        machine_rule = {
            'conditions': rule_conditions,
            'prediction': node.value
        }

        display_rule = "JIKA " + " DAN ".join(
            [cond.replace(" <= ", " LEBIH KECIL SAMA DENGAN ").replace(" > ", " LEBIH BESAR DARI ") for cond in rule_conditions]
        ) + f" MAKA PREDICT {node.value}"

        rules.append({
            'display_rule': display_rule,
            'machine_rule': machine_rule,
            'confidence': round(confidence, 2),
            'coverage': round(coverage, 2)
        })
        return rules

    if node.feature and node.threshold is not None:
        condition = f"{node.feature} <= {node.threshold}"
        left_rules = extract_rules(node.left, rule_conditions + [condition], path_data[path_data[node.feature] <= node.threshold])
        right_rules = extract_rules(node.right, rule_conditions + [condition.replace('<=', '>') ], path_data[path_data[node.feature] > node.threshold])
        return left_rules + right_rules
    return rules

import requests # type: ignore

if __name__ == '__main__':
    app.run(debug=True)

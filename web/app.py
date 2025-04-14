import pandas as pd
import numpy as np
import math
from collections import Counter
from flask import Flask, render_template, request, jsonify, send_file
from io import BytesIO
import json
import os

app = Flask(__name__)

# Initial sample data
data = {
    'Usia': [45, 50, 60, 35, 55, 40, 65, 48, 52, 58],
    'Jenis_Kelamin': ['L', 'P', 'L', 'P', 'L', 'P', 'L', 'P', 'L', 'P'],
    'Tekanan_Darah': ['140/90', '130/85', '150/95', '120/80', '160/100', '110/70', '170/110', '140/90', '130/85', '150/95'],
    'Kolesterol': [220, 180, 240, 160, 260, 170, 280, 200, 190, 230],
    'Gula_Darah': [120, 110, 140, 90, 150, 100, 160, 120, 110, 130],
    'Nyeri_Dada': ['Ya', 'Tidak', 'Ya', 'Tidak', 'Ya', 'Tidak', 'Ya', 'Ya', 'Tidak', 'Ya'],
    'Sesak_Napas': ['Ya', 'Tidak', 'Ya', 'Tidak', 'Ya', 'Tidak', 'Ya', 'Tidak', 'Tidak', 'Ya'],
    'Kelelahan': ['Ya', 'Tidak', 'Ya', 'Tidak', 'Ya', 'Tidak', 'Ya', 'Ya', 'Tidak', 'Ya'],
    'Denyut_Jantung': [80, 75, 90, 70, 85, 72, 95, 78, 74, 88],
    'Penyakit_Jantung': ['ACS', 'Tidak Ada', 'Gagal Jantung', 'Tidak Ada', 'PJB Sianotik', 'Tidak Ada', 'Demam Reumatik', 'ACS', 'Tidak Ada', 'Gagal Jantung']
}

df = pd.DataFrame(data)

# Preprocess data
df[['Sistolik', 'Diastolik']] = df['Tekanan_Darah'].str.split('/', expand=True).astype(int)
df = df.drop('Tekanan_Darah', axis=1)
df['Jenis_Kelamin'] = df['Jenis_Kelamin'].map({'L': 0, 'P': 1})
df['Nyeri_Dada'] = df['Nyeri_Dada'].map({'Tidak': 0, 'Ya': 1})
df['Sesak_Napas'] = df['Sesak_Napas'].map({'Tidak': 0, 'Ya': 1})
df['Kelelahan'] = df['Kelelahan'].map({'Tidak': 0, 'Ya': 1})

# Split into training and test data
train_data = df.sample(frac=0.8, random_state=42)
test_data = df.drop(train_data.index)

# C5.0 Decision Tree Implementation
def entropy(target):
    counts = Counter(target)
    probabilities = [count / len(target) for count in counts.values()]
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

def information_gain(data, feature, target):
    total_entropy = entropy(data[target])
    values = data[feature].unique()
    weighted_entropy = 0
    for value in values:
        subset = data[data[feature] == value]
        subset_entropy = entropy(subset[target])
        weighted_entropy += (len(subset) / len(data)) * subset_entropy
    return total_entropy - weighted_entropy

def split_info(data, feature):
    values = data[feature].unique()
    split_entropy = 0
    for value in values:
        subset = data[data[feature] == value]
        probability = len(subset) / len(data)
        split_entropy -= probability * math.log2(probability) if probability > 0 else 0
    return split_entropy

def gain_ratio(data, feature, target):
    ig = information_gain(data, feature, target)
    si = split_info(data, feature)
    return ig / si if si != 0 else 0

class DecisionNode:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right

def build_tree(data, target, features, depth=0, max_depth=3, min_samples_split=2):
    if len(data[target].unique()) == 1 or depth == max_depth or len(data) < min_samples_split:
        return DecisionNode(value=Counter(data[target]).most_common(1)[0][0])
    
    best_gain = -1
    best_feature = None
    best_threshold = None
    
    for feature in features:
        if data[feature].dtype == 'object':
            continue
        unique_values = sorted(data[feature].unique())
        for threshold in unique_values:
            gain = gain_ratio(data, feature, target)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
    
    if best_gain == -1 or best_gain == 0:
        return DecisionNode(value=Counter(data[target]).most_common(1)[0][0])
    
    left_data = data[data[best_feature] <= best_threshold]
    right_data = data[data[best_feature] > best_threshold]
    
    left = build_tree(left_data, target, features, depth + 1, max_depth, min_samples_split)
    right = build_tree(right_data, target, features, depth + 1, max_depth, min_samples_split)
    
    return DecisionNode(feature=best_feature, threshold=best_threshold, left=left, right=right)

# Initial model training
target = 'Penyakit_Jantung'
features = df.columns.drop(target)
tree = build_tree(train_data, target, features, max_depth=3)

def predict(tree, sample):
    if tree.value is not None:
        return tree.value
    if sample[tree.feature] <= tree.threshold:
        return predict(tree.left, sample)
    else:
        return predict(tree.right, sample)

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
    def tree_to_dict(node):
        if node.value is not None:
            return {'name': f"Prediction: {node.value}", 'value': 'Leaf', 'samples': 50}
        return {
            'name': f"{node.feature} ≤ {node.threshold}",
            'value': 'Decision',
            'samples': 100,
            'children': [
                tree_to_dict(node.left) if node.left else None,
                tree_to_dict(node.right) if node.right else None
            ]
        }
    return jsonify(tree_to_dict(tree))

@app.route('/api/rules')
def get_rules():
    rules = []
    def extract_rules(node, rule_conditions=[]):
        if node.value is not None:
            rules.append({
                'rule': " AND ".join(rule_conditions),
                'prediction': node.value,
                'confidence': 0.9,  # Placeholder confidence
                'coverage': 0.3     # Placeholder coverage
            })
            return
        if node.feature and node.threshold:
            extract_rules(node.left, rule_conditions + [f"{node.feature} <= {node.threshold}"])
            extract_rules(node.right, rule_conditions + [f"{node.feature} > {node.threshold}"])
    
    extract_rules(tree)
    page = int(request.args.get('page', 1))
    per_page = 10
    start = (page - 1) * per_page
    end = start + per_page
    paginated_rules = rules[start:end]
    return jsonify({
        'data': paginated_rules,
        'total': len(rules),
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
    global tree, results, train_data

    # Ambil parameter dari form di frontend
    max_depth = int(request.form.get('max_depth', 3))
    min_samples_split = int(request.form.get('min_samples_split', 2))
    confidence = float(request.form.get('confidence', 0.25))  # Placeholder, belum diimplementasikan di build_tree
    boosting = request.form.get('boosting', 'false') == 'true'  # Placeholder, belum diimplementasikan

    # Validasi data pelatihan
    if train_data.empty:
        return jsonify({'status': 'error', 'message': 'No training data available'}), 400

    # Bangun pohon keputusan dengan parameter yang diberikan
    tree = build_tree(train_data, target, features, depth=0, max_depth=max_depth, min_samples_split=min_samples_split)

    # Hitung metrik pohon
    def calculate_tree_depth(node):
        if node.value is not None:
            return 0
        left_depth = calculate_tree_depth(node.left) if node.left else 0
        right_depth = calculate_tree_depth(node.right) if node.right else 0
        return max(left_depth, right_depth) + 1

    def calculate_total_nodes(node):
        if node.value is not None:
            return 1
        left_nodes = calculate_total_nodes(node.left) if node.left else 0
        right_nodes = calculate_total_nodes(node.right) if node.right else 0
        return left_nodes + right_nodes + 1

    def calculate_leaf_nodes(node):
        if node.value is not None:
            return 1
        left_leaves = calculate_leaf_nodes(node.left) if node.left else 0
        right_leaves = calculate_leaf_nodes(node.right) if node.right else 0
        return left_leaves + right_leaves

    tree_metrics = {
        'depth': calculate_tree_depth(tree),
        'nodes': calculate_total_nodes(tree),
        'leaves': calculate_leaf_nodes(tree)
    }

    # Konversi pohon ke format JSON untuk visualisasi
    def tree_to_dict(node):
        if node.value is not None:
            return {
                'name': f"Prediction: {node.value}",
                'samples': len(train_data[train_data[target] == node.value]) if node.value in train_data[target].values else 0,
                'value': 'Leaf'
            }
        return {
            'name': f"{node.feature} ≤ {node.threshold}",
            'samples': len(train_data),
            'value': 'Decision',
            'children': [
                tree_to_dict(node.left) if node.left else None,
                tree_to_dict(node.right) if node.right else None
            ]
        }

    tree_structure = tree_to_dict(tree)

    # Perbarui prediksi
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

    # Kembalikan respons JSON
    return jsonify({
        'status': 'success',
        'metrics': tree_metrics,
        'tree_structure': tree_structure
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
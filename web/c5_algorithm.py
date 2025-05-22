import numpy as np
import pandas as pd
from collections import Counter
import math

class C5Node:
    def __init__(self, feature=None, threshold=None, value=None, gain_ratio=None):
        self.feature = feature      # Feature used for splitting
        self.threshold = threshold  # Threshold for numeric features
        self.value = value          # Class value for leaf nodes
        self.gain_ratio = gain_ratio # Store the gain ratio for this split
        self.children = {}          # For categorical features: dictionary mapping values to child nodes
        self.left = None            # For numeric features: <= threshold
        self.right = None           # For numeric features: > threshold
        self.pruned = False         # Flag for pruning
        self.error_rate = 0.0       # Error estimation for pruning
        self.samples = 0            # Number of samples at this node
        self.distribution = None    # Class distribution at this node
        self.confidence = 1.0       # Confidence of the prediction (for better rule extraction)
        self.feature_type = None    # Type of feature ('numeric' or 'categorical')
        self.class_counts = {}      # Counts of each class at the node

def entropy(y):
    """Calculate the entropy of a target column"""
    if len(y) == 0:
        return 0
    
    counts = Counter(y)
    probabilities = [count / len(y) for count in counts.values()]
    return -sum(p * math.log2(p) for p in probabilities if p > 0)

def information_gain(X, y, feature, threshold=None):
    """Calculate information gain for a feature"""
    parent_entropy = entropy(y)
    
    # Handle categorical and numeric features differently
    if threshold is None:  # Categorical
        # Group by feature values
        values = set(X[feature])
        weighted_entropy = 0
        for value in values:
            subset_indices = X[feature] == value
            subset_y = y[subset_indices]
            weight = len(subset_y) / len(y)
            weighted_entropy += weight * entropy(subset_y)
    else:  # Numeric
        # Split on threshold
        left_indices = X[feature] <= threshold
        right_indices = ~left_indices
        
        left_y = y[left_indices]
        right_y = y[right_indices]
        
        if len(left_y) == 0 or len(right_y) == 0:
            return 0
        
        weighted_entropy = (len(left_y) / len(y)) * entropy(left_y) + \
                          (len(right_y) / len(y)) * entropy(right_y)
    
    return parent_entropy - weighted_entropy

def split_info(X, feature, threshold=None):
    """Calculate split information for gain ratio"""
    n = len(X)
    if n == 0:
        return 1  # Avoid division by zero
    
    if threshold is None:  # Categorical
        # Group by feature values
        counts = Counter(X[feature])
        info = 0
        for value, count in counts.items():
            proportion = count / n
            info -= proportion * math.log2(proportion) if proportion > 0 else 0
    else:  # Numeric
        # Split on threshold
        left_count = sum(X[feature] <= threshold)
        right_count = n - left_count
        
        left_proportion = left_count / n
        right_proportion = right_count / n
        
        info = 0
        if left_proportion > 0:
            info -= left_proportion * math.log2(left_proportion)
        if right_proportion > 0:
            info -= right_proportion * math.log2(right_proportion)
    
    return max(info, 1e-10)  # Avoid division by zero

def gain_ratio(X, y, feature, threshold=None):
    """Calculate gain ratio (used by C5.0)"""
    info_gain = information_gain(X, y, feature, threshold)
    split_i = split_info(X, feature, threshold)
    return info_gain / split_i if split_i > 0 else 0

def find_best_split(X, y, features, feature_types):
    """Find the best feature and split threshold"""
    best_gain_ratio = -1
    best_feature = None
    best_threshold = None
    
    for feature in features:
        if feature_types[feature] == 'numeric':
            # For numeric features, try different split points
            values = sorted(set(X[feature]))
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i+1]) / 2
                current_gain_ratio = gain_ratio(X, y, feature, threshold)
                
                if current_gain_ratio > best_gain_ratio:
                    best_gain_ratio = current_gain_ratio
                    best_feature = feature
                    best_threshold = threshold
        else:
            # For categorical features
            current_gain_ratio = gain_ratio(X, y, feature)
            
            if current_gain_ratio > best_gain_ratio:
                best_gain_ratio = current_gain_ratio
                best_feature = feature
                best_threshold = None
    
    return best_feature, best_threshold, best_gain_ratio

def build_c5_tree(X, y, features, feature_types, max_depth=None, min_samples_split=2, current_depth=0):
    """Build a C5.0 decision tree"""
    # Calculate class distribution
    class_counts = Counter(y)
    total_samples = len(y)
    
    # Base cases
    if len(set(y)) == 1:  # Pure node
        node = C5Node(value=y.iloc[0])
        node.samples = total_samples
        node.distribution = class_counts
        node.confidence = 1.0  # Pure node has 100% confidence
        node.class_counts = dict(class_counts)
        return node
    
    if len(y) < min_samples_split or (max_depth is not None and current_depth >= max_depth) or len(features) == 0:
        most_common = class_counts.most_common(1)[0][0]
        confidence = class_counts[most_common] / total_samples if total_samples > 0 else 0
        
        node = C5Node(value=most_common)
        node.samples = total_samples
        node.distribution = class_counts
        node.confidence = confidence
        node.class_counts = dict(class_counts)
        return node
    
    # Find the best split
    best_feature, best_threshold, best_gain_ratio = find_best_split(X, y, features, feature_types)
    
    # If no good split found
    if best_feature is None or best_gain_ratio <= 0:
        most_common = class_counts.most_common(1)[0][0]
        confidence = class_counts[most_common] / total_samples if total_samples > 0 else 0
        
        node = C5Node(value=most_common)
        node.samples = total_samples
        node.distribution = class_counts
        node.confidence = confidence
        node.class_counts = dict(class_counts)
        return node
      # Create the node
    node = C5Node(feature=best_feature, threshold=best_threshold, gain_ratio=best_gain_ratio)
    node.samples = len(y)
    node.distribution = Counter(y)
    node.class_counts = dict(Counter(y))
    node.feature_type = feature_types[best_feature]
    
    # Split the data
    if feature_types[best_feature] == 'numeric':
        # Binary split for numeric features
        left_indices = X[best_feature] <= best_threshold
        right_indices = ~left_indices
        
        # Check if splits are valid
        if sum(left_indices) > 0:
            node.left = build_c5_tree(
                X[left_indices], 
                y[left_indices], 
                features,  # Keep all features for now
                feature_types, 
                max_depth, 
                min_samples_split, 
                current_depth + 1
            )
        else:
            most_common = Counter(y).most_common(1)[0][0]
            confidence = Counter(y)[most_common] / len(y) if len(y) > 0 else 0
            node.left = C5Node(value=most_common)
            node.left.samples = 0
            node.left.distribution = Counter()
            node.left.confidence = confidence
            node.left.class_counts = {}
        
        if sum(right_indices) > 0:
            node.right = build_c5_tree(
                X[right_indices], 
                y[right_indices], 
                features,  # Keep all features for now
                feature_types, 
                max_depth, 
                min_samples_split, 
                current_depth + 1
            )
        else:
            most_common = Counter(y).most_common(1)[0][0]
            confidence = Counter(y)[most_common] / len(y) if len(y) > 0 else 0
            node.right = C5Node(value=most_common)
            node.right.samples = 0
            node.right.distribution = Counter()
            node.right.confidence = confidence
            node.right.class_counts = {}
    else:
        # Multiway split for categorical features
        node.children = {}
        unique_values = set(X[best_feature])
        remaining_features = [f for f in features if f != best_feature]
        
        for value in unique_values:
            indices = X[best_feature] == value
            if sum(indices) > 0:                node.children[value] = build_c5_tree(
                    X[indices], 
                    y[indices], 
                    remaining_features, 
                    feature_types, 
                    max_depth, 
                    min_samples_split, 
                    current_depth + 1
                )
            else:
                # If no examples with this value, use majority class
                most_common = Counter(y).most_common(1)[0][0]
                confidence = Counter(y)[most_common] / len(y) if len(y) > 0 else 0
                
                child_node = C5Node(value=most_common)
                child_node.samples = 0
                child_node.distribution = Counter()
                child_node.confidence = confidence
                child_node.class_counts = {}
                
                node.children[value] = child_node
    
    # Calculate the confidence of this node based on class distribution
    most_common_class = node.distribution.most_common(1)[0][0]
    node.confidence = node.distribution[most_common_class] / node.samples if node.samples > 0 else 0
    
    return node

def predict_c5_instance(node, instance):
    """
    Predict class for a single instance using the C5.0 tree
    
    Returns:
    --------
    prediction, confidence: The class prediction and confidence score
    """
    if node.value is not None:
        return node.value, getattr(node, 'confidence', 1.0)
    
    feature = node.feature
    if feature not in instance:
        # If feature is missing, return the majority class with confidence
        if node.distribution:
            majority_class = node.distribution.most_common(1)[0][0]
            confidence = node.distribution[majority_class] / node.samples if node.samples > 0 else 0
            return majority_class, confidence
        else:
            return None, 0.0
    
    feature_value = instance[feature]
    
    # Handle missing values
    if pd.isna(feature_value):
        # If value is missing, use the branch with more samples
        if hasattr(node, 'left') and hasattr(node, 'right'):
            if node.left.samples >= node.right.samples:
                return predict_c5_instance(node.left, instance)
            else:
                return predict_c5_instance(node.right, instance)
        # Just use majority class as fallback
        majority_class = node.distribution.most_common(1)[0][0]
        confidence = node.distribution[majority_class] / node.samples if node.samples > 0 else 0
        return majority_class, confidence
    
    if node.feature_type == 'numeric':  # Numeric feature
        if feature_value <= node.threshold:
            if node.left:
                return predict_c5_instance(node.left, instance)
            else:
                majority_class = node.distribution.most_common(1)[0][0] if node.distribution else None
                confidence = node.distribution[majority_class] / node.samples if node.distribution and node.samples > 0 else 0
                return majority_class, confidence
        else:
            if node.right:
                return predict_c5_instance(node.right, instance)
            else:
                majority_class = node.distribution.most_common(1)[0][0] if node.distribution else None
                confidence = node.distribution[majority_class] / node.samples if node.distribution and node.samples > 0 else 0
                return majority_class, confidence
    else:  # Categorical feature
        if feature_value in node.children:
            return predict_c5_instance(node.children[feature_value], instance)
        else:
            # Handle unseen categorical values - find the child with most samples
            if node.children:
                best_child = max(node.children.values(), key=lambda x: getattr(x, 'samples', 0))
                return predict_c5_instance(best_child, instance)
            # Fallback to majority class
            majority_class = node.distribution.most_common(1)[0][0] if node.distribution else None
            confidence = node.distribution[majority_class] / node.samples if node.distribution and node.samples > 0 else 0
            return majority_class, confidence

def extract_rules_from_c5_tree(node, path=None, feature_names=None):
    """Extract rules from a C5.0 decision tree"""
    if path is None:
        path = []
    
    if feature_names is None:
        feature_names = {}
    
    if node is None:
        return []
    
    if node.value is not None:
        # Calculate confidence from the class distribution
        confidence = getattr(node, 'confidence', 1.0)
        
        # Create the rule with more detailed statistics
        rule = {
            'conditions': path.copy(),
            'prediction': node.value,
            'confidence': confidence,
            'samples': node.samples,
            'distribution': dict(node.distribution) if node.distribution else {},
            'class_counts': getattr(node, 'class_counts', {})
        }
        
        return [rule]
    
    rules = []
    
    if node.threshold is not None:  # Numeric feature
        feature_name = feature_names.get(node.feature, node.feature)
        
        # Left child (<=)
        if node.left:
            left_path = path.copy()
            left_path.append((feature_name, '<=', node.threshold))
            rules.extend(extract_rules_from_c5_tree(node.left, left_path, feature_names))
        
        # Right child (>)
        if node.right:
            right_path = path.copy()
            right_path.append((feature_name, '>', node.threshold))
            rules.extend(extract_rules_from_c5_tree(node.right, right_path, feature_names))
    else:  # Categorical feature
        feature_name = feature_names.get(node.feature, node.feature)
        
        for value, child in node.children.items():
            child_path = path.copy()
            child_path.append((feature_name, '==', value))
            rules.extend(extract_rules_from_c5_tree(child, child_path, feature_names))
    
    return rules

def format_rules_for_display(rules, feature_names=None):
    """Format rules for display with enhanced information"""
    if feature_names is None:
        feature_names = {}
    
    formatted_rules = []
    total_samples = sum(r.get('samples', 0) for r in rules)
    
    # Sort rules by confidence (descending) and number of conditions (fewer first)
    sorted_rules = sorted(rules, key=lambda r: (-r.get('confidence', 0), len(r.get('conditions', []))))
    
    for i, rule in enumerate(sorted_rules):
        conditions = []
        for feature, operator, value in rule['conditions']:
            # Get readable feature name
            feature_display = feature_names.get(feature, feature)
            
            # Format the value based on its type and the feature
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.1f}" if isinstance(value, float) and abs(value - round(value)) > 0.01 else str(int(value))
            else:
                formatted_value = str(value)
            
            # Special handling for specific features
            if feature == 'Jenis_Kelamin':
                if operator == '==' and formatted_value in ('0', '1'):
                    formatted_value = "'L'" if formatted_value == '0' else "'P'"
                elif operator == '!=' and formatted_value in ('0', '1'):
                    formatted_value = "'P'" if formatted_value == '0' else "'L'"
            
            if feature in ('Nyeri_Dada', 'Sesak_Napas', 'Kelelahan'):
                if operator == '==' and formatted_value in ('0', '1'):
                    formatted_value = "'Tidak'" if formatted_value == '0' else "'Ya'"
                elif operator == '!=' and formatted_value in ('0', '1'):
                    formatted_value = "'Ya'" if formatted_value == '0' else "'Tidak'"
            
            # Assemble the condition
            conditions.append(f"{feature_display} {operator} {formatted_value}")
        
        condition_text = " AND ".join(conditions) if conditions else "Always"
        prediction = rule['prediction']
        confidence = rule.get('confidence', 0) * 100
        samples = rule.get('samples', 0)
        coverage = samples / total_samples * 100 if total_samples > 0 else 0
        
        # Add distribution information for better understanding
        distribution = rule.get('class_counts', {})
        class_dist_text = ", ".join([f"{cls}: {count}" for cls, count in distribution.items()])
        
        formatted_rules.append({
            'rule_id': i + 1,
            'display_rule': f"IF {condition_text} THEN {prediction}",
            'machine_rule': {
                'conditions': conditions,
                'prediction': prediction
            },
            'confidence': round(confidence, 2),
            'coverage': round(coverage, 2),
            'samples': samples,
            'class_distribution': class_dist_text
        })
    
    return formatted_rules

def calculate_accuracy(tree, X, y):
    """Calculate accuracy of the tree on a dataset"""
    correct = 0
    for i in range(len(X)):
        instance = X.iloc[i].to_dict()
        prediction = predict_c5_instance(tree, instance)
        if prediction == y.iloc[i]:
            correct += 1
    
    return correct / len(X) if len(X) > 0 else 0

class C5Booster:
    def __init__(self, max_depth=None, min_samples_split=2, n_estimators=10):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_estimators = n_estimators
        self.trees = []
        self.weights = []
        self.feature_types = {}
        
    def fit(self, X, y, feature_types=None):
        """Fit C5.0 with boosting"""
        if feature_types is None:
            # Auto-detect feature types
            self.feature_types = {}
            for col in X.columns:
                if pd.api.types.is_numeric_dtype(X[col]):
                    self.feature_types[col] = 'numeric'
                else:
                    self.feature_types[col] = 'categorical'
        else:
            self.feature_types = feature_types
        
        # Initialize sample weights
        n_samples = len(X)
        sample_weights = np.ones(n_samples) / n_samples
        
        features = list(X.columns)
        
        for i in range(self.n_estimators):
            print(f"Building tree {i+1}/{self.n_estimators}...")
            
            # Sample with replacement according to weights
            indices = np.random.choice(n_samples, size=n_samples, replace=True, p=sample_weights)
            X_sample = X.iloc[indices].reset_index(drop=True)
            y_sample = y.iloc[indices].reset_index(drop=True)
            
            # Build tree
            tree = build_c5_tree(
                X_sample, 
                y_sample, 
                features, 
                self.feature_types,
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split
            )
            
            # Make predictions
            predictions = []
            for j in range(len(X)):
                pred = predict_c5_instance(tree, X.iloc[j].to_dict())
                predictions.append(pred)
            
            # Calculate error rate
            incorrect = [predictions[j] != y.iloc[j] for j in range(len(X))]
            error_rate = sum(sample_weights[j] for j in range(len(X)) if incorrect[j])
            
            # Handle degenerate case
            if error_rate <= 0 or error_rate >= 1:
                if i == 0:  # For first tree, keep it even if error rate is high
                    self.trees.append(tree)
                    self.weights.append(1.0)
                break
            
            # Calculate tree weight
            alpha = 0.5 * np.log((1 - error_rate) / error_rate)
            self.trees.append(tree)
            self.weights.append(alpha)
            
            # Update sample weights
            for j in range(len(X)):
                if incorrect[j]:
                    sample_weights[j] *= np.exp(alpha)
                else:
                    sample_weights[j] *= np.exp(-alpha)
            
            # Normalize weights
            sample_weights = sample_weights / np.sum(sample_weights)
            
            print(f"Tree {i+1}: Error={error_rate:.4f}, Weight={alpha:.4f}")
            
            # Stop if perfect fit
            if error_rate == 0:
                break
    
    def predict(self, X):
        """Predict using the boosted ensemble"""
        if len(self.trees) == 0:
            return None
            
        predictions = []
        
        for i, instance in X.iterrows():
            instance_dict = instance.to_dict()
            votes = {}
            
            for j, tree in enumerate(self.trees):
                pred = predict_c5_instance(tree, instance_dict)
                votes[pred] = votes.get(pred, 0) + self.weights[j]
            
            # Get the class with highest weighted votes
            if votes:
                predictions.append(max(votes, key=votes.get))
            else:
                predictions.append(None)
        
        return predictions

def preprocess_data(df, target_col, id_cols=None, encode_categorical=True):
    """
    Preprocess the dataset for C5.0 algorithm
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe
    target_col : str
        The target column name
    id_cols : list, optional
        Columns to drop (ID columns, etc.)
    encode_categorical : bool
        Whether to encode categorical features
        
    Returns:
    --------
    X : pandas.DataFrame
        Features dataframe
    y : pandas.Series
        Target series
    feature_types : dict
        Dictionary mapping feature names to types
    """
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Remove ID columns if specified
    if id_cols:
        for col in id_cols:
            if col in df_processed.columns:
                df_processed = df_processed.drop(col, axis=1)
    
    # Clean up data - remove duplicate columns, handle spaces in values
    for col in df_processed.columns:
        if pd.api.types.is_string_dtype(df_processed[col]):
            df_processed[col] = df_processed[col].str.strip()
    
    # Detect feature types
    feature_types = {}
    cat_cols = []
    
    for col in df_processed.columns:
        if col == target_col:
            continue
            
        # Check if column should be categorical
        if not pd.api.types.is_numeric_dtype(df_processed[col]):
            cat_cols.append(col)
            feature_types[col] = 'categorical'
        elif df_processed[col].nunique() <= 5:  # Low cardinality numeric columns
            cat_cols.append(col)
            feature_types[col] = 'categorical'
        else:
            feature_types[col] = 'numeric'
    
    # Encode categorical features if requested
    if encode_categorical:
        for col in cat_cols:
            if col in df_processed.columns:
                if set(df_processed[col].unique()) == {'Ya', 'Tidak'}:
                    df_processed[col] = df_processed[col].map({'Tidak': 0, 'Ya': 1})
                elif set(df_processed[col].unique()) == {'L', 'P'}:
                    df_processed[col] = df_processed[col].map({'L': 0, 'P': 1})
                # Add other mappings as needed
    
    # Split into features and target
    X = df_processed.drop(target_col, axis=1)
    y = df_processed[target_col]
    
    return X, y, feature_types

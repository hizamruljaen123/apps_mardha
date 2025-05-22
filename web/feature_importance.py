import math

def calculate_feature_importance(node, features, importances=None):
    """
    Calculate feature importance by looking at their usage in the tree
    weighted by gain ratio and number of samples
    
    Parameters:
    -----------
    node : C5Node
        The tree node to analyze
    features : list
        List of feature names
    importances : dict, optional
        Dictionary to accumulate importance scores
        
    Returns:
    --------
    dict
        Feature importance scores
    """
    if importances is None:
        importances = {feature: 0 for feature in features}
    
    if node is None or node.value is not None:
        return importances
    
    # Add importance to this feature based on gain ratio and samples
    if node.feature in importances:
        # Weight importance by gain ratio and number of samples
        gain_weight = getattr(node, 'gain_ratio', 0) or 1.0
        sample_weight = getattr(node, 'samples', 0) or 1.0
        
        # Combine gain ratio and samples for a weighted importance
        importance_weight = gain_weight * (1 + math.log(sample_weight + 1))
        importances[node.feature] += importance_weight
    
    # Recursively process child nodes
    if hasattr(node, 'left') and node.left:
        calculate_feature_importance(node.left, features, importances)
    if hasattr(node, 'right') and node.right:
        calculate_feature_importance(node.right, features, importances)
    elif hasattr(node, 'children') and node.children:
        for child in node.children.values():
            calculate_feature_importance(child, features, importances)
    
    # Normalize importances
    if sum(importances.values()) > 0:
        total_importance = sum(importances.values())
        for feature in importances:
            importances[feature] /= total_importance
    
    return importances

def tree_to_dict(node):
    """Convert a decision tree to a dictionary for visualization"""
    if node is None: 
        return None
    if node.value is not None:
        return {'name': f"Predict: {node.value}", 'value': 'Leaf'}

    node_name = f"{node.feature}"
    if node.threshold is not None:
        threshold_str = f"{node.threshold:.2f}" if isinstance(node.threshold, float) else str(node.threshold)
        node_name += f" <= {threshold_str}"

    left_child_dict = tree_to_dict(node.left) if hasattr(node, 'left') and node.left else None
    right_child_dict = tree_to_dict(node.right) if hasattr(node, 'right') and node.right else None

    return {
        'name': node_name,
        'value': 'Decision',
        'children': [child for child in [left_child_dict, right_child_dict] if child is not None]
    }

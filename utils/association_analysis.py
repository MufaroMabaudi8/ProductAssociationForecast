import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

def create_transaction_matrix(transaction_data):
    """
    Create a transaction matrix for association rule mining
    
    Parameters:
    -----------
    transaction_data : pandas DataFrame
        DataFrame with TransactionID and ProductID columns
    
    Returns:
    --------
    pandas DataFrame
        One-hot encoded transaction matrix
    """
    # Group by TransactionID and aggregate ProductIDs into lists
    transactions = transaction_data.groupby('TransactionID')['ProductID'].apply(list).reset_index()
    
    # Create transaction list
    transaction_list = transactions['ProductID'].tolist()
    
    # Use TransactionEncoder to convert list into one-hot encoded matrix
    te = TransactionEncoder()
    te_ary = te.fit_transform(transaction_list)
    
    # Create DataFrame
    df_onehot = pd.DataFrame(te_ary, columns=te.columns_)
    
    return df_onehot

def perform_association_analysis(transaction_data, min_support=0.01, min_confidence=0.5, min_lift=1.0):
    """
    Perform association rule mining using the Apriori algorithm
    
    Parameters:
    -----------
    transaction_data : pandas DataFrame
        DataFrame with TransactionID and ProductID columns
    min_support : float, optional
        Minimum support threshold for itemsets
    min_confidence : float, optional
        Minimum confidence threshold for rules
    min_lift : float, optional
        Minimum lift threshold for rules
    
    Returns:
    --------
    tuple
        (frequent itemsets DataFrame, association rules DataFrame)
    """
    # Create transaction matrix
    transaction_matrix = create_transaction_matrix(transaction_data)
    
    # Run apriori algorithm to find frequent itemsets
    frequent_itemsets = apriori(
        transaction_matrix, 
        min_support=min_support, 
        use_colnames=True,
        max_len=4  # Limit to combinations of up to 4 items for performance
    )
    
    # If no frequent itemsets found, return empty DataFrames
    if len(frequent_itemsets) == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    # Generate association rules
    rules = association_rules(
        frequent_itemsets, 
        metric="confidence", 
        min_threshold=min_confidence
    )
    
    # Filter by lift
    rules = rules[rules['lift'] >= min_lift]
    
    # Sort by lift
    rules = rules.sort_values('lift', ascending=False)
    
    return frequent_itemsets, rules

def get_top_associations_for_product(rules, product_id, n=5):
    """
    Get top n associated products for a given product
    
    Parameters:
    -----------
    rules : pandas DataFrame
        Association rules DataFrame
    product_id : str
        Product ID to find associations for
    n : int, optional
        Number of top associations to return
    
    Returns:
    --------
    pandas DataFrame
        Top n association rules for the product
    """
    # Find rules where the product is in the antecedents
    product_in_antecedent = rules[rules['antecedents'].apply(lambda x: product_id in x)]
    
    # Sort by lift and take top n
    top_associations = product_in_antecedent.sort_values('lift', ascending=False).head(n)
    
    return top_associations

def get_product_bundles(rules, min_confidence=0.7, min_lift=2.0):
    """
    Identify product bundles based on strong association rules
    
    Parameters:
    -----------
    rules : pandas DataFrame
        Association rules DataFrame
    min_confidence : float, optional
        Minimum confidence for bundle identification
    min_lift : float, optional
        Minimum lift for bundle identification
    
    Returns:
    --------
    list
        List of tuples containing (bundle items, confidence, lift)
    """
    # Filter rules by confidence and lift
    strong_rules = rules[(rules['confidence'] >= min_confidence) & (rules['lift'] >= min_lift)]
    
    bundles = []
    
    for _, rule in strong_rules.iterrows():
        # Combine antecedents and consequents to form a bundle
        bundle_items = list(rule['antecedents']) + list(rule['consequents'])
        confidence = rule['confidence']
        lift = rule['lift']
        
        bundles.append((bundle_items, confidence, lift))
    
    return bundles

def calculate_cross_selling_opportunities(rules, product_id):
    """
    Calculate cross-selling opportunities for a given product
    
    Parameters:
    -----------
    rules : pandas DataFrame
        Association rules DataFrame
    product_id : str
        Product ID to find cross-selling opportunities for
    
    Returns:
    --------
    pandas DataFrame
        Cross-selling opportunities sorted by lift
    """
    # Find rules where the product is in the antecedents
    product_as_antecedent = rules[rules['antecedents'].apply(lambda x: product_id in x)]
    
    # Find rules where the product is in the consequents
    product_as_consequent = rules[rules['consequents'].apply(lambda x: product_id in x)]
    
    # Combine both
    all_rules = pd.concat([product_as_antecedent, product_as_consequent]).drop_duplicates()
    
    # Sort by lift
    all_rules = all_rules.sort_values('lift', ascending=False)
    
    return all_rules

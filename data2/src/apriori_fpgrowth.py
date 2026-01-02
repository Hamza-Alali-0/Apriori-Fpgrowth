"""
Apriori and FP-Growth Algorithm Implementations
Includes association rule generation and metrics calculation
"""

from collections import defaultdict, Counter
from itertools import combinations
from typing import List, Dict, Set, Tuple
import pandas as pd


class AprioriAlgorithm:
    """Apriori algorithm implementation for finding frequent itemsets."""
    
    def __init__(self, min_support: float = 0.01):
        """
        Initialize Apriori algorithm.
        
        Args:
            min_support: Minimum support threshold (fraction of transactions)
        """
        self.min_support = min_support
        self.frequent_itemsets = {}
        self.support_data = {}
    
    def fit(self, transactions: List[List[str]]) -> Dict[int, List[Tuple]]:
        """
        Find all frequent itemsets.
        
        Args:
            transactions: List of transactions (each is a list of items)
        
        Returns:
            Dictionary mapping k -> list of k-itemsets
        """
        n_transactions = len(transactions)
        min_support_count = self.min_support * n_transactions
        
        # Get frequent 1-itemsets
        item_counts = Counter()
        for transaction in transactions:
            for item in set(transaction):
                item_counts[item] += 1
        
        # Filter by minimum support
        frequent_1_itemsets = {
            frozenset([item]): count 
            for item, count in item_counts.items() 
            if count >= min_support_count
        }
        
        self.frequent_itemsets[1] = list(frequent_1_itemsets.keys())
        self.support_data.update(frequent_1_itemsets)
        
        k = 2
        while self.frequent_itemsets.get(k-1, []):
            # Generate candidate k-itemsets
            candidates = self._generate_candidates(self.frequent_itemsets[k-1], k)
            
            # Count support for candidates
            candidate_counts = defaultdict(int)
            for transaction in transactions:
                transaction_set = set(transaction)
                for candidate in candidates:
                    if candidate.issubset(transaction_set):
                        candidate_counts[candidate] += 1
            
            # Filter by minimum support
            frequent_k_itemsets = {
                itemset: count 
                for itemset, count in candidate_counts.items() 
                if count >= min_support_count
            }
            
            if not frequent_k_itemsets:
                break
            
            self.frequent_itemsets[k] = list(frequent_k_itemsets.keys())
            self.support_data.update(frequent_k_itemsets)
            k += 1
        
        return self.frequent_itemsets
    
    def _generate_candidates(self, prev_frequent: List[frozenset], k: int) -> Set[frozenset]:
        """Generate candidate k-itemsets from (k-1)-itemsets."""
        candidates = set()
        n = len(prev_frequent)
        
        for i in range(n):
            for j in range(i+1, n):
                # Join step: combine two (k-1)-itemsets
                union = prev_frequent[i] | prev_frequent[j]
                if len(union) == k:
                    candidates.add(union)
        
        return candidates
    
    def get_support(self, itemset: frozenset) -> float:
        """Get support value for an itemset."""
        if isinstance(itemset, (list, tuple)):
            itemset = frozenset(itemset)
        return self.support_data.get(itemset, 0)


class FPNode:
    """Node in FP-tree."""
    
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.next = None  # Link to next node with same item


class FPTree:
    """FP-tree implementation for FP-Growth algorithm."""
    
    def __init__(self, transactions: List[List[str]], min_support_count: int):
        """
        Build FP-tree from transactions.
        
        Args:
            transactions: List of transactions
            min_support_count: Minimum support count threshold
        """
        # Count item frequencies
        item_counts = Counter()
        for transaction in transactions:
            for item in transaction:
                item_counts[item] += 1
        
        # Filter items by minimum support
        self.frequent_items = {
            item: count 
            for item, count in item_counts.items() 
            if count >= min_support_count
        }
        
        # Create header table
        self.header_table = {item: None for item in self.frequent_items}
        
        # Create root node
        self.root = FPNode(None, 0, None)
        
        # Build tree
        for transaction in transactions:
            # Filter and sort items by frequency (descending)
            filtered_items = [
                item for item in transaction 
                if item in self.frequent_items
            ]
            filtered_items.sort(key=lambda x: self.frequent_items[x], reverse=True)
            
            if filtered_items:
                self._insert_transaction(filtered_items, self.root, 1)
    
    def _insert_transaction(self, items: List[str], node: FPNode, count: int):
        """Insert a transaction into the FP-tree."""
        if not items:
            return
        
        first_item = items[0]
        
        if first_item in node.children:
            # Item already exists, increment count
            node.children[first_item].count += count
        else:
            # Create new node
            new_node = FPNode(first_item, count, node)
            node.children[first_item] = new_node
            
            # Update header table
            if self.header_table[first_item] is None:
                self.header_table[first_item] = new_node
            else:
                # Follow the chain to the end
                current = self.header_table[first_item]
                while current.next is not None:
                    current = current.next
                current.next = new_node
        
        # Recursively insert remaining items
        if len(items) > 1:
            self._insert_transaction(items[1:], node.children[first_item], count)


class FPGrowth:
    """FP-Growth algorithm implementation."""
    
    def __init__(self, min_support: float = 0.01):
        self.min_support = min_support
        self.frequent_itemsets = {}
        self.support_data = {}
    
    def fit(self, transactions: List[List[str]]) -> Dict[int, List[Tuple]]:
        """
        Find all frequent itemsets using FP-Growth.
        
        Args:
            transactions: List of transactions
        
        Returns:
            Dictionary mapping k -> list of k-itemsets
        """
        n_transactions = len(transactions)
        min_support_count = int(self.min_support * n_transactions)
        
        # Build FP-tree
        fp_tree = FPTree(transactions, min_support_count)
        
        # Mine patterns
        patterns = self._mine_tree(fp_tree, set(), min_support_count, n_transactions)
        
        # Organize by itemset size
        for itemset, support in patterns.items():
            k = len(itemset)
            if k not in self.frequent_itemsets:
                self.frequent_itemsets[k] = []
            self.frequent_itemsets[k].append(itemset)
            self.support_data[itemset] = support
        
        return self.frequent_itemsets
    
    def _mine_tree(self, tree: FPTree, suffix: Set, min_support_count: int, n_transactions: int) -> Dict:
        """Recursively mine FP-tree."""
        patterns = {}
        
        # Sort items by frequency (ascending for bottom-up mining)
        items = sorted(tree.frequent_items.items(), key=lambda x: x[1])
        
        for item, count in items:
            # Create new pattern by adding item to suffix
            new_pattern = frozenset(suffix | {item})
            support_count = tree.frequent_items[item]
            
            if support_count >= min_support_count:
                patterns[new_pattern] = support_count
                
                # Find conditional pattern base
                conditional_patterns = []
                node = tree.header_table[item]
                
                while node is not None:
                    path = []
                    parent = node.parent
                    while parent.parent is not None:  # Don't include root
                        path.append(parent.item)
                        parent = parent.parent
                    
                    if path:
                        for _ in range(node.count):
                            conditional_patterns.append(path[::-1])  # Reverse path
                    
                    node = node.next
                
                # Recursively mine conditional FP-tree
                if conditional_patterns:
                    conditional_tree = FPTree(conditional_patterns, min_support_count)
                    if conditional_tree.frequent_items:
                        conditional_patterns_dict = self._mine_tree(
                            conditional_tree, suffix | {item}, min_support_count, n_transactions
                        )
                        patterns.update(conditional_patterns_dict)
        
        return patterns
    
    def get_support(self, itemset: frozenset) -> float:
        """Get support value for an itemset."""
        if isinstance(itemset, (list, tuple)):
            itemset = frozenset(itemset)
        return self.support_data.get(itemset, 0)


def generate_association_rules(
    frequent_itemsets: Dict[int, List[frozenset]],
    support_data: Dict[frozenset, int],
    n_transactions: int,
    min_confidence: float = 0.5
) -> List[Dict]:
    """
    Generate association rules from frequent itemsets.
    
    Args:
        frequent_itemsets: Dictionary of frequent itemsets by size
        support_data: Support counts for itemsets
        n_transactions: Total number of transactions
        min_confidence: Minimum confidence threshold
    
    Returns:
        List of rules with metrics
    """
    rules = []
    
    # Generate rules from itemsets of size >= 2
    for k in range(2, max(frequent_itemsets.keys()) + 1):
        if k not in frequent_itemsets:
            continue
        
        for itemset in frequent_itemsets[k]:
            # Generate all possible antecedent-consequent splits
            items = list(itemset)
            
            for i in range(1, len(items)):
                for antecedent_items in combinations(items, i):
                    antecedent = frozenset(antecedent_items)
                    consequent = itemset - antecedent
                    
                    # Calculate confidence
                    support_antecedent = support_data.get(antecedent, 0)
                    support_itemset = support_data.get(itemset, 0)
                    
                    if support_antecedent == 0:
                        continue
                    
                    confidence = support_itemset / support_antecedent
                    
                    if confidence >= min_confidence:
                        # Calculate other metrics
                        support = support_itemset / n_transactions
                        support_consequent = support_data.get(consequent, 0)
                        
                        # Lift: P(A ∪ B) / (P(A) * P(B))
                        if support_consequent > 0:
                            lift = (support_itemset / n_transactions) / (
                                (support_antecedent / n_transactions) * (support_consequent / n_transactions)
                            )
                        else:
                            lift = 0
                        
                        # Conviction: (1 - P(B)) / (1 - confidence)
                        if confidence < 1:
                            conviction = (1 - support_consequent / n_transactions) / (1 - confidence)
                        else:
                            conviction = float('inf')
                        
                        rules.append({
                            'antecedent': antecedent,
                            'consequent': consequent,
                            'support': support,
                            'confidence': confidence,
                            'lift': lift,
                            'conviction': conviction,
                            'antecedent_support': support_antecedent / n_transactions,
                            'consequent_support': support_consequent / n_transactions
                        })
    
    return rules


def rules_to_dataframe(rules: List[Dict]) -> pd.DataFrame:
    """Convert list of rules to a readable DataFrame."""
    df_rules = pd.DataFrame(rules)
    
    # Convert frozensets to readable strings
    df_rules['antecedent_str'] = df_rules['antecedent'].apply(lambda x: ', '.join(sorted(list(x))))
    df_rules['consequent_str'] = df_rules['consequent'].apply(lambda x: ', '.join(sorted(list(x))))
    
    # Reorder columns for readability
    columns = ['antecedent_str', 'consequent_str', 'support', 'confidence', 'lift', 
               'conviction', 'antecedent_support', 'consequent_support']
    
    return df_rules[columns].sort_values('lift', ascending=False)


if __name__ == "__main__":
    # Test with simple transactions
    transactions = [
        ['milk', 'bread', 'butter'],
        ['milk', 'bread'],
        ['milk', 'butter'],
        ['bread', 'butter'],
        ['milk', 'bread', 'butter', 'eggs'],
    ]
    
    print("Testing Apriori...")
    apriori = AprioriAlgorithm(min_support=0.4)
    freq_itemsets_apriori = apriori.fit(transactions)
    print(f"Frequent itemsets: {freq_itemsets_apriori}")
    
    print("\nTesting FP-Growth...")
    fpgrowth = FPGrowth(min_support=0.4)
    freq_itemsets_fpgrowth = fpgrowth.fit(transactions)
    print(f"Frequent itemsets: {freq_itemsets_fpgrowth}")
    
    print("\nGenerating rules...")
    rules = generate_association_rules(
        freq_itemsets_apriori, 
        apriori.support_data,
        len(transactions),
        min_confidence=0.5
    )
    
    df_rules = rules_to_dataframe(rules)
    print(df_rules.to_string())

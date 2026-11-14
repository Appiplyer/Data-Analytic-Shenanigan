import csv
from itertools import combinations
from pathlib import Path
import numpy as np
def load_transactions(file_path):
	#Read a text file where each line contains one comma-separated transaction.
	transactions = []
	with Path(file_path).open("r", encoding="utf-8", newline="") as file:
		data = file.readlines()
		for line in data:
			transaction = line.strip().split("\t")
			transactions.append(transaction)
	return transactions
transactions = load_transactions("Transaksi.txt")
def support(transactions, itemset):
	# Calculate the support of an itemset in the transactions
	count = 0
	for transaction in transactions:
		if all(item in transaction for item in itemset):
			count += 1
	return count / len(transactions)
def confidence(transactions, itemset, consequent):
	# Calculate the confidence of an association rule
	support_itemset = support(transactions, itemset)
	if support_itemset == 0:
		return 0
	support_consequent = support(transactions, itemset + consequent)
	return support_consequent / support_itemset
def lift(transactions, itemset, consequent):
	# Calculate the lift of an association rule
	support_itemset = support(transactions, itemset)
	support_consequent = support(transactions, consequent)
	if support_itemset == 0 or support_consequent == 0:
		return 0
	support_both = support(transactions, itemset + consequent)
	return support_both / (support_itemset * support_consequent)
def apriori(transactions, min_support):
	itemsets = []
	candidate_itemsets = [{item} for transaction in transactions for item in transaction]
	candidate_itemsets = list(set(frozenset(itemset) for itemset in candidate_itemsets))
	while candidate_itemsets:
		frequent_itemsets = []
		for itemset in candidate_itemsets:
			if support(transactions, list(itemset)) >= min_support:
				frequent_itemsets.append(itemset)
		if not frequent_itemsets:
			break
		itemsets.extend(frequent_itemsets)
		# Generate new candidate itemsets by joining frequent itemsets
		candidate_itemsets = []
		for i in range(len(frequent_itemsets)):
			for j in range(i + 1, len(frequent_itemsets)):
				union_set = frequent_itemsets[i].union(frequent_itemsets[j])
				if len(union_set) == len(frequent_itemsets[i]) + 1:
					candidate_itemsets.append(union_set)
		candidate_itemsets = list(set(candidate_itemsets))
	return itemsets
frequent_itemsets = apriori(transactions, 0.5)
print("Frequent Itemsets:")
for itemset in frequent_itemsets:
	print(itemset)
def generate_association_rules(transactions, frequent_itemsets, min_confidence):
	rules = []
	for itemset in frequent_itemsets:
		if len(itemset) < 2:
			continue
		for i in range(1, len(itemset)):
			subsets = [set(x) for x in combinations(itemset, i)]
			for subset in subsets:
				consequent = itemset.difference(subset)
				conf = confidence(transactions, list(subset), list(consequent))
				if conf >= min_confidence:
					lft = lift(transactions, list(subset), list(consequent))
					rules.append((subset, consequent, conf, lft))
	return rules

association_rules = generate_association_rules(transactions, frequent_itemsets, 0.5)
print("Association Rules:")
for antecedent, consequent, conf, lft in association_rules:
	print(f"{antecedent} => {consequent} (Confidence: {conf}, Lift: {lft})")
import pandas as pd
import numpy as np 
import operator
from math import log

risk = ["high", "high", "moderate", "high", "low", "low", "high", "moderate", "low", "low", "high", "moderate", "low", "high"]
credit_hist = ["bad", "unknown", "unknown", "unknown", "unknown", "unknown", "bad", "bad", "good", "good", "good", "good", "good", "bad"]
debt = ["high", "high", "low", "low", "low", "low", "low", "low", "low", "high", "high", "high", "high", "high"]
collateral = ["none", "none", "none", "none", "none", "adequate", "none", "adequate", "none", "adequate", "none", "none", "none", "none"]
# 0 = $0 to $15k, 1 = $15 to $35k, 2 = over 35k
income = [0, 1, 1, 0, 2, 2, 0, 2, 2, 2, 0, 1, 2, 1]

data = {"income": income, "credit_hist": credit_hist, "debt": debt, "collateral": collateral, "outcomes": risk}
df = pd.DataFrame(data = data)

log2 = lambda x:log(x)/log(2)


def entropy(outcomes, n_rows):
    results = outcomes.value_counts()
    ent = 0.0
    for result in results:
        prob = float(result) / n_rows
        ent = ent - prob * log2(prob)
    return ent


def split_info(df, column):
    n_rows = len(df[column])
    data = df[[column, "outcomes"]]
    ent = 0.0
    for cat in data[column].unique():
        prob = len(data.loc[data[column] == cat]) / n_rows
        ent = ent - prob * log2(prob)
    return ent


def gini(df, column):
    data = df[[column, "outcomes"]]
    ent = 1.0
    for cat in data[column].unique():
        n_rows = len(data.loc[data[column] == cat])
        for result in data["outcomes"].unique():
            prob = (len(data.loc[(data[column] == cat) & (data["outcomes"] == result)]) / n_rows)**2
            ent = ent - prob
    return ent


def get_info_cont(df, column, outcomes):
    column_values = df[column]
    labeled_data = pd.concat([column_values, outcomes], axis=1)
    ent_dict = {}
    for cat_value in column_values.unique():
        cat_list = labeled_data.loc[labeled_data[column] == cat_value]
        n_rows = len(cat_list)
        ent = entropy(cat_list["outcomes"], n_rows)
        ent_dict[cat_value] = {"entropy":ent, "n_rows": n_rows}
    return ent_dict


def get_info_score(df, outcomes):
    total_rows = len(df)
    top_layer_ent = entropy(outcomes, total_rows)
    columns = df.columns.drop("outcomes")
    ig_scores = {}
    for column in columns:
        ent_dict = get_info_cont(df, column, outcomes)
        # get the second half of the ic formula
        second_half_ig = sum(map(lambda cat_value: ent_dict[cat_value]["n_rows"]/total_rows*ent_dict[cat_value]["entropy"], ent_dict.keys()))
        ig_score = top_layer_ent - second_half_ig
        ig_scores[column] = ig_score
    # simple logging
    print("Information Gain scores:")
    print(ig_scores)
    return ig_scores


def make_tree():
    next_layer = get_info_score(df, df.outcomes)
    layer = max(next_layer.items(), key=operator.itemgetter(1))[0]
    print("Layer 1:")
    print(layer)
    print(df[layer].unique())
    for node in df[layer].unique():
        node_df = df.loc[df[layer] == node]
        n_rows = len(node_df)
        if entropy(node_df["outcomes"], n_rows) == 0:
            print("Node:")
            print(layer)
            print(node)
            print(node_df.outcomes.unique())
        else:
            print("Node:")
            print(layer)
            print(node)
            node_df = node_df.drop([layer], axis = 1)
            next_layer = get_info_score(node_df, node_df.outcomes)
            next_layer = max(next_layer.items(), key=operator.itemgetter(1))[0]
            print("Layer 2:")
            print(next_layer)
            print(df[next_layer].unique())
            for next_node in node_df[next_layer].unique():
                next_node_df = node_df.loc[node_df[next_layer] == next_node]
                n_rows = len(next_node_df)
                if len(next_node_df.outcomes.unique()) == 1:
                    print("Node:")
                    print(next_layer)
                    print(next_node)
                    print(next_node_df.outcomes.unique())
                else:
                    print("Node:")
                    print(next_layer)
                    print(next_node)
                    next_node_df = next_node_df.drop([next_layer], axis = 1)
                    last_layer = get_info_score(next_node_df, next_node_df.outcomes)
                    last_layer = max(last_layer.items(), key=operator.itemgetter(1))[0]
                    print("Layer 3:")
                    print(last_layer)
                    print(next_node_df[last_layer].unique())
                    for final_node in node_df[last_layer].unique():
                        final_node_df = next_node_df.loc[next_node_df[last_layer] == final_node]
                        n_rows = len(final_node_df)
                        if len(final_node_df.outcomes.unique()) == 1:
                            print("Node:")
                            print(last_layer)
                            print(final_node)
                            print(final_node_df.outcomes.unique())
                        else:
                            print("Node:")
                            print(last_layer)
                            print(final_node)
                            final_node_df = final_node_df.drop([last_layer], axis = 1)
                            bottom_layer = get_info_score(final_node_df, final_node_df.outcomes)
                            bottom_layer = max(bottom_layer.items(), key=operator.itemgetter(1))[0]
                            print("Layer 4:")
                            print(bottom_layer)
                            print(final_node_df[bottom_layer].unique())
            
            
if __name__ == "__main__":
    make_tree()

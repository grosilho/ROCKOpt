def plot_options():
    import matplotlib.pyplot as plt

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    markers = ["o", "x", "s", "d", "^", "<", ">"]

    return colors, markers


def print_table(stats):
    import pandas as pd

    table = [list(stat.values()) for stat in stats.values()]

    rows = list(stats.keys())
    columns = list(stats[rows[0]].keys())
    df = pd.DataFrame(table, columns=columns, index=rows)
    print(df)


def key_to_tex(key):
    if key == "fx":
        return r"$f(x)$"
    elif key == "train_loss":
        return r"Train Loss"
    elif key == "train_accuracy":
        return r"Train Accuracy"
    elif key == "test_loss":
        return r"Test Loss"
    elif key == "test_accuracy":
        return r"Test Accuracy"
    elif key == "batch_loss":
        return r"Batch Loss"
    elif key == "batch_accuracy":
        return r"Batch Accuracy"
    else:
        return key

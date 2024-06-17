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

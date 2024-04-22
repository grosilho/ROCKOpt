def plot_options():
    opts = dict()

    opts["TR"] = {"plot_circles": False, "marker": "o", "color": "blue"}
    opts["StabTR"] = {"plot_circles": False, "marker": "x", "color": "green"}
    opts["StabGF"] = {"plot_circles": False, "marker": "s", "color": "red"}
    opts["StabNF"] = {"plot_circles": False, "marker": "d", "color": "purple"}
    opts["SplitStabNF"] = {"plot_circles": False, "marker": "^", "color": "orange"}
    opts["ExactGF"] = {"plot_circles": False, "marker": "none", "color": "black"}
    opts["ExactNF"] = {"plot_circles": False, "marker": "<", "color": "brown"}

    return opts


def print_table(stats):
    import pandas as pd

    table = [list(stat.values()) for stat in stats.values()]

    rows = list(stats.keys())
    columns = list(stats[rows[0]].keys())
    df = pd.DataFrame(table, columns=columns, index=rows)
    print(df)

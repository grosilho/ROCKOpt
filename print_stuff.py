def plot_options():
    opts = dict()

    opts["TrustRegion"] = {"plot_circles": False, "marker": "o", "color": "blue"}
    opts["StabilizedTrustRegion"] = {"plot_circles": False, "marker": "x", "color": "green"}
    opts["StabilizedGradientFlow"] = {"plot_circles": False, "marker": "s", "color": "red"}
    opts["StabilizedNewtonFlow"] = {"plot_circles": False, "marker": "d", "color": "purple"}
    opts["SplitStabilizedNewtonFlow"] = {"plot_circles": False, "marker": "^", "color": "orange"}
    opts["ExactGradientFlow"] = {"plot_circles": False, "marker": "none", "color": "black"}
    opts["ExactNewtonFlow"] = {"plot_circles": False, "marker": "<", "color": "brown"}

    return opts


def print_table(stats):
    import pandas as pd

    table = [list(stat.values()) for stat in stats.values()]

    rows = list(stats.keys())
    columns = list(stats[rows[0]].keys())
    df = pd.DataFrame(table, columns=columns, index=rows)
    print(df)

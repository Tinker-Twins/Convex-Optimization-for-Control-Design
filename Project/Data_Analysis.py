from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DATA = {
    "Straight":    {"Open-Loop": (3.81e-2, 5.03e-2), "Non-Robust": (9.73e-2, 6.04e-2), "Robust": (1.83e-2, 1.82e-2)},
    "Lane-Change": {"Open-Loop": (7.59e-2, 6.59e-2), "Non-Robust": (5.78e-2, 3.38e-2), "Robust": (1.20e-2, 1.18e-2)},
    "Skidpad":     {"Open-Loop": (1.09e-1, 1.01e-1), "Non-Robust": (7.91e-2, 8.36e-2), "Robust": (2.14e-2, 2.06e-2)},
    "Fishhook":    {"Open-Loop": (5.52e-2, 6.79e-2), "Non-Robust": (4.23e-2, 7.75e-2), "Robust": (1.50e-2, 1.82e-2)},
    "Slalom":      {"Open-Loop": (1.82e-1, 1.91e-1), "Non-Robust": (2.87e-1, 2.52e-1), "Robust": (6.14e-2, 6.15e-2)},
    "Figure-8":    {"Open-Loop": (1.43e-1, 4.15e-1), "Non-Robust": (6.85e-2, 5.47e-1), "Robust": (4.67e-2, 1.07e-1)},
}
COLORS = {"Open-Loop": "#d97706", "Non-Robust": "#7c3aed", "Robust": "#087e8b"}
MARKERS = {"Straight": "o", "Lane-Change": "s", "Skidpad": "^", "Fishhook": "D", "Slalom": "P", "Figure-8": "X"}

def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax, parity) = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    maneuvers = list(DATA)
    for row, maneuver in enumerate(maneuvers):
        for offset, (controller, color) in zip((-0.22, 0, 0.22), COLORS.items()):
            sim, real = DATA[maneuver][controller]
            y = row + offset
            ax.plot([sim, real], [y, y], color=color, alpha=0.55, lw=2)
            ax.scatter(sim, y, s=68, facecolors="white", edgecolors=color, linewidths=1.8, zorder=3)
            ax.scatter(real, y, s=68, color=color, edgecolors=color, zorder=3)
    ax.set_yticks(range(len(maneuvers)), maneuvers)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Error (ε) ↓")
    ax.set_ylabel("Maneuver")
    ax.set_title("Error Magnitude", loc="left", fontweight="bold")
    ax.grid(axis="x", which="both", alpha=0.2)
    controller_handles = [Line2D([0], [0], color=c, lw=3, label=n) for n, c in COLORS.items()]
    fill_handles = [Line2D([0], [0], marker="o", color="0.35", markerfacecolor="white", linestyle="None", label="Sim"),
                    Line2D([0], [0], marker="o", color="0.35", markerfacecolor="0.35", linestyle="None", label="Real")]
    ax.legend(handles=controller_handles + fill_handles, loc="upper right", ncol=2, frameon=False)

    for maneuver, values in DATA.items():
        for controller, (sim, real) in values.items():
            parity.scatter(sim, real, color=COLORS[controller], marker=MARKERS[maneuver],
                           s=85, edgecolor="white", linewidth=0.7, zorder=3)
    low, high = 0.009, 0.7
    parity.plot([low, high], [low, high], color="0.35", linestyle="--", lw=1.3, label="Real = Sim")
    parity.set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high),
               xlabel="Sim Error (ε) ↓", ylabel="Real Error (ε) ↓")
    parity.set_aspect("equal", adjustable="box")
    parity.set_title("Sim2Real Parity", loc="left", fontweight="bold")
    parity.grid(which="both", alpha=0.2)
    parity.text(0.02, 0.25, "Real > Sim", color="0.35")
    parity.text(0.16, 0.02, "Real < Sim", color="0.35")
    shape_handles = [Line2D([0], [0], marker=m, color="0.35", linestyle="None", markersize=7, label=n)
                     for n, m in MARKERS.items()]
    parity.legend(handles=shape_handles + [Line2D([0], [0], color="0.35", linestyle="--", label="Real = sim"),
                                          Line2D([], [], linestyle="None", label=""),
                                          Line2D([], [], linestyle="None", label="")],
                  loc="upper left", ncol=3, frameon=False, fontsize=9)
    fig.suptitle("Sim2Real Robust Control: Performance and Benchmarking", fontsize=15, fontweight="bold")
    output = Path(__file__).resolve().with_name("Data_Analysis.png")
    fig.savefig(output, dpi=200, bbox_inches="tight")
    print(f"Saved {output}")
    plt.show()

if __name__ == "__main__":
    main()
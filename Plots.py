import numpy as np
import matplotlib.pyplot as plt

"""
Plot function takes list of results for various models.
Input example:
    results = [[t1_acc, v1_acc], [t2_acc, v2_acc]],
    name = "Accuracy"

"""
def plot(results, name, loc="lower right"):
    plt.figure(dpi=200)
    plt.gca().yaxis.grid(True, linestyle='--')
    colors = ['lightskyblue', 'red', 'black']
    names = ["Plain", "Baseline", "AutoAugment"]

    for i, values in enumerate(results):
        """ Training values. """
        plt.plot(values[0], label=names[i], color=colors[i])
        """ Validation values. """
        plt.plot(values[1], color=colors[i], linestyle='--', linewidth=1)

    plt.legend(loc=loc)
    plt.ylabel(name)
    plt.xlabel("Epochs")
    plt.xticks(np.arange(0, len(values)+1, 5))
    plt.savefig(name+".png")

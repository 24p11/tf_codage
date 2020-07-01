import matplotlib.pyplot as plt


def plot_metrics(history):
    fig, axes = plt.subplots(2, 2)

    metrics = ["loss", "recall", "precision"]

    for metric_name, ax in zip(metrics, axes.flat):
        ax.plot(history[metric_name], label=metric_name)
        ax.plot(history["val_" + metric_name], label="val_" + metric_name)
        ax.legend(frameon=False)

    axes[1, 1].set_visible(False)
    axes[1, 0].set_xlabel("epochs")
    axes[0, 1].set_xlabel("epochs")

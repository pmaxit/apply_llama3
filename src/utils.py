### Plot
import matplotlib.pyplot as plt
import pandas as pd

def print_metrics(filename, save=False):
    metrics = pd.read_csv(filename)
    
    aggreg_metrics = []
    agg_col = "epoch"
    for i, dfg in metrics.groupby(agg_col):
        agg = dict(dfg.mean())
        agg[agg_col] = i
        aggreg_metrics.append(agg)
    
    df_metrics = pd.DataFrame(aggreg_metrics)
    df_metrics[["train_loss", "val_loss"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="Loss"
    )

    if save:
        plt.savefig("suggest_loss.pdf")
    
    df_metrics[["train_accuracy", "val_accuracy"]].plot(
        grid=True, legend=True, xlabel="Epoch", ylabel="ACC"
    )

    if save:
        plt.savefig("suggest_acc.pdf")
    
    plt.show()
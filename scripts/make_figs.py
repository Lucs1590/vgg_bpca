import pandas as pd
import matplotlib.pyplot as plt

# load dataframes with bpca_pooling, avg_pooling, max_pooling
accuracies = pd.read_csv("/Volumes/SSD/Mestrado/colab-tests/sib_food_accs.csv")
losses = pd.read_csv("/Volumes/SSD/Mestrado/colab-tests/sib_food_loss.csv")
# remove unnamed column
accuracies = accuracies.drop(accuracies.columns[0], axis=1)
losses = losses.drop(losses.columns[0], axis=1)

# multiply accuracies by 100
accuracies = accuracies * 100

# rename columns
accuracies.columns = ["AvgPooling", "MaxPooling", "BPCAPooling"]
losses.columns = ["AvgPooling", "MaxPooling", "BPCAPooling"]

# plot accuracies and losses in a single figure
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
# make a point line (with different colors) in the accuracies plot for the best accuracy
ax[0].plot(accuracies["AvgPooling"].idxmax(), accuracies["AvgPooling"].max(), "ro", label="AvgPooling")
ax[0].plot(accuracies["MaxPooling"].idxmax(), accuracies["MaxPooling"].max(), "ro", label="MaxPooling")
ax[0].plot(accuracies["BPCAPooling"].idxmax(), accuracies["BPCAPooling"].max(), "ro", label="BPCAPooling")
# make a point line in the losses plot for the best loss
ax[1].plot(losses["AvgPooling"].idxmin(), losses["AvgPooling"].min(), "ro")
ax[1].plot(losses["MaxPooling"].idxmin(), losses["MaxPooling"].min(), "ro")
ax[1].plot(losses["BPCAPooling"].idxmin(), losses["BPCAPooling"].min(), "ro")
# plot accuracies and losses
accuracies.plot(ax=ax[0])
losses.plot(ax=ax[1])
ax[0].set_title("Accuracies")
ax[1].set_title("Losses")
ax[0].set_xlabel("Epochs")
ax[1].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy (%)")
ax[1].set_ylabel("Loss")

# save figure
plt.tight_layout()
plt.savefig("sib_food.png")
plt.show()
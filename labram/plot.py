import json
import matplotlib.pyplot as plt
import os

def plot_metrics_from_log(log_path="log.txt", output_filename="training_metrics.png"):
    """
    Load data from a log file and plot training metrics.

    Args:
        log_path (str): path to log file
        output_filename (str): path to save the plot
    """
    if not os.path.exists(log_path):
        print(f"Error: log file '{log_path}' not exist。")
        return

    metrics = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_class_acc": [],
        "val_accuracy": [],
        "val_balanced_accuracy": [],
        "test_accuracy": [],
        "test_balanced_accuracy": [],
        "train_lr": [],
        "val_f1_weighted": [],
        "test_f1_weighted": [],
    }

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                # convert the JSON string to a dictionary
                log_entry = json.loads(line)
                
                for key in metrics.keys():
                    if key in log_entry:
                        metrics[key].append(log_entry[key])

            except json.JSONDecodeError:
                print(f"Warning: can not resolve: {line}")
                continue
    
    if not metrics["epoch"]:
        print("Error: can not parse any data from the log file. Please check the file format.")
        return

    # create a 2x2 subplot
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training and Evaluation Metrics', fontsize=16)

    epochs = metrics["epoch"]

    # Loss
    ax = axs[0, 0]
    if metrics["train_loss"]:
        ax.plot(epochs, metrics["train_loss"], label='Train Loss')
    if metrics["val_loss"]:
        ax.plot(epochs, metrics["val_loss"], label='Validation Loss')
    if metrics["test_loss"]:
        ax.plot(epochs, metrics["test_loss"], label='Test Loss')
    ax.set_title('Loss over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True)

    # Accuracy
    ax = axs[0, 1]
    if metrics["train_class_acc"]:
        ax.plot(epochs, metrics["train_class_acc"], label='Train Accuracy')
    if metrics["val_accuracy"]:
        ax.plot(epochs, metrics["val_accuracy"], label='Validation Accuracy')
    if metrics["test_accuracy"]:
        ax.plot(epochs, metrics["test_accuracy"], label='Test Accuracy')
    ax.set_title('Accuracy over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True)

    # Learning Rate
    ax = axs[1, 0]
    if metrics["train_lr"]:
        ax.plot(epochs, metrics["train_lr"], label='Learning Rate', color='green')
        ax.set_title('Learning Rate over Epochs')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.legend()
        ax.grid(True)

    # Balanced Accuracy
    ax = axs[1, 1]
    if metrics["val_balanced_accuracy"]:
        ax.plot(epochs, metrics["val_balanced_accuracy"], label='Validation Balanced Accuracy')
    if metrics["test_balanced_accuracy"]:
        ax.plot(epochs, metrics["test_balanced_accuracy"], label='Test Balanced Accuracy')
    ax.set_title('Balanced Accuracy over Epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Balanced Accuracy')
    ax.legend()
    ax.grid(True)

    # If you want to plot F1 Score, 
    # comment one section above and uncomment the following section

    # # F1 Score
    # ax = axs[1, 1]
    # if metrics["val_f1_weighted"]:
    #     ax.plot(epochs, metrics["val_f1_weighted"], label='Validation F1 (Weighted)')
    # if metrics["test_f1_weighted"]:
    #     ax.plot(epochs, metrics["test_f1_weighted"], label='Test F1 (Weighted)')
    # ax.set_title('F1 Score (Weighted) over Epochs')
    # ax.set_xlabel('Epoch')
    # ax.set_ylabel('F1 Score')
    # ax.legend()
    # ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_filename)
    plt.close(fig)  
    print(f"Figure saved at: {output_filename}")

if __name__ == '__main__':
    plot_metrics_from_log(
        log_path = "./labram/ckpts/finetune_hmc_base/log.txt",
        output_filename = "./labram/result/hmc_training_metrics.png"
    )
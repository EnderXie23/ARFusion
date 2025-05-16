import os
import argparse
import re
import matplotlib.pyplot as plt
import numpy as np

best_epoch = 0

def plot_loss(losses, save_path, title="Training Loss"):
    """
    Plot the training loss and save the figure.

    Args:
        losses (list): List of loss values.
        save_path (str): Path to save the figure.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss', color='blue')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

    plt.axvline(x=best_epoch, color='red', linestyle='--', linewidth=1, label='Best Epoch')
    plt.text(best_epoch + 0.5, plt.ylim()[1]*0.9, f'Epoch {best_epoch}\nLoss {losses[best_epoch]}', color='red')
    plt.legend(loc='upper left')
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    combined_save_path = os.path.join(save_path, f'{title.replace(" ", "_").lower()}.png')
    plt.savefig(combined_save_path)
    plt.close()  # Close the figure to free memory

def plot_accuracy(accuracies, save_path, title="Training Accuracy"):
    """
    Plot the training accuracy and save the figure.

    Args:
        accuracies (list): List of accuracy values.
        save_path (str): Path to save the figure.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(accuracies, label='Accuracy', color='green')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()

    plt.axvline(x=best_epoch, color='red', linestyle='--', linewidth=1, label='Best Epoch')
    plt.text(best_epoch + 0.5, plt.ylim()[1]*0.9, f'Epoch {best_epoch}\nAccuracy {accuracies[best_epoch]}', color='red')
    plt.legend(loc='upper left')
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    combined_save_path = os.path.join(save_path, f"{title.replace(' ','_').lower()}.png")
    plt.savefig(combined_save_path)
    plt.close()  # Close the figure to free memory

def plot_loss_and_accuracy(losses, accuracies, save_dir, title="Training Loss and Accuracy"):
    """
    Plot both loss and accuracy in the same graph.

    Args:
        losses (list): List of loss values.
        accuracies (list): List of accuracy values.
        save_dir (str): Directory to save the figures.
    """
    plt.figure(figsize=(10, 5))

    # Create primary axis for loss
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(losses, label='Loss', color='blue')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Create secondary axis for accuracy
    ax2 = ax1.twinx()
    ax2.plot(accuracies, label='Accuracy', color='green')
    ax2.set_ylabel('Accuracy', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Add title and grid
    plt.title(title)
    ax1.grid()
    ax1.axvline(x=best_epoch, color='red', linestyle='--', linewidth=1, label='Best Epoch')
    ax1.legend(loc='upper left')
    ax1.text(best_epoch + 0.5, ax1.get_ylim()[1]*0.9, f'Epoch {best_epoch}\nAccuracy {accuracies[best_epoch]}\nLoss {losses[best_epoch]}', color='red')

    # Save the figure
    os.makedirs(save_dir, exist_ok=True)
    combined_save_path = os.path.join(save_dir, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(combined_save_path)
    plt.close()

def extract_losses(log_file):
    """
    Extract losses from a log file.

    Args:
        log_file (str): Path to the log file.

    Returns:
        list: List of loss values.
    """
    losses = []
    i = 0
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(rf'Is_training: False\. \[{i},99\]\[101,111\], G_loss: ([0-9]*\.?[0-9]+)', line)
            if match:
                losses.append(float(match.group(1)))
                i += 1
    return losses

def extract_accuracies(log_file):
    """
    Extract accuracies from a log file.

    Args:
        log_file (str): Path to the log file.

    Returns:
        list: List of accuracies.
    """
    global best_epoch

    accuracies = []
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(r'Lastest model updated\. Epoch_acc=([0-9]*\.?[0-9]+)', line)
            if match:
                accuracies.append(float(match.group(1)))

    # Find the best epoch
    best_epoch = np.argmax(accuracies)
    return accuracies

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training plotting script')
    parser.add_argument('--log_file', type=str, required=True, help='Path to the log file')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the plots')
    parser.add_argument('--loss_title', type=str, default="", help='List of loss values')
    parser.add_argument('--acc_title', type=str, default="", help='List of accuracy values')
    parser.add_argument('--la_title', type=str, default="", help='List of loss and accuracy values')
    args = parser.parse_args()

    # print(extract_accuracies(args.log_file))
    # print(extract_losses(args.log_file))
    
    if args.acc_title != "":
        print("Plotting accuracy...")
        plot_accuracy(extract_accuracies(args.log_file), args.save_dir, args.acc_title)
    if args.loss_title != "":
        print("Plotting loss...")
        plot_loss(extract_losses(args.log_file), args.save_dir, args.loss_title)
    if args.la_title != "":
        print("Plotting loss and accuracy...")
        plot_loss_and_accuracy(extract_losses(args.log_file), extract_accuracies(args.log_file), args.save_dir, args.la_title)
    print(f"Plots saved to {args.save_dir}")

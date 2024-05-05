import matplotlib.pyplot as plt

def parse_log_file(file_path):
    epochs = []
    bce_losses = []
    precisions = []
    recalls = []
    f1_scores = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Train Epoch:") or line.startswith("Test Epoch:"):
                epoch = int(line.split()[2])
                epochs.append(epoch)
            elif line.startswith("Iteration"):
                parts = line.split("-")
                bce_loss = float(parts[1].split()[3])
                precision = float(parts[2].split()[2])
                recall = float(parts[3].split()[2])
                f1 = float(parts[4].split()[2])
                bce_losses.append(bce_loss)
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)

    return epochs, bce_losses, precisions, recalls, f1_scores

def plot_metrics(epochs, bce_losses, precisions, recalls, f1_scores, output_dir):
    # Plot BCE Loss
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, bce_losses, marker='o')
    plt.title('BCE Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig(output_dir + "/bce_loss.png")
    plt.close()

    # Plot Precision
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, precisions, marker='o')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.tight_layout()
    plt.savefig(output_dir + "/precision.png")
    plt.close()

    # Plot Recall
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, recalls, marker='o')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.tight_layout()
    plt.savefig(output_dir + "/recall.png")
    plt.close()

    # Plot F1 Score
    plt.figure(figsize=(6, 4))
    plt.plot(epochs, f1_scores, marker='o')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.tight_layout()
    plt.savefig(output_dir + "/f1_score.png")
    plt.close()

if __name__ == "__main__":
    log_file_path = "logs.txt"
    output_dir = "dsmil/figures/"
    epochs, bce_losses, precisions, recalls, f1_scores = parse_log_file(log_file_path)
    plot_metrics(epochs, bce_losses, precisions, recalls, f1_scores, output_dir)

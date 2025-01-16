import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix_from_csv(file_path, title):
    cm = pd.read_csv(file_path, index_col=0)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f"{title}_confusion_matrix.png")  # Save the figure as an image file
    plt.show()

# File paths for confusion matrix data
file_paths = {
    'CNN': 'C:/Users/Syber/Desktop/VS_code/code/Metrics/CNN_confusion_matrix_data.csv',
    'CNN_LSTM': 'C:/Users/Syber/Desktop/VS_code/code/Metrics/CNN_LSTM_confusion_matrix_data.csv',
    'MH_CNN_LSTM': 'C:/Users/Syber/Desktop/VS_code/code/Metrics/MH-CNN-LSTM_confusion_matrix_data.csv',
    'LSTM': 'C:/Users/Syber/Desktop/VS_code/code/Metrics/LSTM_confusion_matrix_data.csv'
}

# Plot each confusion matrix in a separate figure
for model_name, file_path in file_paths.items():
    plot_confusion_matrix_from_csv(file_path, model_name)

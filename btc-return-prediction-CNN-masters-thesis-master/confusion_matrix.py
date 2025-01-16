import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix_from_csv(file_path, ax, title):
    cm = pd.read_csv(file_path, index_col=0)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

# File paths for confusion matrix data
file_paths = {
    'CNN': 'C:/Users/Syber/Desktop/VS_code/code/Metrics/CNN_confusion_matrix_data.csv',
    'CNN_LSTM': 'C:/Users/Syber/Desktop/VS_code/code/Metrics/CNN_LSTM_confusion_matrix_data.csv',
    'MH_CNN_LSTM': 'C:/Users/Syber/Desktop/VS_code/code/Metrics/MH-CNN-LSTM_confusion_matrix_data.csv',
    'LSTM': 'C:/Users/Syber/Desktop/VS_code/code/Metrics/LSTM_confusion_matrix_data.csv'
}

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot each confusion matrix
plot_confusion_matrix_from_csv(file_paths['CNN'], axes[0, 0], 'CNN')
plot_confusion_matrix_from_csv(file_paths['CNN_LSTM'], axes[0, 1], 'CNN_LSTM')
plot_confusion_matrix_from_csv(file_paths['MH_CNN_LSTM'], axes[1, 0], 'MH_CNN_LSTM')
plot_confusion_matrix_from_csv(file_paths['LSTM'], axes[1, 1], 'LSTM')

# Adjust layout
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix_with_table(file_path, title):
    cm = pd.read_csv(file_path, index_col=0)
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [5, 2]})
    
    # Plot the confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=ax[0])
    ax[0].set_title(title)
    ax[0].set_xlabel('Predicted')
    ax[0].set_ylabel('Actual')
    
    # Create a table below the confusion matrix
    ax[1].axis('tight')
    ax[1].axis('off')
    table_data = cm.values
    table = ax[1].table(cellText=table_data, colLabels=cm.columns, rowLabels=cm.index, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    
    plt.tight_layout()
    plt.savefig(f"{title}_confusion_matrix_with_table.png")  # Save the figure as an image file
    plt.show()

# File paths for confusion matrix data
file_paths = {
    'CNN': 'C:/Users/Syber/Desktop/VS_code/code/Metrics/CNN_confusion_matrix_data.csv',
    'CNN_LSTM': 'C:/Users/Syber/Desktop/VS_code/code/Metrics/CNN_LSTM_confusion_matrix_data.csv',
    'MH_CNN_LSTM': 'C:/Users/Syber/Desktop/VS_code/code/Metrics/MH-CNN-LSTM_confusion_matrix_data.csv',
    'LSTM': 'C:/Users/Syber/Desktop/VS_code/code/Metrics/LSTM_confusion_matrix_data.csv'
}

# Plot each confusion matrix with table in a separate figure
for model_name, file_path in file_paths.items():
    plot_confusion_matrix_with_table(file_path, model_name)

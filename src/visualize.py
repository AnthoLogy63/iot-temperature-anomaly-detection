import matplotlib.pyplot as plt
import seaborn as sns


def plot_anomalies(X, anomalies, filepath, title='Detección de Anomalías'):
    plt.figure(figsize=(10, 6))

    if X.shape[1] >= 2:
        cols = X.columns[:2].tolist()
        data = X.copy()
        data['anomaly'] = 0
        data.iloc[anomalies, data.columns.get_loc('anomaly')] = 1

        colors = data['anomaly'].map({0: 'blue', 1: 'red'})
        plt.scatter(data[cols[0]], data[cols[1]], c=colors, alpha=0.6, edgecolors='k', linewidth=0.5)
        plt.xlabel(cols[0])
        plt.ylabel(cols[1])
    else:
        plt.scatter(range(len(X)), X.iloc[:, 0], c='blue', alpha=0.6, label='Normal')
        plt.scatter(anomalies, X.iloc[anomalies, 0], c='red', alpha=0.8, label='Anomalía')
        plt.xlabel('Índice')
        plt.ylabel(X.columns[0])

    plt.title(title)
    plt.legend(['Normal', 'Anomalía'])
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def load_all_loss_curves(loss_results_directory, target_column):
    """Loads all loss curves from previous runs in a certain directory."""
    loss_curves = []

    loss_files = list(loss_results_directory.rglob("RUN_*/losses_*.parquet"))
    for file_name in loss_files:
        # run_id = file_name.relative_to(loss_results_directory).parts[0]
        df_losses = pd.read_parquet(file_name)
        loss_curves.append(df_losses[target_column].to_numpy())
    return np.stack(loss_curves, axis=0)


def find_convergence_epoch(loss_curve, threshold=0.98):
    """Finds the epoch where a certain percentage of the loss decrease have been achieved."""
    initial_loss = loss_curve[0]
    min_loss = min(loss_curve)
    loss_range = initial_loss - min_loss

    # Compute the epoch where 99% of the loss decrease has occurred
    target_loss = initial_loss - threshold * loss_range

    for epoch in range(len(loss_curve)):
        if loss_curve[epoch] <= target_loss:
            return epoch
    return len(loss_curve) - 1  # If never reached, return last epoch


if __name__ == '__main__':
    # target column can be:
        # train_total_loss,
        # valid_total_loss,
        # train_r2,
        # valid_r2

    target_column = 'valid_total_loss'
    threshold = 0.99

    # all_curves = load_all_loss_curves(Path("\\\\wsl$/Ubuntu/home/ralf/autoencodix/reports"), target_column)
    all_curves = load_all_loss_curves(Path("/home/ralf/autoencodix/reports"), target_column)

    print(all_curves.shape)

    # Find convergence epochs for all runs
    convergence_epochs = np.array([find_convergence_epoch(curve, threshold) for curve in all_curves])

    # Compute statistics
    mean_convergence = np.mean(convergence_epochs)
    median_convergence = np.median(convergence_epochs)

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(all_curves.T, color='gray', alpha=0.3)
    plt.axvline(mean_convergence, color='red', linestyle='--', label=f'Mean Convergence Epoch: {mean_convergence:.1f}')
    plt.axvline(median_convergence, color='blue', linestyle='--',
                label=f'Median Convergence Epoch: {median_convergence:.1f}')
    plt.xlabel('Epoch')
    plt.ylabel(target_column)
    if target_column == 'train_total_loss':
        plt.ylim(0, 2e3)
    elif target_column == 'valid_total_loss':
        plt.ylim(0, .2e8)
        None
    else:
        plt.ylim(-.5, .8)  # train_r2, valid_r2
    plt.title('Loss Curves and Convergence Points')
    plt.legend()
    plt.show()

    print(f"Mean convergence epoch: {mean_convergence:.1f}")
    print(f"Median convergence epoch: {median_convergence:.1f}")

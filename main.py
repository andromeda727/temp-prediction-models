import torch
from data import process_data
from pathlib import Path
from models.simple_MLP import SimpleMLP
from train.train import train
from pathlib import Path
import numpy as np
from lightning.fabric import seed_everything
import einops

def main():
    # set a manual, constant seed of 42 for most random number generators (numpy, torch, etc.)
    seed=42
    seed_everything(seed, workers=True)

    # Verify the installation and check the version
    print(torch.__version__)

    # Check if CUDA (GPU support) is available
    print(torch.cuda.is_available())
    
    root = Path(__file__).resolve().parent
    path_raw = root / "data" / "jena_climate_2009_2016.csv"
    path_processed = root / "data" / "jena_climate_data_processed.npz"

    process_data.create_processed_data(path_raw, path_processed, False)

    # load processed data from file (npz)
    with np.load(path_processed) as data:
        X_train = torch.tensor(data['X_train'], dtype=torch.float32)
        y_train = torch.tensor(data['y_train'], dtype=torch.float32)
        X_val = torch.tensor(data['X_validate'], dtype=torch.float32)
        y_val = torch.tensor(data['y_val'], dtype=torch.float32)
        X_test = torch.tensor(data['X_test'], dtype=torch.float32)
        y_test = torch.tensor(data['y_test'], dtype=torch.float32)
        print(f"Days: train={len(X_train)}, validation={len(X_val)}, test={len(X_test)}")

    print(f'Training data shape: {X_train.shape}\n')
    input_size = X_train.shape[1] * X_train.shape[2]
    # flatten along day entries + features axis (144*12)
    X_train_mlp, X_val_mlp, X_test_mlp = einops.rearrange(X_train, 'd e f -> d (e f)'), einops.rearrange(X_val, 'd e f -> d (e f)'), einops.rearrange(X_test, 'd e f -> d (e f)')
    model = SimpleMLP(input_size, 2)
    train(epochs=100, model=model, xdata=X_train_mlp, ydata=y_train,
          scaler=None, loss_func=1, opt_method=2, lr=0.001, diagnostics=False)

    # calculate validation and test MSE and report findings
    model.eval()
    with torch.no_grad():
        val_mse = torch.nn.functional.mse_loss(model(X_val_mlp), y_val).item()
        predictions = model(X_test_mlp)
        test_mse = torch.nn.functional.mse_loss(predictions, y_test).item()
        test_mae = (predictions - y_test).abs().mean(dim=1)
    print(f"Validation MSE: {val_mse:.4f} degC^2")
    print(f"Test MSE: {test_mse:.4f} degC^2")
    print(f"Test MAE: minimum={test_mae.min().item():.4f} degC, maximum={test_mae.max().item():.4f} degC")
    print("First 5 test predictions [min, max] in degC:")
    for i in range(min(5, len(y_test))):
        pred_min, pred_max = predictions[i].tolist()
        actual_min, actual_max = y_test[i].tolist()
        print(f"  {i + 1}: predicted [{pred_min:.2f}, {pred_max:.2f}]; actual [{actual_min:.2f}, {actual_max:.2f}]")

if __name__ == "__main__":
    main()
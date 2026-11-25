import torch
from data import process_data
from pathlib import Path
from models.standard_MLP import StandardMLP
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

    def validate(model, val_x, val_y) -> tuple[float, float, torch.Tensor]:
        '''Run validation on model using given validation X and y.
        Returns a tuple containing MSE loss in degC^2, MAE loss in degC, and actual predicted values as a torch tensor respectively'''
        model.eval()
        with torch.no_grad():
            predictions = model(val_x)
            val_mse = torch.nn.functional.mse_loss(model(val_x), val_y).item()
            val_mae = torch.nn.functional.l1_loss(model(val_x), val_y).item()
        return (val_mse, val_mae, predictions)

    
    def test(model, test_x, test_y) -> tuple[float, float, torch.Tensor]:
        '''Run test on model using given test X and y.
        Returns a tuple containing MSE loss in degC^2, MAE loss in degC, and actual predicted values as a torch tensor respectively'''
        model.eval()
        with torch.no_grad():
            predictions = model(test_x)
            test_mse = torch.nn.functional.mse_loss(model(test_x), test_y).item()
            test_mae = torch.nn.functional.l1_loss(model(test_x), test_y).item()
        return (test_mse, test_mae, predictions)

    # flatten along day entries + features axis (144*12)
    X_train_mlp, X_val_mlp, X_test_mlp = einops.rearrange(X_train, 'd e f -> d (e f)'), einops.rearrange(X_val, 'd e f -> d (e f)'), einops.rearrange(X_test, 'd e f -> d (e f)')

    
if __name__ == "__main__":
    main()
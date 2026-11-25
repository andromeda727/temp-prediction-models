import torch
from data import process_data
from pathlib import Path
from models.standard_MLP import StandardMLP
from models.standard_transformer import SETN
from models.custom_cnn import CustomCNN
from models.custom_transformer import CETN
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

    check, scaler = process_data.create_processed_data(path_raw, path_processed, False)

    assert check
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
    # input_size = X_train.shape[1] * X_train.shape[2] = 144 * 12

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

    # initialize models
    smlp = StandardMLP(144*12, 16, 2)
    setn = SETN(12, 256, 1024, 16, 32, 2)
    ccnn = CustomCNN(144, 12, 8, 2)
    cetn = CETN(12, 256, 1024, 16, 32, 2)

    # train models
    # we only have ~2000 training examples, so have to be very careful of overfitting, especially with transformers (>4,000,000 params)
    smlp_train_error_norm, smlp_train_error_raw = train(100, smlp, X_train_mlp, y_train, scaler, opt_method=2, diagnostics=False)

    setn_train_error_norm, setn_train_error_raw = train(10, setn, X_train, y_train, scaler, opt_method=2, diagnostics=False)

    ccnn_train_error_norm, ccnn_train_error_raw = train(25, ccnn, X_train, y_train, scaler, opt_method=2, diagnostics=False)

    cetn_train_error_norm, cetn_train_error_raw = train(10, cetn, X_train, y_train, scaler, opt_method=2, diagnostics=False)


    # validate models
    smlp_val_mse, smlp_val_mae, smlp_preds = validate(smlp, X_val_mlp, y_val)

    setn_val_mse, setn_val_mae, setn_preds = validate(setn, X_val, y_val)

    ccnn_val_mse, ccnn_val_mae, ccnn_preds = validate(ccnn, X_val, y_val)

    cetn_val_mse, cetn_val_mae, cetn_preds = validate(cetn, X_val_mlp, y_val)

    ## TODO DO NOT TEST YET! CHECK VALIDATION RESULTS AND TWEAK MODELS AS NEEDED FIRST.
    
if __name__ == "__main__":
    main()
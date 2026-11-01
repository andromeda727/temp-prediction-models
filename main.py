import torch
from data import process_data
from pathlib import Path
# from lightning.fabric import seed_everything

def main():
    # Verify the installation and check the version
    print(torch.__version__)

    # Check if CUDA (GPU support) is available
    print(torch.cuda.is_available())
    

    path_raw = Path("./jena_climate_2009_2016.csv")
    path_processed = Path("jena_climate_data_processed.csv")

    process_data.create_processed_data_csv(path_raw, path_processed, True)

if __name__ == "__main__":
    main()
# Persistent Belief-Guided HARQ

## Environment

Use the following environments. Retrieve the weights after cloning:

```powershell
git lfs install
git lfs pull
```

### Kodak

```powershell
conda create -n core-kodak python=3.12 -y
conda activate core-kodak
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements_kodak.txt
```

### Token Query Experiment

```powershell
conda create -n core-token-query python=3.9 -y
conda activate core-token-query
python -m pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r requirements_erasure.txt
```

Run the following commands from the repository root. Replace `KODAK_DIRECTORY` with the directory containing `kodim01.png` through `kodim24.png`, and `CELEBA_HQ_DIRECTORY` with the directory containing the images listed in the supplied image manifest. Each output directory must be new.

## Kodak: AWGN Channel

```powershell
conda activate core-kodak
python run_test.py r025 --dataset Kodak --kodak-root "KODAK_DIRECTORY" --channels awgn --retransmissions 3 --device cuda --output "results/kodak_awgn"
```

## Kodak: Rayleigh Channel

```powershell
conda activate core-kodak
python run_test.py r025 --dataset Kodak --kodak-root "KODAK_DIRECTORY" --channels rayleigh --retransmissions 3 --device cuda --output "results/kodak_rayleigh"
```

## Token Query Experiment

```powershell
conda activate core-token-query
python run_test.py token-query --celeba-hq-root "CELEBA_HQ_DIRECTORY" --erased 0 4 8 12 16 --retransmissions 4 --trials 5 --device cuda --output "results/token_query"
```

## Code and Weights Availability

This repository currently provides a subset of the code and pretrained weights. The complete code and all pretrained weights will be publicly released after the paper is published.

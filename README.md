# Coding Task

## Project structure

```sh
.
├── dat/  # Default directory for dataset
├── doc/  # Assignment documentation
├── hpc/  # Slurm-scripts for high performance computing
├── results/  # Results task 2
│   └── task2_finetuned/  # Fine-tuned model for hand-in (Task 2) 
│       └── best_model.pt  # Trained model from the two augmentations
│       └── test_logits_best_model.npy  # Saved logits on test split
│       └── training_history_**.png  # Training history for each augmentation
│       └── training_results.json  # JSON dump for training history on both augmentation
│       └── test/  # Directory which is created when reproduction script is run
│   └── task3_finetuned/  # Fine-tuned model for hand-in (Task 3) 
│       └── best_model.pt  # Trained model from the two augmentations
│       └── test_logits_best_model.npy  # Saved logits on test split
│       └── training_history_**.png  # Training history for each augmentation
│       └── training_results.json  # JSON dump for training history on both augmentation
│       └── test/  # Directory which is created when reproduction script is run│   └── run_YYYYMMDD_hhmmss/  # New run of fine-tuning script 
└── src/  # Source code for task 1 to 3
    ├── task1/
    ├── task2/
    ├── task3/
    └── util/
```

## Requirements

- Python 13.13 or higher
- Git LFS installed

## Getting Started

- Create a virtual environment `python -m venv .venv`
- Activate virtual environment `source .venv/bin/activate`
- Install requirements `pip install -r requirements.txt`
- Download zipped data files (`EuroSAT_MS.zip` and `EuroSAT_RGB.zip`) to a `dat` directory of your choice. This `dat` directory must be provided as argument when running the script (see below).

## Run

For each task, activate your virtual environment, first, and set active working directory to project root.

- Activate virtual environment `source .venv/bin/activate`
- Set working directory to project root

### Task 1

Unzip raw files and split data into train, test, val set. Split files and classname-index associations are saved to `DAT_DIR`.

```sh
python -m src.task1.split_data -D <ABSOLUTE_PATH_TO_DAT_DIR_>
```

### Task 2

#### Fine tune model

Fine-tune model for RGB data. When script is run, a new directory named `run_YYYYMMDD_hhmmss` is created and hyper parameters from `constants.py` are used for training.

```sh
python -m src.task2.fine_tune -D <ABSOLUTE_PATH_TO_DAT_DIR_>
```

#### Test saved model for reproducibility and perform visual ranking check

Load the saved model at `./results/task2_finetuned/best_model.pt`. Runs predictions on model on test split and prints results to terminal. Creates images for top/bottom 5 best scoring models for 3 classes.

```sh
python -m src.task2.reproduce -D <ABSOLUTE_PATH_TO_DAT_DIR_>
```

### Task 3

#### Fine tune model

```sh
python -m src.task3.fine_tune -D <ABSOLUTE_PATH_TO_DAT_DIR_>
```

#### Test saved model from task3

```sh
python -m src.task3.reproduce -D <ABSOLUTE_PATH_TO_DAT_DIR_>
```

## High Performance Computing
### Preparation
1. Create a `/dat` directory in your `/work` folder and copy the `.zip` files into it.
2. Copy the `requirements.txt` file as well as the `/src` and `/hpc` folders as well.
3. Create a conda environment named `dl_coding_task`, activate it, and run `pip -r install requirements.txt`

### Run

#### Task 1

```sh
sbatch hpc/task1.slurm
```

### Task 2
#### Fine-Tune
```sh
sbatch hpc/task2_1.slurm
```
#### Reproduce
```sh
sbatch hpc/task2_2.slurm
```

### Task 3
#### Fine-Tune
```sh
sbatch hpc/task3_1.slurm
```
#### Reproduce
```sh
sbatch hpc/task3_2.slurm
```

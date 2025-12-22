# Coding Task

## Project structure

```sh
.
├── dat/  # Default directory for dataset
├── doc/  # Assignment documentation
├── results/  # Results task 2
│   └── run_YYYYMMDD_hhmmss/  # One run of fine-tuning script 
│       └── test/  # Files for reproduction and testset predictions
└── src/  # Source code for task 1 to 3
    ├── task1/
    ├── task2/
    └── util/
```

## Requirements

- Python 13.13 or higher

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

Unzip raw files and split data into train, test, val set:

```sh
python -m src.task1.split_data -d <ABSOLUTE_PATH_TO_DAT_DIR_>
```

## Task 2

### Fine tune model

Fine-tune model

```sh
python -m src.task2.fine_tune -d <ABSOLUTE_PATH_TO_DAT_DIR_>
```

### Test saved model for reproducibility and perform visual ranking check

```sh
python -m src.task2.reproduce -d <ABSOLUTE_PATH_TO_DAT_DIR_>
```

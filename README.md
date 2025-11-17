# Advanced Machine Learning Challenge: Master of Stitching 🤖🪡

This repo contains all the code necessary to run the project.  
This repo contains both approch we used in the final two submissions.

## Setup
To set up the environment, run:
```bash
uv sync
```
This will create a virtual environment and install all the required packages.

After, you have to download the dataset from [here](https://drive.google.com/file/d/1pPF7D_hhDZe5fKwYbjqjM17GAhdOEjXd/view?usp=drive_link), unzip the archive and place the files in the `data/` directory.  
However, you can also setup the dataset manually by:
- Downloading the dataset from the competition page and put it inside the `data/` directory.
- Download CoCo 2017 dataset from [here](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset) and put it inside the `data/coco2017` directory.
- Running the preprocessing script:
```bash
uv run src/coco/main.py
```
This will create the .npz files for CoCo experiment.

## Running Experiments
To run CoCo experiment, use:
```bash
uv run coco.py
```
To run Ensemble experiment, use:
```bash
uv run ensemble.py
```
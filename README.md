# HOLMES: HOLonym-MEronym based Semantic inspection

![](images/OverviewNew.png?raw=true)

## Introduction
This repository contains the code for running the experiments of HOLMES: HOLonym-MEronym based Semantic inspection. <br />

This code includes:

* Code to run HOLMES on VGG16.
    The script `run_pipeline.py` runs all the HOLMES pipeline steps.
    
* Code to run only one of the HOLMES pipeline steps at a time.
    The scripts `run_scraping_only.py`, `run_training_only.py` and `run_explanations_only.py` respectively run the HOLMES scraping, training and explanations step.
    
* Code to evaluate the HOLMES explanations.
    The script `run_evaluation.py` runs the evaluation of the part-based explanations and collects the related measurements.

* Code to extract and organize the ImageNet validation samples into subfolders named to be compatible with the HOLMES experiments.
    The script `extract_and_rename.py` extracts the ImageNet validation samples and organizes them into subfolders named with the associated ImageNet class (i.e., Arabian camel, airliner, etc.).
    
## Installation

* This code was tested and run in the Python3 (v3.8.2) environment. The required libraries and dependecies were installed with the pip package manager, version 21.0.1.
* Prior to installation, make sure to have a version equal or greater than 21.0.1, by executing the command:

```
    python -m pip -V
```

* In case your pip version is lesser than 21.0.1, upgrade it by executing the command:

```
    python -m pip install --upgrade pip
```

* Execute the setup script, which will install all the required libraries and dependecies for running the HOLMES experiment.

```
    setup.sh
```

* Then, if your Operating System is Unix based, execute the following commands (skip if on Windows OS):

```
    sudo apt-get update
    sudo apt-get install chromium-chromedriver
    sudo python unix_additional_setup.py install
```

* Download the ILSVRC 2012 ImageNet validation set (50,000 validation images) from [here](https://image-net.org/challenges/LSVRC/2012/index.php#) and put the downloaded archive (ILSVRC2012_img_val.tar) into the `Holonyms` folder.
* Execute the script `extract_and_rename.py` from the `Holonyms` folder.

```
    python extract_and_rename.py
```

## Run

* Run the HOLMES algorithmic pipeline. You can run `run_pipeline.py` to execute the complete HOLMES pipeline.  

```
    python run_pipeline.py
```

* Alternatively, you can run `run_scraping_only.py`, `run_training_only.py` and `run_explanations_only.py` (in this order) to execute each step separately.

```
    python run_scraping_only.py
    python run_training_only.py
    python run_explanations_only.py
```

* IMPORTANT: in case you want to skip the first two steps and execute either the explanations step or the evaluation, follow the instruction at the end of the **Report** Section and then execute the `download.py` script with the following arguments:

```
    python download.py https://drive.google.com/drive/folders/1UkJEN8tcIf9kODo5j-jFx8tdiU2URlxa?usp=sharing Checkpoints
    python download.py https://drive.google.com/drive/folders/14w6ohQYxBg7_9FL4E7a2iOYfeYHgGP1M?usp=sharing Checkpoints
    python download.py https://drive.google.com/drive/folders/1ETY0igoirK9wWMgpCP9gsXexyn21ZYNP?usp=sharing Checkpoints
```

In this way you will obtain the same trained models weights obtained during the HOLMES experiment, and they will be located in the `Checkpoints` folder.

* After the execution of the whole pipeline, run `run_evaluation.py` to collect the insertion/deletion curves measurements for a quantitative evaluation of the per-part HOLMES generated explanations.

```
    python run_evaluation.py
```

## Report
* At the end of the execution of each step, a json file will be generated in the main directory, containing the measurements which are needed for starting the next steps. For example, after having executed the scraping, the training and the explanations steps, you will have:

```
    scrape.json
    train.json
    exp.json
```

* Similarly, after having executed the evaluation script, you will have:

```
    curves.json
```

* IMPORTANT: In case you want to skip one of the steps and execute one of next ones, copy into the main directory the json files associated to the previous steps from the `Results` folder, which already contains the final measurements recorded during the HOLMES experiment at the end of each step.

## Notebooks 
The Notebooks folder includes:

* `Meronyms_Extraction.ipynb` displays the Meronyms Extraction procedure and how the Holonyms and the Meronyms for the HOLMES experiment where selected.
* `Results_analysis.ipynb` displays the results obtained during the HOLMES whole experiment.
* `Explanations_examples(Animals).ipynb` shows a subset of the explanations generated for the Animals classes.
* `Explanations_examples(Man-made).ipynb` shows a subset of the explanations generated for the Man-made object classes.

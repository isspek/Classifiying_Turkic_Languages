## Dataset

Dataset is subset of [old newspapers](https://www.kaggle.com/alvations/old-newspapers) dataset at Kaggle.

We choose the following Turkic langs for our analysis:
  - Turkish
  - Azerbaijani 
  - Uzbek
  
 
## Installation
 
We use Python 3.8, and create an virtual environment.

    `virtualenv -p /usr/local/bin/python3.8 venv`
    `source venv/bin/activate`
    
Install the requirements after activating the virtual environment:
    `pip install -r requirements.txt`
    
Run the `install_zemberek.sh` bash script for installing python wrapper of Zemberek Turkish NLP tool.
    
## Reproducing

Run the `dataset.sh` bash for the data set splits.

Run the `experiment.sh` bash for training and evaluating models.

`predict.py` is for saving the predictions by the best model using split by random seed 42.

`stats.py` is for data set statistics.

`notebooks/error_analysis.ipynb` is for presenting error analysis.


!!!Report will be shared in the future. 



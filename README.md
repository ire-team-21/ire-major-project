# IRE Major Project 

## Application   
The code for the application can be found in the ```app``` folder. To run the application, clone the repository and run  
```
streamlit run app.py
```

## Models 
The trained models are present in the ```Models``` directory. Due to GitLab storage constraints, only some of the models have been stored here. The rest can be found [here](https://duckduckgo.com)

## Embeddings  
The embeddings used as features in the ML models are present in the ```Embeddings``` directory  

## Dataset 
The datasets used are present in the ```data``` directory. Within it are three directories. The ```labels``` directory contains the mapping from the index labels to the author names for each dataset. The ```original``` directory contains the complete datasets. The ```pruned``` directory contains the test, train, and val splits for each dataset.  The models were trained on ths.   

## Source code 
The source code for the implemented models, baseline and scripts used to build the news authors dataset are present in the ```src``` directory. Within it, each approach is present in its own directory.  
The baseline approach is included in the ```baseline``` directory.  
The ```util_scripts``` directory contains the scraping and hydrating tools for building the news authors dataset.   
The ```logs``` folder contains the training logs of the deep learning models for verification of claimed results.   
The results can be reproduced by using the ``nn_trainer.py``, ```nn_trainer_style.py```, and ```transformer_trainer.py``` scripts.  

## Note 
The codebase was originally stored in GitHub at https://github.com/aditya-hari/ire-major-project/. However the extremely stringent storage constraints of GitHub made it infeasible to use this for storing the models. The project was migrated to GitLab later. 



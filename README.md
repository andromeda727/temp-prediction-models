# temp-prediction-models

In this repository, we explore the effectiveness of various different neural networks at predicting next-day maximum and minimum temperature using the Jena Climate Dataset from Kaggle: https://www.kaggle.com/datasets/mnassrib/jena-climate. 

# Models

For now, we plan to try 4 different neural network models:

Standard MLP

Standard Transformer (Sinusoidal, Standard MLP Head)

Custom CNN (multi-layered with non-standard activations, MLP at end)

Custom Transformer (Learned Positional Encodings, CNN->MLP head)

See more details about each model and their architectures within their respective python files in "models".

# Repository Structure

This repository is organized into a few different folders: "data", "models", "train", and "test", which contain python code pertaining to each of those steps in our ML process. 

"data" contains the original csv file, as well as a restructured/processed data csv for use with the models, and the python code used to create that processed data.

"models" contains each of our NN models, described above.

"train" contains python code for training our models.

"test" contains python code for perofrming validation and tests on the trained models.

In the root of the repo, there is also a main.py script responsible for using all of the above code to complete the ML process from start to finish.
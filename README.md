# Optimizing a ML model pipeline in Azure

This project is part of the Udacity Azure ML Nanodegree. 
In this project, we build and optimize an Azure ML pipeline using Hyperdrive for parameter tuning in Python SDK and a provided Scikit-learn model. 
This model is then compared to an Azure AutoML run.

# Dataset

The dataset is a bank marketing dataset including personal informationo of individuals for marketing campaigns.
The data includes details such as age, marital status, job, etc. of the individuals.

# Summary of the model performances

The best model performance achieved with Hyperdrive: 
  - 0.9173 of accuracy with the use of Logistic Regression algorithm

The best model performance achieved with Azure Auto ML:
  - 0.9169 of accuracy with the use of VotingEnsemble algorithm

# Pipeline architecture
The pipeline was created in a Jupyter notebook.

The pipeline include the following steps:
- Create a workspace
- Create a CPU cluster with a given VM size for running the ML trainings
- Set run parameters:
  - Parameters sampling
  - Policy
  - Estimator
  
- The estimator is using the train.py, which does the following:
      - Get the dataset from a given url,
      - Get the data cleaned (steps documented in the function cleandata),
      - Split the data into train and test dataset,
      - Fit the data to the model (Logistic Regression was the given algorithm to use)

- Create the Hyperdriveconfig with the estimator, hyperparameter sampler and policy
- Run the Hyperdrive
- Log the model performances (Accuracy was used as performance metric)
- The best model performance was found and the best model parameters have been saved


# Parameter sampling

RandomParameterSampler was used as based on experience the random sampling provides good results without the need of screening the entire grid space.
It is cost and time efficient way for finding the optimal hyperparameters.
I had the following hyperparameters to fine-tune: 
- C, which stands for inverse of regularization strength : uniform search applied between 0 and 100
- max_iter, which stands for the max number of iterations until it converges : possible values of 50, 75, 100, 150

# Policy

I applied an early termination policy with using BanditPolicy.
It included 2 parameters:
- slack_factor, meaning the slack allowed with respect to the best performing training run. This parameter was set to 0.1, meaning any run whose best metric is less than (1/(1+0.1) or 91% of the best performing run will be terminated.

- evaluation interval, which shows the frequency of applying the policy between the performance measurements. This was set to 1, meaning every time an accuracy value was calculated, the policy was applied.

# AutoML

The best model found by the Hyperdrive was compared against best model found by AutoML.
The AutoML was run over 30minutes and was running 41 different algorithms.
Among the algorithms we had MaxAbsScaler LightGBM, MaxAbsScaler XGBoostClassifier, VotingEnsemble, etc.

To retrieve the best AutoMl Model and its parameters, the following code was used:

    # Retrieve your best automl model.
    best_run = remote_run.get_best_child()
    best_run.get_details()

The best model performance was achieved by the VotingEnsemble algorithm, with the following parameter settings:
- min_samples_leaf = 0.01
- min_samples_split= 0.2442
- n_jobs=1
- n_estimators=10

The best model found by AutoML is not automatically saved. For saving the model the function .register_model() was used and the model could be saved in the desired folder:

    #save best automl model
    best_run.register_model(model_name='automl_best_model.pkl',model_path='outputs/')


# Comparison of model performances

We achieved comparably similar performance results by using Hyperdrive and AutoML for ML pipeline optimizations.
The Hyperdrive only slightly outperformed the AutoML (0.9173 vs. 0.9169)
AutoML provides a very convenient and easy way for quickly finding a relatively good ML model with right hyperparameters to our business problem. 
It is a great tool we can use for producing first results in a very short time.


# Further improvement possibilities

The model performance can be presumably improved if we consider the following possibilities:
We increase the size of dataset, we test further algorithms and hyperparameters with hyperdrive
and/or apply the AutoML best model and try to fine-tune it further with Hyperdrive



























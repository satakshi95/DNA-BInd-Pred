# DNA-BInd-Pred

**Prediction of DNA Binders:-**

In this assignment, we are building models using machine learning techniques to predict the given protein sequences if they are DNA binding or not.

- We have the training dataset containing 3049 protein sequences with labels (output) if they are DNA binding or non DNA.
- **Feature Extraction** - We created different inputs/features for the proteins to be used in machine learning techniques.

Features like amino acid composition, dipeptide composition, atom composition, bond composition, Shannon entropy, amino acid length were calculated using **Pfeature** software and coding via **Biopython**. These features were saved in different csv file and then merged to have a single csv file to be used in training the model.

Feature Extraction was done in the following ways:-

- Installing protlearn and biopython.
- Created method csv\_to\_fasta which will convert csv files to fasta format.
- Imported remove\_unnatural to remove unnatural amino acid residues from the protein sequence (like X).
- Length module was imported from features library of protlearn to find the length of the amino acid sequence.
- Aac was imported to calculate the amino acid composition (feature) of the protein sequence.
- Ngram module was imported to calculate the dipeptide composition.
- Imported atc to find the atom and bond composition.
- Aaindex1 was imported to find the amino acid index.
- Finally all the descriptors (features) were displayed.

- Feature selection was done using **BorutaPy** by importing it from Boruta.
- From training data, we split separate training and testing data using the train\_test\_split module of sklearn library.

This training (x\_train) and testing data (x\_test) was also preprocessed and standardized using StandardScaler library to bring the feature values around the mean.

- Support Vector classifier was used with rbf kernel to build the model and do the prediction. Then the accuracy was checked using confusion matrix (72.3%) and cross validation score.
- Various other machine learning techniques were also used like decision tree, random forest, XGBClassifier,Nearest Neighbors, SGD Classifier, Gaussian Naïve Bayes, Neural Network, Gaussian Process Classifier out of which highest accuracy was received on **XGBoost** (73.93%) and highest cross validation accuracy was received on **SVM classifier** (72.49%) on the training dataset.
- Validation dataset consists of 1071 protein sequences and we need to find labels- DNA binding or non DNA binding proteins.
- Features were extracted for the Validation dataset also like – atom composition, dipeptide composition, Shannon entropy, amino acid composition, bond composition and amino acid length.
- In the validation dataset, we split the data into training and testing data and features were standardized by StandardScaler library.
- Feature selection was done using the BorutaPy (same as we did for training dataset).
- Random Forest Classifier method was then used for training the model and for predicting. Accuracy was received 74.1% and cross validation accuracy as 71.63%.
- Support Vector classifier with rbf kernel which yielded the accuracy as 71.8% and cross validation accuracy as 73.31%.
- Other machine learning techniques were also tried like Decision Tree, random forest, XGBClassifier, SVM, Nearest Neighbors, SGD Classifier, Gaussian NB, Neural Network, Gaussian Process classifier. Out of which highest accuracy was attained on Gaussian Process classifier (72.29%) and cross validation accuracy as 73.31% on SVM classifier.
- PCA was used for dimensionality reduction and then the final labels were calculated using SVM classifier with cross validation accuracy as 71.07%.
- Final labels were calculated in the form of a dataframe series, showing +1 and -1 values.

1 for DNA binding protein sequences and -1 for non DNA binding protein sequences.

**Dataset Link** : https://www.kaggle.com/c/mlba1/overview/description

**References** :-

https://protlearn.readthedocs.io/en/latest/introduction.html

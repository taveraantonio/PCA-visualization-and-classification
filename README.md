# Principal Component Analysis (PCA)

The code applies Principal Component Analysis (PCA) on a set of images. It is shown what happen if different principal components are chosen as basis for image representation and classification.
This code was developed for the Machine Learning and Artificial Intelligence Course held at the Politecnico di Torino 

## What the code does
Here is the brief list of what the code does:
1. Download and load the provided subset of PACS dataset; setup your programming environ-ment accordingly your needings.
2. Chooses one image and shows what happens to the image when it is re-project it with only first 60 PC, first 6 PC, first 2 PC, last 6 PC. 
3. Using scatter-plot, visualizes the dataset projected on first 2 PC, with the 3&4 PC, and with 10&11. 
4. Classifies the dataset (divided into training and test set) using a Naive Bayes Classifier in those cases: unmodified images, images projected into first 2PC, and on 3&4 PC. 

An in-depth description of what the code does, is present as comment of the code itself. 
An detailed analysis of the work and of the results is described in the Report file.

## How it works
To run the code you have first to unzip the **PACS_homework** folder; then you can run the code with a Python environment (like Spyder) or directly from the terminal (inside the root folder of the project) like this: 
```
> python /src/PCA_source_code.py
```


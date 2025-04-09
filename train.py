import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# We will now save our model using the skops Python package.
# This will help us save both the scikit-learn pipeline and model.
import skops.io as sio

# By using pipelines, we can ensure reproducibility, modularity, and clarity in our code.
from sklearn.pipeline import Pipeline

# We will build a processing pipeline using ColumnTransformer,
# which will convert categorical values into numbers, fill in missing values,
# scale the numerical columns
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# After that, we'll create a training pipeline that will take the transformed data and train a random forest classifier.
# Finally, we'll train the model.
from sklearn.ensemble import RandomForestClassifier

# Evaluate the performance of the model by calculating both the accuracy and F1 score.
from sklearn.metrics import accuracy_score, f1_score

from sklearn.model_selection import train_test_split


"""
This will be the standardized training script that will run in CI workflow whenever there is a change 
in the data or code.
"""


if __name__ == '__main__':
    # Loading the Dataset
    drug_df = pd.read_csv('data/drug200.csv')
    drug_df = drug_df.sample(frac=1)
    print(drug_df.head())

    X = drug_df.drop('Drug', axis=1)
    y = drug_df['Drug']

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=125)

    # Machine Learning Pipelines
    cat_col = [1, 2, 3]
    num_col = [0, 4]

    transform = ColumnTransformer(
        [
            ('encoder', OrdinalEncoder(), cat_col),
            ('num_imputer', SimpleImputer(strategy='median'), num_col),
            ('num_scaler', StandardScaler(), num_col)
        ]
    )

    pipe = Pipeline(
        steps=[
            ('preprocessing', transform),
            ('model', RandomForestClassifier(n_estimators=100, random_state=125))
        ]
    )

    pipe.fit(X_train, y_train)

    # Model Evaluation
    prediction = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    f1 = f1_score(y_test, prediction, average='macro')

    print(f'Accuracy: {accuracy:.2f}%, F1: {f1:.2f}')  # Our model has performed exceptionally well.

    # Create the metrics file and save it in the Results folder.
    with open('results/metrics.txt', 'w') as outfile:
        outfile.write(f'Accuracy: {accuracy:.2f}, F1: {f1:.2f}\n')

    # We will then create the confusion matrix and save the image file into the Results folder.
    cm = confusion_matrix(y_test, prediction, labels=pipe.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
    disp.plot()
    plt.savefig('results/model_results.png')

    # Saving the Model
    sio.dump(pipe, 'model/drug_pipeline.skops')

    # You can just load the entire pipeline, and it will work out of the box without processing
    # your data or making edits to the code.
    # print(sio.load('model/drug_pipeline.skops', trusted=['numpy.dtype']))

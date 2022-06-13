"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any Loan of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

def preprocessing(df):
    df.drop(['ID', 'ZIP Code'], axis = 1, inplace = True)
    df['Experience'] = df['Experience'].replace(to_replace=[-1, -2, -3], value=np.nan)
    cols = set(df.columns)
    cols_numeric = set(['Age', 'Experience', 'Income', 'CCAvg', 'Mortgage'])
    cols_categorical = list(cols - cols_numeric)
    for x in cols_categorical:
        df[x] = df[x].astype('category')
    data_num = df.select_dtypes(include='number')
    data_cat = df.select_dtypes(include='category')
    df = pd.concat([data_num, data_cat], axis=1)
    return df

# Split the dataframe into test and train data
def split_data(df):
    X = df.drop('Y', axis=1)
    y = df['Y']
    X_dummied = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X_dummied, y, test_size=0.2, random_state=0)
    knn = KNNImputer(n_neighbors=3)
    X_train_imp = pd.DataFrame(knn.fit_transform(X_train), columns = X_train.columns)
    X_test_imp = pd.DataFrame(knn.transform(X_test), columns = X_test.columns)

    sc = StandardScaler()
    X_train_scaled = pd.DataFrame(sc.fit_transform(X_train_imp), columns=X_train_imp.columns)
    X_test_scaled = pd.DataFrame(sc.transform(X_test_imp), columns=X_test_imp.columns)

    data = {"train": {"X": X_train_scaled, "y": y_train},
            "test": {"X": X_test_scaled, "y": y_test}}
    return data



# Train the model, return the model
def train_model(data, rf_args):
    rf_model = RandomForestClassifier(**rf_args)
    rf_model.fit(data["train"]["X"], data["train"]["y"])
    return rf_model


# Evaluate the metrics for the model
def get_model_metrics(model, data):
    preds = model.predict(data["test"]["X"])
    score = accuracy_score(preds, data["test"]["y"])
    f1 = f1_score(preds, data["test"]["y"], average = 'micro')
    precision = precision_score(preds, data["test"]["y"], average='micro')
    recall = recall_score(preds, data["test"]["y"], average='micro')
    metrics = {"score": score, "F1_score": f1, "Precision": precision, "Recall": recall}
    return metrics


def main():
    print("Running train.py")

    # Define training parameters
    rf_args = {"n_estimators": 50, "criterion": 'gini',"max_depth": 5}

    # Load the training data as dataframe
    data_dir = "data"
    data_file = os.path.join(data_dir, 'personal_loan.csv')
    train_df = pd.read_csv(data_file)

    preprocessed = preprocessing(train_df)

    data = split_data(preprocessed)



    # Train the model
    model = train_model(data, rf_args)

    # Log the metrics for the model
    metrics = get_model_metrics(model, data)
    for (k, v) in metrics.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()

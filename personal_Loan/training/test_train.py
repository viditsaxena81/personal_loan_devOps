import numpy as np
from personal_Loan.training.train import train_model, get_model_metrics


def test_train_model():
    X_train = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    y_train = np.array([1, 0, 1, 1, 0, 0])
    data = {"train": {"X": X_train, "y": y_train}}

    rf_model = train_model(data, {"n_estimators": 50, "criterion": 'gini',"max_depth": 5})

    preds = rf_model.predict([[1], [2]])
    np.testing.assert_almost_equal(preds, [1, 0])


def test_get_model_metrics():

    class MockModel:

        @staticmethod
        def predict(data):
            return ([0, 1])

    X_test = np.array([3, 4]).reshape(-1, 1)
    y_test = np.array([0, 1])
    data = {"test": {"X": X_test, "y": y_test}}

    metrics = get_model_metrics(MockModel(), data)

    assert 'score' in metrics
    acc = metrics['score']
    np.testing.assert_almost_equal(acc, 1)

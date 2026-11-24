from model_pipeline import prepare_data, train_model, evaluate_model


def test_prepare_data_shapes():
    x_train, x_test, y_train, y_test = prepare_data()
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)
    assert len(x_train) > len(x_test)


def test_train_and_evaluate():
    x_train, x_test, y_train, y_test = prepare_data()
    model = train_model(x_train, y_train)
    assert evaluate_model(model, x_test, y_test) > 0.7

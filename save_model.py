import h2o

def save_model(model, file_name: str) -> None:
    """
    Saves an H2O machine learning model to disk.

    Arguments:
    ----------
    - model : H2OEstimator
        An H2O machine learning model object.
        
    - file_name : str
        A string representing the name of the file to save the model to.

    Returns:
        None
    """
    save_model_path = '/home/stinky/Documents/ntnu/ml/models/'

    h2o.save_model(model=model, path=save_model_path, force=True, filename=file_name)
import pandas as pd

from read_data import ReadData
from concatinate_training_data import ConcatinateTrainingData

class CreateTrainingData:
    """
    Concatinates the training data from all locations and returns it as a DataFrame.
    """
    def __init__(self):
        """
        Initialize a CreateTrainingData instance while using the ReadData and
        ConcatinateTrainingData classes.
        """
        self.data_A = ReadData('A')
        self.data_B = ReadData('B')
        self.data_C = ReadData('C')

        self.train_observed_A = self.data_A.import_train_observed_data()
        self.train_observed_B = self.data_B.import_train_observed_data()
        self.train_observed_C = self.data_C.import_train_observed_data()

        self.train_estimated_A = self.data_A.import_train_estimated_data()
        self.train_estimated_B = self.data_B.import_train_estimated_data()
        self.train_estimated_C = self.data_C.import_train_estimated_data()

        self.A = ConcatinateTrainingData(self.train_observed_A, self.train_estimated_A)
        self.B = ConcatinateTrainingData(self.train_observed_B, self.train_estimated_B)
        self.C = ConcatinateTrainingData(self.train_observed_C, self.train_estimated_C)

        self.training_A = self.A.concatinate_training_data()
        self.training_B = self.B.concatinate_training_data()
        self.training_C = self.C.concatinate_training_data()
    
    def add_target_to_training_data(self) -> None:
        """
        Add the target data to the training data.
        """
        A = ReadData('A')
        B = ReadData('B')
        C = ReadData('C')

        target_A = A.import_target_data()
        target_B = B.import_target_data()
        target_C = C.import_target_data()

        self.training_A['pv_measurement'] = target_A['pv_measurement']
        self.training_B['pv_measurement'] = target_B['pv_measurement']
        self.training_C['pv_measurement'] = target_C['pv_measurement']

    def create_training_data(self):
        """
        Returns a DataFrame with the training data from all locations.
        """
        self.add_target_to_training_data()

        return pd.concat([self.training_A, self.training_B, self.training_C], axis=0)
    
def create_target_data() -> pd.DataFrame:
    """
    Returns a DataFrame with the target data from all locations.
    Index is set to 'time'.
    """
    data_A = ReadData('A')
    data_B = ReadData('B')
    data_C = ReadData('C')

    target_A = data_A.import_target_data()
    target_B = data_B.import_target_data()
    target_C = data_C.import_target_data()

    target_data = pd.concat([target_A, target_B, target_C], axis=0)
    target_data.set_index('time', inplace=True)

    return target_data

def create_test_data() -> pd.DataFrame:
    """
    Returns a DataFrame with the test data from all locations.
    Index is set to 'date_forecast'.
    """
    data_A = ReadData('A')
    data_B = ReadData('B')
    data_C = ReadData('C')

    test_A = data_A.import_test_estimated_data()
    test_B = data_B.import_test_estimated_data()
    test_C = data_C.import_test_estimated_data()

    test_data = pd.concat([test_A, test_B, test_C], axis=0)
    test_data.set_index('date_forecast', inplace=True)

    return test_data
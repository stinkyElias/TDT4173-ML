import matplotlib.pyplot as plt
import read_data as rd

def plot_target_data() -> None:
    """
    Plots the target data for all locations.
    """
    data_A = rd.ReadData('A')
    data_B = rd.ReadData('B')
    data_C = rd.ReadData('C')

    target_A = data_A.import_target_data()
    target_B = data_B.import_target_data()
    target_C = data_C.import_target_data()

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(target_A['time'], target_A['pv_measurement'], label='A')
    ax.plot(target_B['time'], target_B['pv_measurement'], label='B')
    ax.plot(target_C['time'], target_C['pv_measurement'], label='C')

    ax.set_xlabel('Time')
    ax.set_ylabel('Power')

    ax.set_title('Target data')

    ax.grid()
    ax.legend()

    plt.show()
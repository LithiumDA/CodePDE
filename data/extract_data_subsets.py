import h5py
import numpy as np
import os

def work(dataset_path, subset_path, subset_selection):
    # Load data from file
    with h5py.File(dataset_path, 'r') as f:
        # Load the data
        t_coordinate = np.array(f['t-coordinate'])[:-1]  # Keep as is
        x_coordinate = np.array(f['x-coordinate'])  # Keep as is
        u = subset_selection(np.array(f['tensor']))

        # Navier-Stokes data has different structure
        # Vx = subset_selection((f['Vx']))
        # density = subset_selection(np.array(f['density']))
        # pressure = subset_selection(np.array(f['pressure']))

    # Verify shapes
    print(t_coordinate.shape, x_coordinate.shape, u.shape)
    # (201,) (1024,) (100, 201, 1024) for burgers equation

    # Save the subset to a new HDF5 file
    with h5py.File(subset_path, 'w') as f:
        # Create datasets in the new file
        f.create_dataset('t-coordinate', data=t_coordinate)
        f.create_dataset('tensor', data=u)
        f.create_dataset('x-coordinate', data=x_coordinate)

        # Uncomment if you want to save Navier-Stokes specific data
        # f.create_dataset('Vx', data=Vx)
        # f.create_dataset('density', data=density)
        # f.create_dataset('pressure', data=pressure)

    print(f"Subset data saved successfully at {subset_path}!")

if __name__ == '__main__':

    dataset_dir = '../dataset/1D/Burgers/Train'
    test_subset_size = 100
    dev_subset_size = 50
    subset_dir = '../dataset/CodePDE/Burgers'
    if not os.path.exists(subset_dir):
        print(f"Creating: {subset_dir}")
        os.makedirs(subset_dir)
    else:
        print(f"Exist: {subset_dir}")

    for item in os.listdir(dataset_dir):
        full_path = os.path.join(dataset_dir, item)
        if os.path.isfile(full_path):
            print(full_path)

            subset_path = os.path.join(subset_dir, item)
            work(full_path, subset_path, lambda x: x[:test_subset_size])

            development_subset_path = subset_path.replace('.hdf5', '_development.hdf5')
            work(full_path, development_subset_path, lambda x: x[-dev_subset_size:])

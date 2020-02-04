""" Try the sequential inversion framework on synthetic data (conic volcano).

The synthetic data should be located in ./out/.

"""
import os
import numpy as np


def main():
    # Load
    data_folder = "./out/"
    F = np.load(os.path.join(data_folder, "F_synth.npy"))
    reg_coords = np.load(os.path.join(data_folder,"reg_coords_synth.npy"))
    volcano_inds = np.load(os.path.join(data_folder,"volcano_inds_synth.npy"))
    data_coords = np.load(os.path.join(data_folder,"data_coords_synth.npy"))
    data_values = np.load(os.path.join(data_folder,"data_values_synth.npy"))
    ground_truth = np.load(os.path.join(data_folder,"density_synth.npy"))

    # List of cells belonging to the volcano (same as cells_coords in the
    # Stromboli setup). Here we extract them from a regular grid.
    volcano_coords = reg_coords[volcano_inds]
    print("Size of inversion grid: {} cells.".format(volcano_coords.shape[0]))
    print("Number of datapoints: {}.".format(data_coords.shape[0]))
    size = data_coords.shape[0]*volcano_coords.shape[0]*8 / 1e9
    print("Size of Pushforward matrix: {} GB.".format(size))

    # Partition the data.

if __name__ == "__main__":
    main()

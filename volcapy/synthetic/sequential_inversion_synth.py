""" Try the sequential inversion framework on synthetic data (conic volcano).

The synthetic data should be located in ./out/.

"""

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

    # Partition the data.
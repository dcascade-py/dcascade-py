"""
Created on Tue Oct 29 10:58:54 2024

@author: diane
"""

import numpy as np


class Cascade:
    """
    @brief Sediment cascade object.

    A sediment cascade is a volume mobilised from a reach during one time step.

    @param provenance
        Reach from which the cascade is mobilised at this time step
    @param elapsed_time
        Time since the begining of the time step. Is updated as the cascade moves through different reaches.
    @param volume
        The mobilised sediment volume itself. It is a 2d array.
        N_col = n_classes + metadata. N_row = number of layers separated by initial provenance
    """

    def __init__(self, provenance, elapsed_time, volume):

        self.provenance = provenance
        self.elapsed_time = elapsed_time # can contain nans, in case a class has 0 volume
        self.volume = volume # size = n_classes + metadata, to include the original provenance in a first column

        # To be filled during the time step
        self.velocities = np.nan # in m/s
        # Flag to know if the cascade is from external source (default False)
        self.is_external = False


import sys
import os


your_path = os.getcwd()

ae_path = os.path.join(your_path, 'Autoencoder')
sys.path.append(ae_path)

constellations_path = os.path.join(your_path, 'constellations')
sys.path.append(constellations_path)

plot_path = os.path.join(your_path, 'plots')
sys.path.append(plot_path)

transmission_path = os.path.join(your_path, 'Transmission')
sys.path.append(transmission_path)


import sys
import os


your_path = os.getcwd()

constellations_path = os.path.join(your_path, 'constellations')
sys.path.append(constellations_path)

plot_path = os.path.join(your_path, 'plots')
sys.path.append(plot_path)

plot_path = os.path.join(your_path, 'Autoencoder')
sys.path.append(plot_path)


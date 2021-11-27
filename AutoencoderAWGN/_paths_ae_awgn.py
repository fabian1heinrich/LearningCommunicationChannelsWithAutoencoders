import sys
import os


your_path = os.getcwd()

ae_path = os.path.join(your_path, 'Autoencoder')
sys.path.append(ae_path)

plot_path = os.path.join(your_path, 'plots')
sys.path.append(plot_path)

constellation_path = os.path.join(your_path, 'constellations')
sys.path.append(constellation_path)

constellation_path = os.path.join(your_path, 'ChannelGAN')
sys.path.append(constellation_path)

constellation_path = os.path.join(your_path, 'MINE')
sys.path.append(constellation_path)
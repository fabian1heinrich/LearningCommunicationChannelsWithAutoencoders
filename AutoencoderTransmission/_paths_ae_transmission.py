import sys
import os


your_path = os.getcwd()

ae_path = os.path.join(your_path, 'Autoencoder')
sys.path.append(ae_path)

ae_path = os.path.join(your_path, 'Transmission')
sys.path.append(ae_path)

plots_path = os.path.join(your_path, 'plots')
sys.path.append(plots_path)

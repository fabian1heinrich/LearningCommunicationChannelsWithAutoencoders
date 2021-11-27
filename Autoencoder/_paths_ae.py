import sys
import os


your_path = os.getcwd()

ae_func_path = os.path.join(your_path, 'Autoencoder_functionals')
sys.path.append(ae_func_path)

tx_path = os.path.join(your_path, 'Transmission')
sys.path.append(tx_path)

saleh_path = os.path.join(your_path, 'Saleh')
sys.path.append(saleh_path)


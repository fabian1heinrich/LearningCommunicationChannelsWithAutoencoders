import sys
import os


your_path = os.getcwd()

ae_path = os.path.join(your_path, 'Autoencoder')
sys.path.append(ae_path)

ae_tx_path = os.path.join(your_path, 'AutoencoderTransmission')
sys.path.append(ae_tx_path)

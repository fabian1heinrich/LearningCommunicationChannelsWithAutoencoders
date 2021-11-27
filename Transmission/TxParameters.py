class TxParameters():

    def __init__(self,
                 snr,
                 input_backoff,
                 amam_params,
                 ampm_params,
                 rolloff,
                 samples_per_symbol,
                 n_symbols_pulse_shaping):
        super(TxParameters, self).__init__()

        self.snr = snr
        self.input_backoff = input_backoff
        self.amam_params = amam_params
        self.ampm_params = ampm_params
        self.rolloff = rolloff
        self.samples_per_symbol = samples_per_symbol
        self.n_symbols_pulse_shaping = n_symbols_pulse_shaping

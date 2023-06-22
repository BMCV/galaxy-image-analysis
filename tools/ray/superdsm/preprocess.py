from .pipeline import Stage


class Preprocessing(Stage):

    ENABLED_BY_DEFAULT = True

    def __init__(self):
        super(Preprocessing, self).__init__('preprocess',
                                            inputs  = ['g_raw'],
                                            outputs = ['y'])

    def process(self, input_data, cfg, out, log_root_dir):
        return {
            'y': input_data['g_raw']
        }


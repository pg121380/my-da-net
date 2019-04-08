class ModelModule(object):
    '''
    abstract class for a sub-module of model

    Args:
        model: Model instance
    '''
    def __init__(self, model, name):
        if hparams.DEBUG:
            self.debug_fetches = {}
        self.name = name
        self.model = model

    def __call__(self, s_dropout_keep=1.):
        raise NotImplementedError()


class Encoder(ModelModule):
    '''
    maps log-magnitude-spectra to embedding
    '''
    def __init__(self, model, name):
        super(Encoder, self).__init__(model, name)

    def __call__(self, s_mixture, s_dropout_keep=1.):
        '''
        Args:
            s_mixture: tensor variable
                3d tensor of shape [batch_size, length, feature_size]

            s_dropout_keep: scalar const or variable
                keep probability for dropout layer

        Returns:
            [batch_size, length, feature_size, embedding_size]

        Notes:
            `length` is a not constant
        '''
        raise NotImplementedError()


class Estimator(ModelModule):
    '''
    Estimates attractor location, either from TF-embedding,
    or true source
    '''
    USE_TRUTH=True  # set this to true if it uses ground truth
    def __init__(self, model, name):
        super(Estimator, self).__init__(model, name)

    def __call__(self, s_embed, **kwargs):
        '''
        Args:
            s_embed: tensor of shape [batch_size, length, feature_size, embedding_size]

        Returns:
            s_attractors: tensor of shape [batch_size, num_signals, embedding_size]
        '''
        raise NotImplementedError()


class Separator(ModelModule):
    '''
    Given mixture power spectra, attractors, and embedding,
    produce power spectra of separated signals
    '''
    def __init__(self, model, name):
        super(Separator, self).__init__(model, name)

    def __call__(self, s_mixed_signals_pwr, s_attractors, s_embed_flat):
        '''
        Args:
            s_mixed_signals_pwr:
                tensor of shape [batch_size, length, feature_size]

            s_attractors:
                tensor of shape [num_attractor, embed_dims]

            s_embed_flat:
                tensor of shape [batch_size, num_signals, length, feature_size]
        '''
        raise NotImplementedError()

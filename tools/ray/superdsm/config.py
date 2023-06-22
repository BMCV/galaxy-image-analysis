import json
import hashlib


def _cleanup_value(value):
    return value.entries if isinstance(value, Config) else value


class Config:
    """Represents a set of hyperparameters.

    Hyperparameters can be worked with as follows:

    .. code-block:: python
    
       cfg = superdsm.config.Config()
       cfg['global-energy-minimization/beta'] = 1000
       cfg['global-energy-minimization/max_iter'] = 5

    A dictionary can be wrapped into a :py:class:`Config` object by passing it to the constructor (no copying occurs). If another :py:class:`Config` object is passed to the constructor, a deep copy is created.
    """

    def __init__(self, other=None):
        if other is None: other = dict()
        if isinstance(other, dict):
            self.entries = other
        elif isinstance(other, Config):
            self.entries = json.loads(json.dumps(other.entries))
        else:
            raise ValueError(f'Unknown argument: {other}')

    def pop(self, key, default):
        """Removes a hyperparameter from this configuration.

        :param key: The hyperparameter to be removed.
        :param default: Returned if the hyperparameter ``key`` is not set.
        :return: The value of the hyperparameter ``key`` or ``default`` if ``key`` is not set.
        """
        if '/' in key:
            keys = key.split('/')
            config = self
            for key in keys[:-1]:
                config = config.get(key, {})
            return config.pop(keys[-1], default)
        else:
            return self.entries.pop(key, default)

    def set_default(self, key, default, override_none=False):
        """Sets a hyperparameter if it is not set yet.

        :param key: The hyperparameter to be set.
        :param default: Returned if the hyperparameter ``key`` is not set.
        :param override_none: ``True`` if a hyperparameter set to ``None`` should be treated as not set.
        :return: The value of the hyperparameter ``key`` after the method invocation is finished.
        """
        if '/' in key:
            keys = key.split('/')
            config = self
            for key in keys[:-1]:
                config = config.set_default(key, {}, override_none)
            return config.set_default(keys[-1], default, override_none)
        else:
            if key not in self.entries or (override_none and self.entries[key] is None):
                self.entries[key] = _cleanup_value(default)
            return self[key]

    def get(self, key, default):
        """Returns the value of a hyperparameter.

        :param key: The hyperparameter to be queried.
        :param default: Returned if the hyperparameter ``key`` is not set.
        :return: The value of the hyperparameter ``key`` or ``default`` if ``key`` is not set.
        """
        if '/' in key:
            keys = key.split('/')
            config = self
            for key in keys[:-1]:
                config = config.get(key, {})
            return config.get(keys[-1], default)
        else:
            if key not in self.entries: self.entries[key] = _cleanup_value(default)
            value = self.entries[key]
            return Config(value) if isinstance(value, dict) else value

    def __getitem__(self, key):
        """Returns the value of a hyperparameter.

        :param key: The hyperparameter to be queried.
        :return: The value of the hyperparameter ``key``.
        :raises KeyError: Raised if the hyperparameter ``key`` is not set.
        """
        if '/' in key:
            keys = key.split('/')
            config = self
            for key in keys[:-1]:
                config = config[key]
            return config[keys[-1]]
        else:
            value = self.entries[key]
            return Config(value) if isinstance(value, dict) else value

    def __contains__(self, key):
        """Checks whether a hyperparameter is set.

        :param key: The hyperparameter to be queried.
        :return: ``True`` if the hyperparameter ``key`` is set and ``False`` otherwise.
        """
        try:
            self.__getitem__(key)
            return True
        except KeyError:
            return False

    def update(self, key, func):
        """Updates a hyperparameter by mapping it to a new value.

        :param key: The hyperparameter to be set.
        :param func: Function which maps the previous value to the new value.
        :return: The new value.
        """
        if '/' in key:
            keys = key.split('/')
            config = self
            for key in keys[:-1]:
                config = config.get(key, {})
            return config.update(keys[-1], func)
        else:
            self.entries[key] = _cleanup_value(func(self.entries.get(key, None)))
            return self.entries[key]

    def __setitem__(self, key, value):
        """Sets the value of a hyperparameter.

        :param key: The hyperparameter to be set.
        :param value: The new value of the hyperparameter.
        :return: The updated :py:class:`~.Config` object (itself).
        """
        self.update(key, lambda *args: value)
        return self

    def merge(self, config_override):
        """Updates this configuration using the hyperparameters set in another configuration.

        The hyperparameters of this configuration are set to the values from ``config_override``. If a hyperparameter was previously not set in this configuration, it is set to the value from ``config_override``.

        :param config_override: A :py:class:`~.Config` object corresponding to the configuration which is to be merged.
        :return: The updated :py:class:`~.Config` object (itself).
        """
        for key, val in _cleanup_value(config_override).items():
            if not isinstance(val, dict):
                self.entries[key] = val
            else:
                self.get(key, {}).merge(val)
        return self

    def copy(self):
        """Returns a deep copy.
        """
        return Config(self)

    def derive(self, config_override):
        """Creates and returns an updated deep copy of this configuration.

        The configuration ``config_override`` is merged into a deep copy of this configuration (see the :py:meth:`~.merge` method).

        :param config_override: A :py:class:`~.Config` object corresponding to the configuration which is to be merged.
        :return: The updated deep copy.
        """
        return self.copy().merge(config_override)

    def dump_json(self, fp):
        """Writes the JSON representation of this configuration.

        :param fp: The file pointer where the JSON representation is to be written to.
        """
        json.dump(self.entries, fp)
        
    @property
    def md5(self):
        """The MD5 hash code associated with the hyperparameters set in this configuration.
        """
        return hashlib.md5(json.dumps(self.entries).encode('utf8'))
    
    def __str__(self):
        """Readable representation of this configuration.
        """
        return json.dumps(self.entries, indent=2)


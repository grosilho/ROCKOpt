import logging


class HelperClass:
    def __init__(self, default_params, params):

        self.logger = logging.getLogger(self.__class__.__name__)

        params = self.fix_missing_params_with_defaults(default_params, params)
        params = self.fix_types(default_params, params)

        # store all problem params dictionary values as attributes
        for key, val in params.items():
            setattr(self, key, val)

        self.params = params  # store, so we can change and revert if needed

    def fix_missing_params_with_defaults(self, default_params, params):
        for key, val in default_params.items():
            if key not in params:
                params[key] = val
        return params

    def fix_types(self, default_params, params):
        for key in params:
            if type(params[key]) is not type(default_params[key]):
                old_type = type(params[key])
                params[key] = type(default_params[key])(params[key])
                self.logger.warning(
                    f"Type mismatch for parameter with key: '{key}'. Converting from {old_type} to {type(default_params[key])}. New value: {params[key]}"
                )

        return params

    def set(self, params):
        for key, value in params.items():
            if key in self.params:
                setattr(self, key, value)
            else:
                self.logger.warning(f"Parameter '{key}' not found in {self.__class__.__name__}")

    def revert(self, keys):
        for key in keys:
            if key in self.params:
                setattr(self, key, self.params[key])
            else:
                self.logger.warning(f"Parameter '{key}' not found in {self.__class__.__name__}")

    def init_stats(self, keys):
        self.stats = {key: 0 for key in keys}

    def init_history(self, keys):
        self.history = {key: [] for key in keys}

    def append_to_history(self, **kwargs):
        for key, value in kwargs.items():
            self.history[key].append(value)

    def add_stats(self, stats1, stats2):
        stats = stats1.copy()
        for key, value in stats2.items():
            if key in stats:
                stats[key] += value
            else:
                stats[key] = value
        return stats

    def unify_histories(self, history1, history2):
        # Check for overlapping keys
        overlapping_keys = set(history1.keys()) & set(history2.keys())
        if overlapping_keys:
            raise Exception(f"Cannot unify histories with overlapping keys: {overlapping_keys}")

        unified_history = {**history1, **history2}
        return unified_history

    def pre_loop(self, x, fx):
        pass

    def post_loop(self, x, fx):
        pass

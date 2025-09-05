from addict import Dict

def load_and_merge_config(config_path: str, **kwargs) -> Dict:
    """
    Load a configuration file and merge it with additional parameters.
    
    Args:
        config_path (str): Path to the configuration file.
        **kwargs: Additional parameters to override configuration settings.

    Returns:
        Dict: Merged configuration as an addict.Dict object.
    """
    import yaml

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    config = Dict(config)

    for key, value in kwargs.items():
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            if k not in d:
                d[k] = Dict()
            d = d[k]

        if isinstance(d.get(keys[-1]), dict) and isinstance(value, dict):
            d[keys[-1]].update(value)
        else:
            d[keys[-1]] = value

    return config
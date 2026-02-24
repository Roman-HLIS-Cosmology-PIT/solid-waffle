"""
Routines to run the infrared flat correlations.

Classes
-------
MultiConfig
    Inherits Config classs to extract configuration data from multiple configuration files. 

Functions
---------

"""

import numpy as np
from solid_waffle.correlation_run import Config

_ALLOWED_TO_DIFFER = frozenset([
    "lightfiles",
    "darkfiles",
    "outstem",
    "vislightfiles",
    "visdarkfiles",
    "full_info",
    "is_good",
    "lightref",
    "darkref",
    "NTMAX",
    "mean_full_info",
    "std_full_info",
    "nlfit",
    "nlder"
    ])

def _values_match(a, b):
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    else:
        return a == b

class MultiConfig(Config):
    def __init__(self, config_files, visible_run=False, verbose=False):
        if len(config_files) < 2:
            raise ValueError(f"Need at least 2 config files, got {len(config_files)}")
        self.configs = []
        for path in config_files:
            with open(path) as fh:
                cfg = Config(fh.readlines(), visible_run=visible_run, verbose=verbose)
                self.configs.append(cfg)
        ref_cfg = self.configs[0]
        self.__dict__.update(vars(ref_cfg))
        mismatches = []
        for cfg in self.configs[1:]:
            for key in vars(ref_cfg):
                if key in _ALLOWED_TO_DIFFER:
                    continue
                cfg_attr = getattr(cfg, key)
                ref_cfg_attr = getattr(ref_cfg, key)
                if not _values_match(cfg_attr, ref_cfg_attr):
                    mismatches.append(f"{key}: ref = {ref_cfg_attr}, got = {cfg_attr}")
        if mismatches:
            raise ValueError(f"ERROR: configuration files mismatched: {mismatches}")
        self._combine_results()

    def _combine_results(self):
        all_info = np.stack([cfg.full_info for cfg in self.configs], axis=0)
        all_good = np.stack([cfg.is_good for cfg in self.configs], axis=0)
        combined_good = np.prod(all_good, axis=0)
        combined_info = np.mean(all_info, axis=0)
        combined_info[combined_good < 0.5, :] = 0
        self.full_info = combined_info
        self.is_good = combined_good

    @classmethod
    def from_summaries(cls, config_files, visible_run=False, verbose=False):
        instance = cls.__new__(cls)
        if len(config_files) < 2:
            raise ValueError(f"Need at least 2 config files, got {len(config_files)}")
        instance.configs = []
        for path in config_files:
            with open(path) as fh:
                cfg = Config(fh.readlines(), visible_run=visible_run, verbose=verbose)
                instance.configs.append(cfg)
        ref_cfg = instance.configs[0]
        instance.__dict__.update(vars(ref_cfg))
        mismatches = []
        for cfg in instance.configs[1:]:
            for key in vars(ref_cfg):
                if key in _ALLOWED_TO_DIFFER:
                    continue
                cfg_attr = getattr(cfg, key)
                ref_cfg_attr = getattr(ref_cfg, key)
                if not _values_match(cfg_attr, ref_cfg_attr):
                    mismatches.append(f"{key}: ref = {ref_cfg_attr}, got = {cfg_attr}")
        if mismatches:
            raise ValueError(f"ERROR: configuration files mismatched: {mismatches}")
        for cfg in instance.configs:
            summary_path = cfg.outstem + "_summary.txt"
            data = np.loadtxt(summary_path)
            cfg.full_info = data[:, 2:].reshape(cfg.ny, cfg.nx, cfg.swi.N)
            cfg.is_good = np.where(cfg.full_info[:, :, cfg.swi.g] > 1e-49, 1, 0)
        instance._combine_results()
        return instance





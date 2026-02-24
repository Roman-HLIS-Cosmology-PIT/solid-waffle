"""
Routines to run the infrared flat correlations.

Classes
-------
MultiConfig
    Aggregates multiple completed IR characterization runs into one.

Functions
---------
_values_match
    Safely compares two values for equality, handling numpy arrays
    and plain Python types correctly.


"""

import numpy as np

from solid_waffle.correlation_run import Config

_ALLOWED_TO_DIFFER = frozenset(
    [
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
        "nlder",
    ]
)


def _values_match(a, b):
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    if hasattr(a, '__dict__'):
        return a.__dict__ == b.__dict__
    else:
        return a == b


class MultiConfig(Config):
    """
    Aggregates multiple completed IR characterization runs into one.

    Inherits all attributes and methods from Config. See Config docstring
    for full details of inherited attributes and methods.

    Parameters
    ----------
    config_files : list of str
        Paths to the individual configuration files (one per run).
    visible_run : bool, optional
        Passed through to each Config constructor.
    verbose : bool, optional
        Print progress and mismatch details.

    Raises
    ------
    ValueError
        If fewer than two config files are supplied, or if any configuration
        mismatch is found between runs.
    """

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
        """
        Combines full_info and is_good arrays across all runs.

        Averages full_info across runs and builds an intersection good-pixel
        map where a super-pixel is only good if it is good in every run.

        Returns
        -------
        None
        """
        all_info = np.stack([cfg.full_info for cfg in self.configs], axis=0)
        all_good = np.stack([cfg.is_good for cfg in self.configs], axis=0)
        combined_good = np.prod(all_good, axis=0)
        combined_info = np.mean(all_info, axis=0)
        combined_info[combined_good < 0.5, :] = 0
        self.full_info = combined_info
        self.is_good = combined_good

    @classmethod
    def from_summaries(cls, config_files, visible_run=False, verbose=False):
        """
        Build a MultiConfig from already-completed runs.

        Loads full_info directly from each run's _summary.txt instead
        of recomputing via fit_parameters. Use this when run_ir_all has
        already been called on every config file.

        Parameters
        ----------
        config_files : list of str
            Paths to the individual configuration files. Each must have a
            corresponding <outstem>_summary.txt on disk.
        visible_run : bool, optional
            Passed through to each Config constructor.
        verbose : bool, optional
            Print progress and mismatch details.

        Returns
        -------
        instance : MultiConfig
            Combined configuration loaded from summary files.

        Raises
        ------
        ValueError
            If fewer than two config files are supplied, or if any
            configuration mismatch is found between runs.
        """
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
            cfg.full_info = data[:, 2 : cfg.ski.N + 2].reshape(cfg.ny, cfg.nx, cfg.swi.N)
            cfg.is_good = np.where(cfg.full_info[:, :, cfg.swi.g] > 1e-49, 1, 0)
        instance._combine_results()
        return instance

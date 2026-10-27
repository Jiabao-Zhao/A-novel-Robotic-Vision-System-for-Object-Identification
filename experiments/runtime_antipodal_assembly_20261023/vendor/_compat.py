"""Minimal host integration for the vendored sampler's two utility imports.

The upstream sampler only needs a configuration constructor and log_warning.
No EmbodiChain simulation engine, assets, services or learned model is imported.
"""
from dataclasses import dataclass
import logging

configclass = dataclass


class _Logger:
    @staticmethod
    def log_warning(message):
        logging.getLogger(__name__).warning(message)


logger = _Logger()


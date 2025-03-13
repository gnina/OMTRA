from __future__ import annotations

_DEFAULT_DISTANCE_RANGE = {
    # Taken from Biotite Python
    # Taken from Allen et al.
    #               min   - 2*std     max   + 2*std
    ("B", "C"): (1.556 - 2 * 0.015, 1.556 + 2 * 0.015),
    ("BR", "C"): (1.875 - 2 * 0.029, 1.966 + 2 * 0.029),
    ("BR", "O"): (1.581 - 2 * 0.007, 1.581 + 2 * 0.007),
    ("C", "C"): (1.174 - 2 * 0.011, 1.588 + 2 * 0.025),
    ("C", "CL"): (1.713 - 2 * 0.011, 1.849 + 2 * 0.011),
    ("C", "F"): (1.320 - 2 * 0.009, 1.428 + 2 * 0.009),
    ("C", "H"): (1.059 - 2 * 0.030, 1.099 + 2 * 0.007),
    ("C", "I"): (2.095 - 2 * 0.015, 2.162 + 2 * 0.015),
    ("C", "N"): (1.325 - 2 * 0.009, 1.552 + 2 * 0.023),
    ("C", "O"): (1.187 - 2 * 0.011, 1.477 + 2 * 0.008),
    ("C", "P"): (1.791 - 2 * 0.006, 1.855 + 2 * 0.019),
    ("C", "S"): (1.630 - 2 * 0.014, 1.863 + 2 * 0.015),
    ("C", "SE"): (1.893 - 2 * 0.013, 1.970 + 2 * 0.032),
    ("C", "SI"): (1.837 - 2 * 0.012, 1.888 + 2 * 0.023),
    ("CL", "O"): (1.414 - 2 * 0.026, 1.414 + 2 * 0.026),
    ("CL", "P"): (1.997 - 2 * 0.035, 2.008 + 2 * 0.035),
    ("CL", "S"): (2.072 - 2 * 0.023, 2.072 + 2 * 0.023),
    ("CL", "SI"): (2.072 - 2 * 0.009, 2.072 + 2 * 0.009),
    ("F", "N"): (1.406 - 2 * 0.016, 1.406 + 2 * 0.016),
    ("F", "P"): (1.495 - 2 * 0.016, 1.579 + 2 * 0.025),
    ("F", "S"): (1.640 - 2 * 0.011, 1.640 + 2 * 0.011),
    ("F", "SI"): (1.588 - 2 * 0.014, 1.694 + 2 * 0.013),
    ("H", "N"): (1.009 - 2 * 0.022, 1.033 + 2 * 0.022),
    ("H", "O"): (0.967 - 2 * 0.010, 1.015 + 2 * 0.017),
    ("I", "O"): (2.144 - 2 * 0.028, 2.144 + 2 * 0.028),
    ("N", "N"): (1.124 - 2 * 0.015, 1.454 + 2 * 0.021),
    ("N", "O"): (1.210 - 2 * 0.011, 1.463 + 2 * 0.012),
    ("N", "P"): (1.571 - 2 * 0.013, 1.697 + 2 * 0.015),
    ("N", "S"): (1.541 - 2 * 0.022, 1.710 + 2 * 0.019),
    ("N", "SI"): (1.711 - 2 * 0.019, 1.748 + 2 * 0.022),
    ("O", "P"): (1.449 - 2 * 0.007, 1.689 + 2 * 0.024),
    ("O", "S"): (1.423 - 2 * 0.008, 1.580 + 2 * 0.015),
    ("O", "SI"): (1.622 - 2 * 0.014, 1.680 + 2 * 0.008),
    ("P", "P"): (2.214 - 2 * 0.022, 2.214 + 2 * 0.022),
    ("P", "S"): (1.913 - 2 * 0.014, 1.954 + 2 * 0.005),
    ("P", "SE"): (2.093 - 2 * 0.019, 2.093 + 2 * 0.019),
    ("P", "SI"): (2.264 - 2 * 0.019, 2.264 + 2 * 0.019),
    ("S", "S"): (1.897 - 2 * 0.012, 2.070 + 2 * 0.022),
    ("S", "SE"): (2.193 - 2 * 0.015, 2.193 + 2 * 0.015),
    ("S", "SI"): (2.145 - 2 * 0.020, 2.145 + 2 * 0.020),
    ("SE", "SE"): (2.340 - 2 * 0.024, 2.340 + 2 * 0.024),
    ("SI", "SE"): (2.359 - 2 * 0.012, 2.359 + 2 * 0.012),
}

LIGAND_MAP = ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I"]
NPNDE_MAP = [
    "C",
    "N",
    "O",
    "S",
    "F",
    "P",
    "Cl",
    "Br",
    "I",
    "Si",
    "Pd",
    "Ca",
    "Sc",
    "Al",
    "Pt",
    "Cr",
    "Rh",
    "Mo",
    "Y",
    "Cu",
    "Sm",
    "Bi",
    "Ir",
    "Be",
    "Pb",
    "Kr",
    "As",
    "Sn",
    "Lu",
    "Co",
    "Ni",
    "K",
    "Am",
    "Rb",
    "Gd",
    "Ce",
    "Ba",
    "Au",
    "Na",
    "Th",
    "Pr",
    "Yb",
    "Ti",
    "Mg",
    "Sr",
    "Cs",
    "Tl",
    "B",
    "Xe",
    "Cm",
    "Li",
    "Zr",
    "Cf",
    "Ag",
    "Cd",
    "Mn",
    "Eu",
    "In",
    "Hg",
    "W",
    "Tb",
    "La",
    "Sb",
    "Te",
    "Se",
    "Zn",
    "Os",
    "Ga",
    "Fe",
    "Nd",
    "Pu",
    "V",
    "U",
    "Ru",
    "Re",
]

# Copyright (c) 2024, Plinder Development Team
# Distributed under the terms of the Apache License 2.0


import inspect
import logging
import os

LOGGING_FORMAT: str = "%(asctime)s | %(name)s:%(lineno)d | %(levelname)s : %(message)s"
try:
    DEFAULT_LOGGING_LEVEL: int = int(os.getenv("PLINDER_LOG_LEVEL", "20"))
except ValueError:
    DEFAULT_LOGGING_LEVEL = logging.INFO


def setup_logger(
    logger_name: str | None = None,
    log_level: int = DEFAULT_LOGGING_LEVEL,
    log_file: str | None = None,
    propagate: bool = False,
) -> logging.Logger:
    """
    Setup logger for the module name as the logger name by default
    for easy tracing of what's happening in the code

    Parameters
    ----------
    logger_name : str
        Name of the logger
    log_level : int
        Log level
    log_file: str | None
        optional log file to write to
    propagate : bool
        propagate log events to parent loggers, default = False

    Returns
    -------
    logging.Logger:
        logger object

    Examples
    --------
    >>> logger = setup_logger("some_logger_name")
    >>> logger.name
    'some_logger_name'
    >>> logger.level
    20
    >>> logger = setup_logger(log_level=logging.DEBUG)
    >>> logger.name
    'log.py'
    >>> logger.level
    10
    """

    if logger_name is None:
        # Get module name as the logger name, this is copied from:
        # https://stackoverflow.com/questions/13699283/how-to-get-the-callers-filename-method-name-in-python
        frame = inspect.stack()[1]
        module = inspect.getmodule(frame[0])
        file_path = __file__ if module is None else module.__file__
        logger_name = os.path.basename(file_path) if file_path is not None else "log"

    # Set up logger with the given logger name
    logger = logging.getLogger(logger_name)
    # Check if logging level has been set externally otherwise first pass logger.level == 0 (NOTSET)
    set_level = not bool(logger.level)
    if set_level:
        logger.setLevel(log_level)
    handler = logging.StreamHandler()
    if set_level:
        handler.setLevel(log_level)
    formatter = logging.Formatter(LOGGING_FORMAT)
    handler.setFormatter(formatter)
    if not len(logger.handlers):
        logger.addHandler(handler)

    if log_file is None:
        log_file = os.environ.get("LOG_FILE_PATH", "default.log")

    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        if set_level:
            file_handler.setLevel(log_level)
        if not [h for h in logger.handlers if h.__class__ == logging.FileHandler]:
            logger.addHandler(file_handler)
    logger.propagate = propagate

    return logger

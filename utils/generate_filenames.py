"""
Filename generation utilities for saving and loading models and data.

Provides utilities for creating consistent, parameter-based filenames
with automatic sanitization and project root handling.
"""

import re
import os
import sys
from pathlib import Path


def generate_filename(
    prefix: str,
    params: dict,
    extension: str = "eqx",
    max_length: int = 255,
    project_root: Path | None = None
) -> Path:
    """Generate a safe filename with parameters, optionally prepending project root.
    
    Combines a prefix, parameter dictionary, and extension into a filesystem-safe
    filename. Automatically sanitizes special characters and truncates if needed.
    
    Parameters
    ----------
    prefix : str
        Directory path + base filename (without parameters or extension).
    params : dict
        Parameters to include in filename. Keys and values are concatenated
        as "key-value" pairs separated by underscores.
    extension : str, optional
        File extension without dot. Default is "eqx".
    max_length : int, optional
        Maximum total filename length in characters. Default is 255 (typical
        filesystem limit). If exceeded, filename is truncated.
    project_root : Path or None, optional
        If provided, directory path will be relative to this root. Default is None.
    
    Returns
    -------
    Path
        Full path to the generated filename as a pathlib.Path object.
    
    Examples
    --------
    >>> params = {"lr": 0.001, "epochs": 100}
    >>> generate_filename("models/policy", params, extension="pt")
    PosixPath('models/policy_lr-0.001_epochs-100.pt')
    """
    # Split directory and base name
    directory, base = Path(prefix).parent, Path(prefix).name

    # Prepend project root if provided
    if project_root:
        directory = Path(project_root) / directory

    # Build parameter string
    param_str = "_".join(f"{key}-{value}" for key, value in params.items())

    # Build full filename
    filename = f"{base}_{param_str}.{extension}"

    # Sanitize filename by replacing special characters
    filename = re.sub(r'[\\/*?:"<>|]', "_", filename)

    # Trim if too long
    if len(filename) > max_length:
        filename = filename[:max_length - len(extension) - 1] + f".{extension}"

    # Combine directory and filename
    return directory / filename

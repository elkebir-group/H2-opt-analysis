"""Location of the raw data: $H2OPT_DATA, or data/raw at the repository root."""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get('H2OPT_DATA', REPO / 'data' / 'raw'))


def data(*parts):
    """Path of a raw data file, from its parts below the data root."""
    return DATA.joinpath(*parts)

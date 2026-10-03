"""Location of the data directory: $H2OPT_DATA, or data/ at the repository root."""

import os
from pathlib import Path

DATA = Path(os.environ.get('H2OPT_DATA', Path(__file__).resolve().parent.parent / 'data'))


def data(*parts):
    return DATA.joinpath(*parts)

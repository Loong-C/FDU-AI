"""Theme-aware lookup of presentation assets."""

import json
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CATALOG_PATH = _PROJECT_ROOT / 'Assets' / 'themes.json'


def load_theme(theme_name=None):
    """Return the selected theme definition from the asset manifest."""
    with _CATALOG_PATH.open(encoding='utf-8') as catalog_file:
        catalog = json.load(catalog_file)
    selected_name = theme_name or catalog['defaultTheme']
    try:
        return catalog['themes'][selected_name]
    except KeyError as error:
        raise ValueError('Unknown asset theme: %s' % selected_name) from error


def sprite_path(sprite_key, theme_name=None):
    """Resolve a logical sprite key to an absolute path without exposing paths to renderers."""
    theme = load_theme(theme_name)
    try:
        relative_path = theme['sprites'][sprite_key]
    except KeyError as error:
        raise ValueError('Missing sprite key: %s' % sprite_key) from error
    return str(_PROJECT_ROOT / 'Assets' / relative_path)

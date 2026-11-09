"""Shared canonical loading for coursework in the grader and evaluation worker."""

from dataclasses import dataclass
import importlib
import importlib.util
import sys


def load_module_file(module_name, file_path):
    """Register one module identity; restore the previous identity on failure."""
    package = None
    if '.' in module_name:
        package_name, attribute = module_name.rsplit('.', 1)
        package = importlib.import_module(package_name)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous
        raise
    if package is not None:
        setattr(package, attribute, module)
    return module


@dataclass(frozen=True)
class DeferredStudentModule:
    """A source location only: optional student code must not execute in the grader."""
    path: str = None
    error: str = None

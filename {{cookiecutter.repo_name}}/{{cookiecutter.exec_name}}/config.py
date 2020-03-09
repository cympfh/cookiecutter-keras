import configparser
import sys
from typing import Any, Optional, Tuple


class TryCast:

    @classmethod
    def _int(cls, x: str) -> Optional[int]:
        try:
            return int(x)
        except ValueError:
            return None

    @classmethod
    def _float(cls, x: str) -> Optional[float]:
        try:
            return float(x)
        except ValueError:
            return None

    @classmethod
    def _bool(cls, x: str) -> Optional[bool]:
        label = x.lower()
        if label in ('true', 'yes', 'on'):
            return True
        if label in ('false', 'no', 'off'):
            return False
        return None

    @classmethod
    def cast(cls, x: str) -> Any:
        b = cls._bool(x)
        if b is not None:
            return b
        i = cls._int(x)
        if i is not None:
            return i
        f = cls._float(x)
        if f is not None:
            return f
        return x  # str


class Config:

    def __init__(self, section: str):
        self._config = configparser.ConfigParser()
        self._config.read('conf.ini')
        if section not in self._config.sections():
            self._config.add_section(section)
            print(f"[WARNING] section {section} not found in conf.ini",
                  file=sys.stderr)
        self.section = section

    def __call__(self, option: str, dtype=None) -> Any:
        """
        Return the value by (`section`, `option`).
        The value will be casted as `dtype`.
        If `dtype` is None, the type be estimated.
        """
        val = self._config.get(self.section, option)
        if dtype is not None:
            return dtype(val)
        return TryCast.cast(val)

    def __iter__(self) -> Tuple[str, Any]:
        for key in self._config.options(self.section):
            yield (key, self(key))

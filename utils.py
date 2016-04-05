from enum import Enum, IntEnum
import collections
from itertools import chain


class classproperty(object):
    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class StrFn:
    ellipsis = ".."

    @classmethod
    def truncate(cls, string, maxlen=30):
        stop = maxlen - len(cls.ellipsis)
        return (string[:stop] + cls.ellipsis) if len(
            string) > maxlen else string


class BoolEnum(Enum):
    def __bool__(self):
        return bool(self.value)


class AddEnum(IntEnum):
    def __add__(self, other):
        cls = type(self)
        return cls(self.value + int(other))


class YesNo(BoolEnum):
    no = False
    yes = True


class EnumMember:
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        return self.f(*args, **kwargs)


class FormatDict(collections.MutableMapping):
    def __init__(self, *args, **kwargs):
        data = {}
        data.update(*args, **kwargs)
        self.lazy_data = []

        self.data = data
        self.add(data)

    def add_data(self, keys, obj, get_item=False):
        self.lazy_data.append((keys, obj, get_item))

    def add(self, other):
        if isinstance(other, FormatDict):
            self.lazy_data += other.lazy_data
        elif isinstance(other, dict):
            self.add_data(other.keys, other, True)

    # The next five methods are requirements of the ABC.
    def __setitem__(self, key, value):
        self.data[key] = value

    def __getitem__(self, key):
        for keys, obj, get_item in self.lazy_data:
            try:
                if get_item:
                    return obj[key]
                else:
                    return getattr(obj, key)
            except (AttributeError, KeyError):
                continue

        raise KeyError(key)

    def __delitem__(self, key):
        del self.data[key]

    def __iter__(self):
        return chain(*self._key_groups)

    @property
    def _key_groups(self):
        for d in self.lazy_data:
            if callable(d[0]):
                yield d[0]()
            else:
                yield d[0]

    def __len__(self):
        return sum(map(len, self._key_groups))

    # The final two methods aren't required, but nice for demo purposes:
    def __str__(self):
        '''returns simple dict representation of the mapping'''
        return str(self.keys())

    def __repr__(self):
        '''echoes class, id, & reproducible representation in the REPL'''
        return '{}, D({})'.format(super().__repr__(), self.keys())

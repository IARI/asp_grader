from abc import ABCMeta, abstractmethod
from ascii import add_article
from peewee import Model, Field, SelectQuery
from models import BaseModel, Searchable
from enum import Enum
from utils import YesNo, EnumMember
from datetime import datetime
import re


class Parameter(metaclass=ABCMeta):
    def __init__(self, name, prompt=None, remember=False, lookup=True):
        self.remember = remember
        self._name = name
        self._prompt = prompt
        self.lookup = lookup

    def check(self, value):
        if not self.valid(value):
            raise ValueError("invalid input: '{}'".format(value))
        return self.coerce(value)

    def valid(self, value):
        if value not in self.options:
            txt = "invalid input: '{}'\nmust be one of \n\t{}"
            str_lst = ", ".join("'{0}'".format(n) for n in self.options)
            raise ValueError(txt.format(value, str_lst))
        return True

    def make_remember(self, value=True):
        self.remember = value
        return self

    @classmethod
    def inject(cls, par):
        if isinstance(par, Parameter):
            return par
        elif isinstance(par, Field):
            return DataPar(par)
        elif isinstance(par, type):
            if issubclass(par, Enum):
                return EnumPar(par)
            elif issubclass(par, Model):
                return DataPar(par)
            else:
                return TypePar(par)
        elif isinstance(par, str):
            return TypePar(str, par)
        elif isinstance(par, SelectQuery):
            return DataPar(par)
        elif isinstance(par, list):
            return ListPar(par)
        raise TypeError('invalid parameter type {}'.format(type(par)))

    @property
    @abstractmethod
    def options(self):
        raise NotImplementedError

    @abstractmethod
    def coerce(self, v):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def prompt_text(self):
        return self._prompt if self._prompt else "please select {}:".format(
            add_article(self.name)) + "\n\t{cmds}"

    def __or__(self, other):
        return SumPar(self, other)


class DataPar(Parameter):
    SPLITTER = " # "

    @classmethod
    def Nullable(cls, *args, **kwargs):
        dp = cls(*args, **kwargs)
        return SumPar(ListPar([None]), dp, **kwargs)

    def __init__(self, field_model_func, data=None, prompt=None,
                 remember=False, lookup=True, name=None, primary_keys=False):
        '''
        :param field_model_func: Determines how the data parameter is selected.
            can be either a Field, Model, or a Function
                    * Field: each record will be selected by the field
                    * Model: no specific given, so default searchable fields
                                will be used, unless data=PrimaryKeyField,
                                then Primary Keys will be used
                    * Function: a function recordset->string will determine
                                selection
        '''
        if isinstance(field_model_func, Field):
            self.fields = [field_model_func]
            self.data = field_model_func.model_class
        elif isinstance(field_model_func, SelectQuery):
            self.init_fields(field_model_func.model_class, primary_keys)
            self.data = field_model_func
        elif isinstance(field_model_func, type
                        ) and issubclass(field_model_func, Model):
            self.init_fields(field_model_func, primary_keys)
            self.data = field_model_func
        elif callable(field_model_func) and data:
            self.field_f = field_model_func
        else:
            raise TypeError('invalid type {}'.format(type(field_model_func)))

        if data is not None:
            self.data = data

        if not name:
            if hasattr(self.data, '__name__'):
                name = self.data.__name__
            elif hasattr(self, 'fields'):
                name = self.fields[0].model_class.__name__
            else:
                name = self.field_f.__name__
        super().__init__(name, prompt, remember, lookup)

    def init_fields(self, model, primary_keys):
        if primary_keys:
            self.fields = model._meta.get_primary_key_fields()
        else:
            self.fields = Searchable.fields[model]
            assert self.fields, \
                Exception('The model {} has no searchable fields. '
                          'Try constructing the parameter with '
                          'primary_keys=True'.format(
                    model.__name__))

    def field_f(self, r):
        return self.SPLITTER.join(self._field_f(r, f) for f in self.fields)

    @staticmethod
    def _field_f(record, field):
        return str(getattr(record, field.db_column))

    @property
    def options(self):
        return [self.field_f(s) for s in self.data.select()]

    def coerce(self, v):
        vs = v.split(self.SPLITTER)
        if hasattr(self, 'fields'):
            assert len(vs) == len(self.fields), ValueError(
                'unequal field length - need {} inputs'.format(
                    len(self.fields)))
            # return getattr(self.data, 'model_class', self.data).get(
            #        *(f == f.coerce(v) for f, v in zip(self.fields, vs)))
            # for s in self.data:
            #    if all(f == f.coerce(v) for f, v in zip(self.fields, vs)):
            #        return s
        for s in self.data:
            if v == self.field_f(s):
                return s
        raise ValueError(
            'could not coerce {} to {}'.format(v, self.options))


class TypePar(Parameter):
    def __init__(self, type: type, name=None, prompt=None,
                 remember=False, lookup=True):
        super().__init__(name if name else type.__name__, prompt,
                         remember, lookup)
        self.type = type

    def coerce(self, v):
        return self.type(v)

    def valid(self, value):
        return True

    @property
    def options(self):
        return []

    def __hash__(self):
        return self._name.__hash__()

    def __eq__(self, other):
        return self.type == other.type and self._name == other._name


class DatePar(TypePar):
    try_formats = {'%d.%m.%y': (),
                   '%d.%m.%Y': (),
                   '%d.%m.': ('year',),
                   '%d.': ('year',),
                   '%x': (),
                   '%X': ('year', ''),
                   '%c': (),
                   }

    def __init__(self, name=None, prompt=None, remember=False, lookup=True):
        super().__init__(datetime, name, prompt, remember, lookup)

    @classmethod
    def get_current(cls, *names):
        now = datetime.now()
        return {name: getattr(now, name) for name in names}

    def coerce(self, v):
        for f, defs in self.try_formats.items():
            try:
                d = datetime.strptime(v, f)
                return d.replace(**self.get_current(*defs))
            except:
                continue
        raise ValueError('entered value {} matched '
                         'none of the formats: \n\t{}'.format(v, "\n\t".join(
            self.try_formats)))


class PathPar(Parameter):
    def __init__(self, root: str, name, pattern=r".*", prompt=None,
                 remember=False,
                 lookup=True):
        super().__init__(name, prompt, remember, lookup)
        self.root = root
        self.pattern = pattern

    def coerce(self, v):
        if re.match(self.pattern, v):
            return v
        raise ValueError(
            'Path "{}" does not match {}.'.format(v, self.pattern))

    def valid(self, value):
        return True

    @property
    def options(self):
        return []


class SumPar(Parameter):
    def __init__(self, *children, **kwargs):
        self.children = [self.inject(c) for c in children]
        name = kwargs.get('name', "".join(c.name for c in self.children))
        super().__init__(name, kwargs.get('prompt', None),
                         kwargs.get('remember', False),
                         kwargs.get('lookup', True))

    @property
    def options(self):
        return [o for c in self.children for o in c.options]

    def _valid(self, value):
        errors = []
        for i, c in enumerate(self.children):
            try:
                if c.valid(value):
                    return i
            except ValueError as e:
                errors.append(e)
        raise ValueError("All checks on sum-parameter failed:\n" + "\n".join(
            str(e) for e in errors))

    def valid(self, value):
        self._valid(value)
        return True

    def coerce(self, v):
        i = self._valid(v)
        return self.children[i].coerce(v)


class EnumPar(Parameter):
    def __init__(self, enum: type, name=None, prompt=None,
                 restrict=False, remember=False, lookup=True):
        super().__init__(name if name else enum.__name__,
                         prompt, remember, lookup)
        self._enum = enum
        self.values = [
            r for r in restrict if isinstance(r, enum)
            ] if isinstance(restrict, list) else list(enum)
        assert issubclass(enum, Enum)

    def coerce(self, v):
        try:
            r = self._enum[v]
        except KeyError:
            raise ValueError(
                "'{}' is not a valid answer. ".format(
                    v) + self.prompt_text)
        return r

    @property
    def options(self):
        return [v.name for v in self.values]

    @property
    def prompt_text(self):
        return self._prompt if self._prompt else super().prompt_text

    def __hash__(self):
        return self._name.__hash__()

    def __eq__(self, other):
        return self.values == other.values and self._name == other._name


class ListPar(Parameter):
    def __init__(self, options, name='List', prompt=None, remember=False,
                 lookup=True):
        super().__init__(name, prompt, remember, lookup)
        self._options = options
        self.opt_dict = {getattr(r, '__name__', str(r)): r for r in options}

    def coerce(self, v):
        try:
            return self.opt_dict[v]
        except KeyError:
            raise ValueError(
                "'{}' is not a valid answer. ".format(
                    v) + self.prompt_text)

    @property
    def options(self):
        return list(self.opt_dict.keys())

    @property
    def prompt_text(self):
        return self._prompt if self._prompt else super().prompt_text

    def __hash__(self):
        return self._name.__hash__()

    def __eq__(self, other):
        return self.values == other.values and self._name == other._name


def ask(question):
    return EnumPar(YesNo, prompt=question)


class Delete:
    def __init__(self, *keys):
        self.keys = keys


class GetVariable:
    def __init__(self, name):
        self.name = name

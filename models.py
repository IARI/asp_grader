from peewee import SqliteDatabase, Model, CharField, IntegerField, \
    FloatField, ForeignKeyField, PrimaryKeyField, DateField, DateTimeField, \
    BooleanField, DoesNotExist, IntegrityError, OP, Expression, JOIN
from os import getcwd, path, listdir, setpgrp
from svn.local import LocalClient
from utils import *
from enum import Enum
import sys
from subprocess import Popen, DEVNULL
from functools import reduce
import socket
import re
from collections import defaultdict
from xml.dom.minidom import parse, Document
from wrapt import ObjectProxy

# from types import MethodType

db = SqliteDatabase('asp.sqlite')
backup_matcher = r"asp.(\d+).sqlite"
backup_pattern = 'asp.{}.sqlite'
os_db = SqliteDatabase(socket.gethostname() + '.sqlite')


class OS(Enum):
    OSX = 'darwin'
    LINUX = 'linux'

    # WIN = 'windows'

    @property
    def is_running(self):
        return self.value == sys.platform


# Todo: make a @lazy decorator for methods

class Record(object):
    def __init__(self, data: dict = None, **kwargs):
        self.data = data.copy() if data else {}
        self.data.update(kwargs)
        self.id_values = None
        self.unique = {}

    def __get__(self, record, model=None):
        if not record is None:
            raise AttributeError('Record may not be called on class')

        if self.id_values:
            return self.sq_from_model(model, **self.id_values).get()

        # necesarry to enable self-reference of records in self-referencial models.
        def recursive(v):
            return v.__get__(None, model) if isinstance(v, type(self)) else v

        data = {k: recursive(v) for k, v in self.data.items()}

        funique = self.first_unique(model, True)
        filter = lambda k, v: k in funique
        r, created = self.get_or_create(model, *self.part_dict(data, filter))

        self.id_values = {fname: getattr(r, fname) for fname in
                          self.primary_index(model)}

        if created:
            print("created {}: {}".format(model, data))
        return r

    def first_unique(self, model, strict=False):
        try:
            return next(x for x in self.unique_indexes(model) if
                        all(id in self.data for id in x))
        except StopIteration:
            if not strict:
                return model._meta.get_field_names()
            raise Exception(
                'No unique index for record {} in {}\nIndexes: {}'.format(
                    self.data, model, self.unique_indexes(model)))

    @classmethod
    def unique_indexes(cls, model):
        r = [fields for fields, unique in model._meta.indexes if unique]
        r.append(cls.primary_index(model))
        return r

    @classmethod
    def unique_indexes(cls, model):
        r = [fields for fields, unique in model._meta.indexes if unique]
        r += [[n] for n, f in model._meta.fields.items() if f.unique]
        r.append(cls.primary_index(model))
        return r

    @classmethod
    def primary_index(cls, model):
        return tuple(f.name for f in model._meta.get_primary_key_fields())

    @classmethod
    def get_or_create(cls, model, key_values, defaults):
        sq = cls.sq_from_model(model, **key_values)
        try:
            return sq.get(), False
        except DoesNotExist:
            try:
                params = {k: v for k, v in key_values.items() if '__' not in k}
                params.update(defaults)
                with model._meta.database.atomic():
                    return model.create(**params), True
            except IntegrityError as exc:
                try:
                    return sq.get(), False
                except DoesNotExist:
                    raise exc

    @classmethod
    def sq_from_model(cls, model, **kwargs):
        null_kvs, notnull_kvs = cls.part_dict(kwargs, lambda k, v: v is None)
        sq = model.select()
        if null_kvs:
            sq = sq.where(*(model._meta.fields[n].is_null() for n in null_kvs))
        if notnull_kvs:
            sq = sq.filter(**notnull_kvs)
        return sq

    @staticmethod
    def part_dict(d: dict, predicade=lambda k, v: v):
        return {k: v for k, v in d.items() if predicade(k, v)}, \
               {k: v for k, v in d.items() if not predicade(k, v)}

    def __set__(self, obj, value):
        raise NotImplementedError
        # self.obj = value


class Searchable:
    fields = defaultdict(list)

    @classmethod
    def register(cls, f, *args, **kwargs):
        field = f(*args, **kwargs)
        temp_add = field.add_to_class

        def add_to_class(model_class, name):
            cls.fields[model_class].append(field)
            temp_add(model_class, name)

        field.add_to_class = add_to_class
        return field

    @classmethod
    def CharField(cls, *args, **kwargs):
        return cls.register(CharField, *args, **kwargs)

    @classmethod
    def IntegerField(cls, *args, **kwargs):
        return cls.register(IntegerField, *args, **kwargs)

    @classmethod
    def PrimaryKeyField(cls, *args, **kwargs):
        return cls.register(PrimaryKeyField, *args, **kwargs)

    @classmethod
    def query(cls, model_classes, value, op=OP.IN):
        def coercable(f, v):
            try:
                f.coerce(v)
                return True
            except:
                return False

        nodes = [Expression(f, op, value) for m in model_classes
                 for f in cls.fields[m] if coercable(f, value)]

        return reduce(lambda f, old: (f) | (old), nodes, True)


class BaseModel(Model):
    @property
    def format_vars(self):

        rv = FormatDict()
        field_names = self._meta.get_field_names()
        class_keys = [k for k, v in self.__class__.__dict__.items() if
                      isinstance(v, property)]
        rv.add_data(field_names, self)
        rv.add_data(class_keys, self)
        return rv

    @property
    def field_values(self):
        return {f: getattr(self, f) for f in self._meta.get_field_names()}

    @classmethod
    def const_records(cls):
        return {n: getattr(cls, n) for n, a in cls.__dict__.items() if
                isinstance(a, Record)}

    def get_pk_names(self):
        for pk in self._meta.get_primary_key_fields():
            yield pk.name

    def clone(self):
        d = {k: v for k, v in self.field_values.items() if
             k not in self.get_pk_names()}
        return type(self)(**d)

    @classproperty
    def searchable(cls):
        return [cls]

    @classmethod
    def search_completer(cls, models=None):
        if models is None:
            models = cls.searchable

        def completer(cls, text, state):
            rm = cls._meta.related_models(True)
            # return []

        return completer

    @classmethod
    def related_paths(cls, limit=None, backrefs=True):
        model_paths = defaultdict(list)
        stack = [(cls, None)]
        while stack:
            model, parent = stack.pop(0)
            if model_paths[model]:
                continue
            if parent:
                model_paths[model] += model_paths[parent]
            model_paths[model].append(model)

            for fk in model._meta.rel.values():
                stack.append((fk.rel_model, model))
            if backrefs:
                for fk in model._meta.reverse_rel.values():
                    stack.append((fk.model_class, model))
        if not limit:
            return dict(model_paths)
        return {k: v for k, v in model_paths.items() if len(v) <= limit}

    class Meta:
        database = db


class FileCommand(BaseModel):
    command = CharField()
    shell = BooleanField(default=False)

    OPEN = Record(id=1,
                  command='open -nW' if OS.OSX.is_running else 'gnome-open')
    ANDROID_STUDIO = Record(id=3,
                            command='open -n /Applications/Android\ Studio.'
                                    'app/ --args' if OS.OSX.is_running else 'android-studio',
                            shell=True)
    DIFF = Record(id=4,
                  command='open -W /Applications/DiffMerge.app/ --args'
                          '' if OS.OSX.is_running else 'meld')

    SHELL_ALL_DEVNULL = dict(
        shell=True,
        stdin=DEVNULL,
        stdout=DEVNULL,
        stderr=DEVNULL
    )

    DEFAULT_POPEN_ARGS = dict(
        preexec_fn=setpgrp,
        # lambda: signal.signal(signal.SIGINT,signal.SIG_IGN)
    )

    def open(self, *args, **kwargs):
        cmd = [s.replace('\ ', ' ') for s in
               re.split(r'(?<!\\) ', self.command)] + list(args)
        print(str(cmd))
        ukwargs = self.DEFAULT_POPEN_ARGS.copy()
        if self.shell:
            ukwargs.update(self.SHELL_ALL_DEVNULL)
        ukwargs.update(kwargs)
        return Popen(cmd, **ukwargs)

    class Meta:
        database = os_db


# safely create FileCommand table, its const records are referenced in FileType
os_db.create_tables([FileCommand], safe=True)


class FileType(BaseModel):
    type = CharField()
    mime = CharField(null=True, default=None)
    command = ForeignKeyField(FileCommand, null=True, default=None)

    DIR = Record(id=1, type="Directory")
    PDF = Record(id=2, type="PDF", command=FileCommand.OPEN)
    TXT = Record(id=3, type="TXT", command=FileCommand.OPEN)
    XML = Record(id=4, type="XML", command=FileCommand.OPEN)
    ASP = Record(id=5, type="AS Project", command=FileCommand.ANDROID_STUDIO)

    OTH = Record(id=10, type="unknown file")

    @property
    def readable(self):
        return self.command is not None

    def open(self, *args, **kwargs):
        return self.command.open(*args, **kwargs)


# safely create FileType table, its const records are referenced in Directory
os_db.create_tables([FileType], safe=True)


class Hierarchy:
    class Method:
        def __init__(self, f, instance=None, typ=None):
            self.function = f
            self.instance = instance
            self.type = typ

        def __get__(self, obj, type=None):
            return self.__class__(self.function, obj, type)

        @property
        def method(self):
            return self.get_method(self.instance)

        def get_method(self, instance, typ=None):
            if typ is None:
                typ = self.type
            assert isinstance(instance, Hierarchy), TypeError(
                '{} is no {}'.format(instance, Hierarchy.__name__))
            return self.function.__get__(instance, typ)

        def __call__(self, *args, **kwargs):
            raise NotImplementedError

    class AncestorMethod(Method):
        def __call__(self, *args, **kwargs):
            m, p = self.method, self.instance.parent
            if p: yield from getattr(p, m.__name__,
                                     self.get_method(p))(*args, **kwargs)
            yield m(*args, **kwargs)

    class DescendantMethod(Method):
        def __call__(self, *args, **kwargs):
            m = self.method
            yield m(*args, **kwargs)
            for c in self.instance.children:
                yield from getattr(c, m.__name__,
                                   self.get_method(c))(*args, **kwargs)

    @property
    def children(self):
        raise NotImplementedError('must be overridden.')

    @property
    def parent(self):
        raise NotImplementedError('must be overridden.')

    @property
    def descendants(self):
        return self.get_descendants()

    @DescendantMethod
    def get_descendants(self):
        return self

    @staticmethod
    def get(*args):
        for h in args:
            yield from h.descendants

    @property
    def ancestors(self):
        return self.get_ancestors()

    @AncestorMethod
    def get_ancestors(self):
        return self

    def __gt__(self, other):
        return other in self.ancestors

    def __lt__(self, other):
        return self in other.ancestors


class DirectoryBase(Hierarchy):
    @property
    def absolute(self):
        raise NotImplementedError('must be overridden.')

    def open(self):
        return self.file_type.open(self.absolute)

    def open_if_exists(self):
        if self.exists:
            return self.open()

    def create_if_not_exists(self):
        if not self.exists:
            f = open(self.absolute, 'x')
            f.close()
            return True
        return False

    @property
    def path(self):
        raise NotImplementedError('must be overridden.')

    def relpath(self, other):
        return path.relpath(self.absolute, other.absolute)

    @property
    def repository_dir(self):
        raise NotImplementedError('must be overridden.')

    @property
    def relative_to_repo(self):
        # assert self > self.repository_dir, '{} is no subdir of {}!'.format(
        #    self, self.repository_dir)
        return self.relpath(self.repository_dir)

    def read_file(self):
        if hasattr(self, '_read_file_data'):
            print('file {} was already read.'.format(self.path))
            return self._read_file_data

        assert self.exists and self.file_type and self.file_type.readable
        if self.file_type == FileType.XML:
            self._read_file_data = parse(self.absolute)
        else:
            f = open(self.absolute)
            self._read_file_data = f.read()
            f.close()

        return self._read_file_data

    def save_file(self, data=None):
        if data is None:
            assert hasattr(self, '_read_file_data')
            data = self._read_file_data
        if self.file_type == FileType.XML:
            assert isinstance(data, Document)
            data = data.toxml()

        f = open(self.absolute, 'w')
        f.write(data)
        f.close()

    @property
    def type(self):
        # TODO: remove this
        return self.Type(self.Type.testDir(self.absolute))

    @property
    def file_type(self):
        raise NotImplementedError('must be overridden.')

    @property
    def exists(self):
        if self.parent:
            return self.parent.exists and self.type.exists(self.path,
                                                           self.parent.absolute)
        else:
            return path.exists(self.absolute)

    def __str__(self):
        return '<{}: {}>'.format(type(self), self.absolute)

    class Type(Enum):
        File = 0
        Directory = 1

        @property
        def test(self):
            return path.isdir if self.value else path.isfile

        @classmethod
        def exists(cls, pth, root):
            for p in cls.split(pth):
                if p not in listdir(root) + ['.', '..']:
                    return False
                root = path.join(root, p)
            return True

        @classmethod
        def split(cls, pth):
            r = path.normpath(pth).split(path.sep)
            return r if r[0] or not r else r[1:]
            # a, b = path.split(pth)
            # return (cls.split(a) if len(a) and len(b) else []) + [b]

        @staticmethod
        def testDir(pth):
            return (path.altsep and pth.endswith(path.altsep)) or pth.endswith(
                path.sep)


class Student(BaseModel):
    first_name = Searchable.CharField()
    last_name = Searchable.CharField()
    mail = CharField(unique=True)
    rep = IntegerField()
    ects = IntegerField()
    user = Searchable.CharField(unique=True)

    @property
    def name(self):
        return self.first_name + " " + self.last_name

    def __str__(self):
        return self.name

    @property
    def generated_user(self):
        return self.last_name.lower()[:7] + self.first_name.lower()[0]


class Category(BaseModel):
    name = Searchable.CharField()

    FutureHints = Record(id=1, name='Hints for future submissions')
    ProjectStructure = Record(id=2, name='Project Structure')
    Warning = Record(id=3, name='Warning')
    Other = Record(id=4, name='Other')

    def __str__(self):
        return "{}: {}".format(type(self).__name__, self.name)


class Exercise(BaseModel):
    nr = Searchable.PrimaryKeyField()
    published = DateField(formats=['%Y-%m-%d', '%d.%m.%Y'])
    due = DateTimeField(formats=['%Y-%m-%d %H:%M:%S', '%d.%m.%Y %H:%M:%S'])

    @property
    def Name(self):
        return "Exercise Sheet {}".format(self.nr)

    def __str__(self):
        return self.Name

    @property
    def max_points(self):
        return sum(c.max_points for c in self.tasks)

    @property
    def root_tasks(self):
        return Task.select().where(Task.parent.is_null(), Task.ex == self)

    def existing_comments(self, s: Student):
        return Comment.select().join(ExerciseComment).where(
            ExerciseComment.ex == self,
            ExerciseComment.student == s)

    def existint_others(self, s: Student):
        return Comment.select().join(ExerciseComment).where(
            ExerciseComment.ex == self,
            ExerciseComment.comment.not_in(self.existing_comments(s)))

    @property
    def all_comments(self):
        return Comment.select().join(ExerciseComment).where(
            ExerciseComment.ex == self,
            ExerciseComment.cat != Category.ProjectStructure) | \
               Comment.select().join(
                   GradingComment).join(Grading).join(Task).where(
                   (Grading.ex == self) | (Task.ex == self))

    @property
    def completed(self):
        return all(GradingStatus(self, s).completed for s in Student)


class Task(BaseModel, Hierarchy):
    name = Searchable.CharField()
    parent = ForeignKeyField('self', related_name='children', null=True)
    ex = ForeignKeyField(Exercise, related_name='tasks', null=True)
    max_points = FloatField(default=0)
    always_processed = BooleanField(default=False)
    hidden_question = BooleanField(default=False)

    Report = Record(name='Report', max_points=2, ex=None)
    Comments = Record(name='Comments', max_points=2, ex=None)
    CodeQuality = Record(name='Code Quality', max_points=2, ex=None,
                         always_processed=True)
    Usability = Record(name='Usability', max_points=2, ex=None,
                       always_processed=True)

    @classmethod
    def general_tasks(cls):
        return cls.select().where(cls.ex.is_null())

    @classmethod
    def general_root_tasks(cls):
        return cls.select().where(cls.ex.is_null(), cls.parent.is_null())

    @classmethod
    def specific_tasks(cls):
        return cls.select().where(cls.ex.is_null(False))

    @property
    def is_general(self):
        return self.ex is None

    @property
    def depth(self):
        return self.parent.depth + 1 if self.parent else 0

    @property
    def visible_depth(self):
        return self.parent.visible_depth + int(
            not self.hidden_question) if self.parent else 0

    @property
    def total_max_points(self):
        return self.max_points + sum(c.total_max_points for c in self.children)

    def get_grading(self, s: Student, ex: Exercise = None):
        queries = [Grading.student == s, Grading.task == self]
        if self.ex is None:
            queries.append(Grading.ex == ex)
        elif self.ex != ex:
            raise DoesNotExist('requested exercise number {} does not '
                               'match task\'s ({})'.format(self.ex, ex))
        try:
            return Grading.get(*queries)
        except DoesNotExist:
            g = Grading()
            g.student = s
            g.task = self
            if not self.ex:
                g.ex = ex
            return g

    @classproperty
    def searchable(cls):
        return [cls, Exercise]

    def __str__(self):
        return "Task " + " > ".join(t.name for t in self.ancestors)

    class Meta:
        indexes = (
            (('name', 'ex'), True),
        )


class GradingStatus:
    def __init__(self, ex: Exercise, s: Student):
        self.ex = ex
        self.stud = s

    @property
    def tasks(self):
        for t in self.root_tasks:
            yield from t.descendants

    @property
    def root_tasks(self):
        yield from Task.general_root_tasks()
        yield from self.ex.root_tasks

    @property
    def general_gradings(self):
        return self.get_desc_gradings(*Task.general_root_tasks())

    @property
    def specific_gradings(self):
        return self.get_desc_gradings(*self.ex.root_tasks)

    @property
    def gradings(self):
        yield from self.general_gradings
        yield from self.specific_gradings

    def get_gradings(self, *tasks):
        for t in tasks:
            try:
                yield t.get_grading(self.stud, self.ex)
            except DoesNotExist:
                pass

    def get_desc_gradings(self, *tasks):
        yield from self.get_gradings(*Hierarchy.get(*tasks))

    def make_failed(self, *tasks):
        return all(g.make_failed() for g in self.get_gradings(*tasks))

    @property
    def total_points(self):
        return sum(g.total_points for g in self.get_gradings(*self.root_tasks)) \
               + sum(c.add_in_total_ex_points for c in self.gradings) \
               + sum(c.point_modifier for c in self.ex_comments)

    @property
    def progress(self):
        return self.Partition((g, g.status) for g in self.gradings)

    @property
    def completed(self):
        return all(g.completed for g in self.gradings)

    @property
    def started(self):
        return any(g.status for g in self.gradings)

    def __str__(self):
        return "{}\n{}".format('completed' if self.completed else 'incomplete',
                               self.progress)

    @property
    def status(self):
        return self.Values(self.started + self.completed)

    @property
    def ex_comments(self):
        return Comment.select().join(ExerciseComment).where(
            ExerciseComment.ex == self.ex,
            ExerciseComment.student == self.stud)

    class Partition:
        def __init__(self, items):
            self.d = {}
            for t, s in items:
                self.d.setdefault(s, [])
                self.d[s].append(t)

        @property
        def all_items(self):
            return [i for v in self.d.values() for i in v]

        def __len__(self):
            return len(self.all_items)

        def items(self, key):
            return self.d.get(key)

        def absolute(self, key):
            return len(self.items(key))

        def relative(self, key):
            return self.absolute(key) / len(self)

        def __str__(self):
            return "\n".join(
                '{k}: {abs}/{tot} ({rel:.1%})'.format(k=k.name,
                                                      abs=self.absolute(k),
                                                      tot=len(self),
                                                      rel=self.relative(k))
                for k in self.d)

    class Values(AddEnum):
        OPEN = 0
        STARTED = 1
        COMPLETED = 2

        def __bool__(self):
            return bool(self.value)


class Comment(BaseModel):
    # id = PrimaryKeyField()
    # task = ForeignKeyField(Task, related_name='comments', null=True)
    message = Searchable.CharField()
    point_modifier = FloatField(default=0)
    visible = BooleanField(default=True)

    ProjectPathMissing = Record(id=1,
                                message="There was nothing committed.",
                                point_modifier=0)
    IdeaProjectFolderMissing = Record(id=2,
                                      message="The {name} is missing! Please commit the whole AS project.",
                                      point_modifier=0)
    CommittedIgnoredItems = Record(id=3,
                                   message="There were items committed that "
                                           "should be ignored.",
                                   point_modifier=-0.5)
    MissingItems = Record(id=4,
                          message="There were important items missing.",
                          point_modifier=-2)
    MissingManifest = Record(id=5, message="The Android Manifest is missing.",
                             point_modifier=-2)
    ReportReminder = Record(id=6, message="Don't forget the report next time!",
                            point_modifier=0)
    # ReportMissing = Record(id=7, message="You have not committed a Report.",
    #                       point_modifier=-2)
    reserved7 = Record(id=7, message="reserved comment 7", point_modifier=0)
    ForgiveCommittedIgnoredItems = Record(id=8,
                                          message="For this exercise it is ok "
                                                  "to have submitted ignored files.",
                                          point_modifier=0.5)
    reserved9 = Record(id=9, message="reserved comment 9", point_modifier=0)

    # The report is not mandatory for the first exercise but keep it in mind for the other exercises.

    def __str__(self):
        visible_str = '' if self.visible else '[invisible] '
        return '{}{} {}'.format(visible_str, self.message, self.point_str)

    @property
    def point_str(self):
        return '({})'.format(
            self.point_modifier) if self.point_modifier else ''

    def format(self, *args, **kwargs):
        self.message = self.message.format(*args, **kwargs)
        return self

    @property
    def categories(self):
        return Category.select().distinct().join(ExerciseComment).where(
            ExerciseComment.id << self.ex_coms)

    @property
    def tasks(self):
        return Task.select().distinct().join(Grading).join(
            GradingComment).where(
            GradingComment.id << self.gradings)

    @property
    def used(self):
        return self.ex_coms.count() + self.gradings.count()

    def get_contexts(self, ex: Exercise, s: Student):
        r = []
        for t in self.tasks:
            try:
                r.append(t.get_grading(s, ex))
            except DoesNotExist:
                pass
        return r + list(self.categories)

    def add_context(self, context, s, ex):
        if isinstance(context, Task):
            g = context.get_grading(s, ex)
            assert g.status, DoesNotExist('Grading missing.')
            return GradingComment.get_or_create(
                grading=g,
                comment=self)
        elif isinstance(context, Category):
            return ExerciseComment.get_or_create(student=s, ex=ex,
                                                 cat=context,
                                                 comment=self)
        else:
            raise TypeError('invalid context: {}'.format(context))

    def move_to(self, context):
        queries = {q.model_class: list(q) for q in
                   (GradingComment.select().where(
                       GradingComment.comment == self),
                    ExerciseComment.select().where(
                        ExerciseComment.comment == self))}

        moved = {
            GradingComment: 0,
            ExerciseComment: 0
        }

        for c, l in queries.items():
            for rel in list(l):
                s, ex = rel.student, rel.ex
                added, nrel = self.add_context(context, s, ex)
                if added:
                    rel.delete_instance()
                    moved[c] += 1

        return {k.__name__:
                    {'moved': v, 'untouched': len(list(queries[k])) - v}
                for k, v in moved.items()}

    def used_for(self, s: Student, ex: Exercise):
        q1where = (ExerciseComment.student == s, ExerciseComment.ex == ex)
        q1 = self.categories.where(*q1where)

        q2where = (Grading.student == s, (Task.ex == ex) | (Grading.ex == ex))
        q2 = self.tasks.where(*q2where)
        r_lst = list(q1) + list(q2)

        return len(r_lst)

    def used_in_Exercises(self, s: Student = None):
        q1where = [ExerciseComment.id << self.ex_coms]
        q2where = [GradingComment.id << self.gradings]

        if s is not None:
            q1where.append(ExerciseComment.student == s)
            q2where.append(Grading.student == s)

        q1 = Exercise.select().join(ExerciseComment).where(*q1where)
        q2 = Grading.select().join(GradingComment).where(*q2where)

        r = []
        for e in list(q1) + [g.true_ex for g in q2]:
            if e not in r:
                r.append(e)

        return r

    @classmethod
    def get_unused(cls):
        return [c for c in cls if not c.used]

    @classmethod
    def get_unused_noconst(cls):
        return [c for c in cls if
                not (c.used or c.id in [cn.id for cn in
                                        cls.const_records().values()])]

    @classproperty
    def searchable(cls):
        return [cls, Task, Category]

    class Created(AddEnum):
        Existed = 0
        Added = 1
        New = 2

        def __str__(self):
            return [
                'The comment existed and had already been added.',
                'The comment existed, but now was added.',
                'The comment was created and added.',
            ][self.value]


class Grading(BaseModel, Hierarchy):
    student = ForeignKeyField(Student)
    task = ForeignKeyField(Task, related_name='gradings')
    ex = ForeignKeyField(Exercise, null=True)
    processed = BooleanField(
        help_text='Has the student has worked on the task at all?')
    completed = BooleanField(default=False)
    point_modifier = FloatField(default=0)

    def existing_comments(self):
        return Comment.select().join(GradingComment).where(
            GradingComment.grading != self).join(Grading).where(
            Grading.task == self.task)

    @property
    def true_ex(self):
        return self.ex or self.task.ex

    def __str__(self):
        return "{}: {} {}".format(self.task.name, self.print_processed,
                                  self.print_points)

    @property
    def visible(self):
        return not self.task.hidden_question
        # (not self.task.hidden_question) or self.lost_points

    @Hierarchy.DescendantMethod
    def make_failed(self):
        self.processed = False
        self.completed = True
        return self.save()

    @Hierarchy.AncestorMethod
    def uncomplete(self):
        self.completed = False
        return self.save()

    @Hierarchy.AncestorMethod
    def make_processed(self):
        self.processed = True
        return self.save()

    def complete(self):
        self.completed = True
        return self.save() and all(g.complete() for g in self.children)

    @property
    def print_processed(self):
        if self.task.always_processed:
            return ""
        return "Ok." if self.processed else "Missing."

    @property
    def zero_points(self):
        return not (self.processed or self.task.always_processed)

    @property
    def add_in_total_ex_points(self):
        return (- self.task.max_points) if self.task.is_general else 0

    @property
    def points(self):
        if self.zero_points:
            return 0
        else:
            return self._point_modifiers + self.task.max_points

    @property
    def _point_modifiers(self):
        return self.point_modifier + sum(
            gc.comment.point_modifier for gc in self.comments)

    @property
    def children(self):
        for st in self.task.children:
            try:
                yield st.get_grading(self.student, self.task.ex)
            except DoesNotExist:
                pass

    @property
    def parent(self):
        if self.task.parent:
            return self.task.parent.get_grading(self.student, self.task.ex)

    @property
    def total_points(self):
        result = 0
        if not self.zero_points:
            result += self.points + sum(c.total_points for c in self.children)
            # sum(c.total_points for c in self.children)
        if result < 0 and self.task.total_max_points:
            print('points for {} were {}, cut off at 0.'.format(self.task,
                                                                result))
            return 0
        return result

    @property
    def lost_points(self):
        return self.total_points < self.task.total_max_points

    @property
    def print_points(self):
        if self.task.total_max_points:
            s = '({} / {} points)'
        elif self.total_points:
            s = '({} points)'
        else:
            s = ''
        return s.format(self.total_points,
                        self.task.total_max_points)

    @property
    def status(self):
        return GradingStatus.Values(
            (self.processed is not None) + self.completed)

    class Meta:
        indexes = (
            (('student', 'task', 'ex'), True),
        )


class GradingComment(BaseModel):
    grading = ForeignKeyField(Grading, related_name='comments')
    comment = ForeignKeyField(Comment, related_name='gradings')

    @property
    def context(self):
        return self.grading

    @property
    def student(self):
        return self.grading.student

    @property
    def ex(self):
        return self.grading.ex if self.grading.ex is not None \
            else self.grading.task.ex


class ExerciseComment(BaseModel):
    student = ForeignKeyField(Student)
    ex = ForeignKeyField(Exercise, related_name='used_ex_comments')
    cat = ForeignKeyField(Category)
    comment = ForeignKeyField(Comment, related_name='ex_coms')

    @property
    def context(self):
        return self.cat


class ExerciseQuestion(BaseModel):
    ex = ForeignKeyField(Exercise, related_name='ex_questions')
    question = CharField()


class ExtraProjects(BaseModel):
    name = Searchable.CharField()
    ex = ForeignKeyField(Exercise, related_name='projects')


class Directory(BaseModel, DirectoryBase):
    urltest = re.compile("^https?://")
    REP_STRING = "asp{rep:02d}"

    name = Searchable.CharField(unique=True)
    path = CharField()
    parent = ForeignKeyField('self', related_name='children', null=True,
                             to_field='name')
    svn_rep = CharField(null=True)
    file_type = ForeignKeyField(FileType, null=True, default=FileType.DIR)

    # self.urltest.match(self.path):

    @property
    def absolute(self, *others):
        pathlist = [getcwd()] + [d.path for d in self.ancestors] + list(
            others)
        abs_path = path.abspath(path.join(*pathlist))
        return abs_path

    @property
    def relative(self):
        return self.rel_root(self)

    @property
    def type(self):
        return self.Type(self.path.endswith('/'))

    @property
    def repository_dir(self):
        return next(x for x in self.ancestors if x.svn_rep)

    @property
    def repository(self):
        return LocalClient(self.repository_dir.absolute)

    @property
    def Name(self):
        return self.name + ' ' + self.type.name.lower()

    root = Record(name="root", path="..")
    student = Record(name="repository {rep}",
                     path="svn/{}/".format(REP_STRING),
                     parent=root, svn_rep="https://proglang.informatik.uni-"
                                          "freiburg.de/svn/asp{rep:02d}")
    internal = Record(name="internal repository",
                      path="svn/internal/",
                      parent=root, svn_rep="https://proglang.informatik.uni-"
                                           "freiburg.de/svn/proglang/teaching/"
                                           "AndroidSmartphoneProgramming/2015")

    exercise_pdf = Record(name="exercise sheet",
                          path="Material/ex/{Name}.pdf",
                          file_type=FileType.PDF,
                          parent=root)
    old_mistakes = Record(name="mistakes from former exercises",
                          path="old/2014/Mistakes_Ex{nr}.txt",
                          file_type=FileType.TXT,
                          parent=internal)
    grading_key = Record(name="grading key",
                         path="grading/ex{nr}.txt",
                         file_type=FileType.TXT,
                         parent=internal)

    grading_pre_folder = Record(name="preliminary gradings",
                                path="gradings/ex{nr}/",
                                parent=internal)
    grading_pre = Record(name="preliminary grading",
                         path="asp{rep:02d}_{user}.txt",
                         file_type=FileType.TXT,
                         parent=grading_pre_folder)
    grading_notes = Record(name="grading notes",
                           path="Notes.txt",
                           file_type=FileType.TXT,
                           parent=grading_pre_folder)
    grading_report = Record(name="gradings report",
                            path="grading.txt",
                            file_type=FileType.TXT,
                            parent=internal)
    grading = Record(name="student grading",
                     path="grading/ex{nr}.txt",
                     file_type=FileType.TXT,
                     parent=student)

    ProjectPath = Record(name="AS Project",
                         path="exercise{nr}/",
                         file_type=FileType.ASP,
                         parent=student)
    Report = Record(name='Report',
                    path='{user}_report{nr}.pdf',
                    file_type=FileType.PDF,
                    parent=ProjectPath)
    IdeaProjectFolder = Record(name='.idea Project',
                               path='.idea/',
                               parent=ProjectPath)
    AppDir = Record(name='App',
                    path="app/",
                    parent=ProjectPath)
    Manifest = Record(name="Manifest",
                      path="src/main/AndroidManifest.xml",
                      file_type=FileType.XML,
                      parent=AppDir)
    StringsXML = Record(name="Strings XML",
                        path="src/main/res/values/strings.xml",
                        file_type=FileType.XML,
                        parent=AppDir)
    ProjectIml = Record(name='project iml',
                        path='{projectname}.iml',
                        file_type=FileType.OTH,
                        parent=ProjectPath)
    AppIml = Record(name='app iml',
                    path='app.iml',
                    file_type=FileType.OTH,
                    parent=AppDir)
    GradleBuild = Record(name='build.gradle',
                         path='build.gradle',
                         file_type=FileType.OTH,
                         parent=ProjectPath)
    GradleBuildApp = Record(name='(app) build.gradle',
                            path='build.gradle',
                            file_type=FileType.OTH,
                            parent=AppDir)
    GradleSettings = Record(name='gradle settings',
                            path='settings.gradle',
                            file_type=FileType.OTH,
                            parent=ProjectPath)
    PackageFolders = Record(name='Package',
                            path='src/androidTest/java/androidlab/{user}/exercise{nr}/',
                            parent=AppDir)

    def stud_ex_path(self, s, ex, project=None):
        kw = dict(directory=self, student=s, ex=ex)
        if project is None:
            args = (CustomStudenPath.project.is_null(),)
        else:
            args = ()
            assert isinstance(project, ExtraProjects), TypeError
            assert project.ex == ex, IntegrityError
            kw['project'] = project
        try:
            return CustomStudenPath.get(*args, **kw)
        except:
            return CustomStudenPath(*args, **kw)

    @classmethod
    def rel_root(cls, pth):
        if isinstance(pth, cls):
            pth = pth.absolute
        return path.relpath(pth, cls.root.absolute)


class CustomStudenPath(BaseModel, DirectoryBase):
    # CURRENT_PROJECT = None
    # DEFAULT_PROJECT = 'exercise{nr}/'

    directory = ForeignKeyField(Directory)
    student = ForeignKeyField(Student)
    project = ForeignKeyField(ExtraProjects, null=True, default=None)
    ex = ForeignKeyField(Exercise)
    custom_path = CharField()

    @property
    def parent(self):
        r = self.directory.parent
        if isinstance(r, Directory):
            return r.stud_ex_path(self.student, self.ex, self.project)

    @property
    def path(self):
        format_vars = self.student.format_vars
        format_vars.add(self.ex.format_vars)
        if self.directory != Directory.ProjectPath:
            pp = Directory.ProjectPath.stud_ex_path(self.student, self.ex,
                                                    self.project).path
            if pp.endswith(path.sep):
                pp = pp[:-1]
            format_vars['projectname'] = path.split(pp)[-1]

        return (self.custom_path or self.directory.path).format(**format_vars)

    @property
    def Name(self):
        return self.directory.Name

    @property
    def file_type(self):
        return self.directory.file_type

    def fix_path(self):
        if self.exists and path.isdir(
                self.absolute) and not self.custom_path.endswith(path.sep):
            self.custom_path += path.sep
            return True
        return False

    @property
    def repository_dir(self):
        return self.directory.repository_dir.stud_ex_path(self.student,
                                                          self.ex,
                                                          self.project)

    @property
    def repository(self):
        return LocalClient(self.repository_dir.absolute)

    @property
    def absolute(self):
        pathlist = [getcwd()] + [d.path for d in
                                 self.ancestors]
        return path.abspath(path.join(*pathlist))

    def get_svn_info(self):
        return self.repository.info(self.relative_to_repo)

    @property
    def original(self):
        return not self.custom_path  # self.is_dirty()

    @property
    def status(self):
        v = 0
        try:
            if self.exists:
                v += 1
                if self.get_svn_info():
                    v += 1
                if self.original:
                    return self.Status.Exists
        except:
            pass
        return self.Status(v)

    @property
    def status_string(self):
        return str(self.status).format(self.Name)

    @property
    def Name(self):
        return self.directory.name + ' ' + self.type.name.lower()

    def append(self, tail):
        self.custom_path = path.join(self.path, tail)

    # def __iadd__(self, other):

    def __add__(self, other):
        c = self.clone()
        c.append(other)
        return c

    class Status(AddEnum):
        DoesNotExist = 0
        ExistsNoSVN = 1
        ExistsChanged = 2
        Exists = 3

        @property
        def comment(self):
            return [
                'The {name} is missing. ({rel_original})',
                'The {name} is missing (but was added to the repository later - this should not be in a final grading!).',
                'The {name} is at {rel_custom} instead of the expected location {rel_original}.',
                None,
            ][self.value]

        def __str__(self):
            return [
                'The {} is missing in the filesystem.',
                'The {} is missing in the repository.',
                'The {} is present (custom location).',
                'The {} is present.',
            ][self.value]

    @classproperty
    def searchable(cls):
        return [cls, Student, Exercise]

    class Meta:
        indexes = (
            (('directory', 'student', 'ex'), False),
            (('directory', 'student', 'ex', 'project'), True),
        )


def model_filter(obj):
    return issubclass(type(obj), type) and \
           issubclass(obj, BaseModel) and \
           obj is not BaseModel

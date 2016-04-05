from builtins import property
from os import remove
from enum import unique
from models import *
from parameter import *
from peewee import attrdict
from shutil import copy2
from svn.local import LocalClient
from svn.errors import SVNError, SVNErrorCode
from inspect import getmro
from ascii import wrap_indent, open_file
from utils import EnumMember  # classproperty
from datetime import datetime


def svn_cmd(rep, cmd, *args):
    cmd_args = list(args)
    try:
        r = rep.run_command(cmd, cmd_args, return_binary=True)
        # self.svn_cmd(*upd_cmd)
    except (SVNErrorCode.RA_SERF_SSL_CERT_UNTRUSTED.errcls,
            SVNErrorCode.AUTHN_FAILED.errcls,) as e:
        yield Message(e)
        if (yield ask("set username and password?")):
            # upd_cmd.append('--trust-server-cert')
            username = yield TypePar(str, prompt='username:')
            cmd_args += ['--username', username]
            password = yield TypePar(str, prompt='password:')
            cmd_args += ['--password', password]
            r = rep.run_command(cmd, cmd_args, return_binary=True)

    print(r.decode())


@unique
class Level(Enum):
    LOG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3


class Message:
    def __init__(self, msg, confirm=False, level: Level = Level.INFO):
        self.msg = str(msg)
        self.confirm = confirm
        self.level = level

    @classmethod
    def inject(cls, msg):
        if isinstance(msg, cls):
            return msg
        else:
            return cls(msg)

    def __repr__(self):
        return "[{}]{}".format(self.level, self.msg)


class ActionMeta(ABCMeta):
    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)

        assert ActionMeta.is_unique([p.name for p in cls.parameters()]), \
            'Parameter Names of Action {} not unique!'.format(name)

    @staticmethod
    def is_unique(l: list):
        return len(l) == len(set(l))


class Action(metaclass=ActionMeta):
    def __init__(self, *args, **kwargs):
        self.par_vals = attrdict(zip(self.named_params(), args))
        self.par_vals.update(kwargs)
        self._paths = attrdict()

    def update_paths(self):
        paths = attrdict()
        args = (getattr(self, 'stud', Student.get()),
                getattr(self, 'ex', Exercise.get()),
                getattr(self, 'current_project', None))
        for n, d in Directory.const_records().items():
            paths[n] = d.stud_ex_path(*args)
        self._paths = paths

    @property
    def paths(self):
        if not len(self._paths):
            self.update_paths()
        return self._paths

    @classmethod
    def parameters(cls):
        return [Parameter.inject(p) for c in getmro(cls) if
                '_parameters' in c.__dict__
                for p in getattr(c, '_parameters')]

    @classmethod
    def _make_unique(cls, name, existing):
        if name not in existing:
            return name
        else:
            repl = lambda m: m.group(1) + str(1 + int('0' + m.group(2)))
            n = re.sub(r"(\D+)(\d*)", repl, name)
            return cls._make_unique(n, existing)

    @classmethod
    def named_params(self):
        return dict(zip(self.param_names(), self.parameters()))

    @classmethod
    def param_names(self):
        names = []
        for p in self.parameters():
            name = self._make_unique(p.name, names)
            names.append(name)
        return names

    @property
    def format_vars(self):
        d = {}
        for v in self.par_vals.values():
            if isinstance(v, BaseModel):
                d.update(v.format_vars)
        return d

    @property
    def internal_repository(self):
        return LocalClient(Directory.internal.absolute)

    @abstractmethod
    def execute(self):
        raise NotImplementedError('To be overridden!')

    @classmethod
    def init_exec(cls, *args, **kwargs):
        new = cls(*args, **kwargs)
        return new.execute()

    def other_action(self, OtherAction, **kwargs):
        other_kwargs = self.par_vals.copy()
        other_kwargs.update(kwargs)
        return OtherAction(**other_kwargs)

    @classmethod
    def pick_from_model(self, model: BaseModel):
        mname = model.__name__
        while True:
            rm = yield ListPar(model.searchable, 'pick_' + mname,
                               prompt='Please select what data you want to specify'
                                      ' to narrow down your search for a {}'
                                      ''.format(mname))
            clause_par = yield DataPar(rm)
            if rm == model:
                return clause_par
            m_path = model.related_paths()[rm]
            query = reduce(lambda b, x: b.join(x), m_path[1:],
                           m_path[0].select())
            if len(rm._meta.get_primary_key_fields()) != 1:
                raise ActionError('Invalid model: {}\n'
                                  'only models with a single primary key '
                                  'are supported.'.format(rm))
            where_clause = rm._meta.primary_key == clause_par

            query = query.where(where_clause)
            lq = list(query)
            if len(lq):
                if len(lq) == 1:
                    yield Message(
                        'There is exactly one records in {} that is '
                        'associated to {}'.format(model.__name__,
                                                  clause_par))
                    return lq[0]
                return (yield DataPar(model, query))
            else:
                yield Message(
                    'There are no records in {} that are '
                    'associated to {}'.format(model.__name__, clause_par))

    class Cancel(IntEnum):
        cancel = 0


class FileWriter:
    def ask_commit(self):
        if (yield ask('commit {}?'.format(self.file.Name))):
            yield from svn_cmd(self.file.repository, 'ci', '-m',
                               'Updated {}'.format(self.file.Name),
                               self.file.absolute)

    def write(self):
        do_remove = False
        if self.file.exists:
            yield Message('Overwriting {}'.format(self.file.absolute))
            copy2(self.file.absolute, self.file_old_name)
            show_diff = self.file_old_name
            do_remove = True
        elif self.diff_file.exists:
            show_diff = self.diff_file.absolute
        else:
            show_diff = False

        yield from self.write_output()

        if show_diff:
            p_meld = FileCommand.DIFF.open(self.file.absolute,
                                           show_diff)
            p_meld.wait()
        else:
            p_text = self.file.open()
            p_text.wait()

        if do_remove:
            remove(self.file_old_name)

        try:
            yield from svn_cmd(self.file.repository, 'add', self.file.absolute)
        except SVNErrorCode.ILLEGAL_TARGET.errcls:
            yield Message(
                '{} was already added to repository.'.format(
                    self.file.absolute))

    def write_output(self):
        with open(self.file.absolute, 'w') as file:
            file.write(self.output)

        yield Message(
            'Successfully written {} to {}'.format(self.file.Name,
                                                   self.file.absolute))

    @property
    def output(self):
        raise NotImplementedError

    @property
    def file(self):
        raise NotImplementedError

    @property
    def diff_file(self):
        return self.file

    @property
    def file_old_name(self):
        return "{}.old".format(self.file.absolute)


class ExAction(Action):
    _parameters = [Exercise]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ex = self.par_vals.Exercise
        """:type : Exercise"""

    @property
    def taskDataPar(self):
        return DataPar(Task.name, Task.select().where(
            Task.ex.is_null() | (Task.ex == self.ex)))


class StudExAction(ExAction):
    _parameters = [Student.user]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stud = self.par_vals.Student
        """:type : Student"""

        self.current_project = None
        self.update_paths()
        self.repository = self.paths.student.repository

    def get_current_project(self, lookup=True):
        if self.extra_projects:
            self.current_project = (yield SumPar([None], self.ex.projects,
                                                 name='current_project',
                                                 remember=True,
                                                 lookup=lookup))
            self.update_paths()

    def set_current_project(self, value):
        self.current_project = value
        self.update_paths()
        yield {'current_project': value}

    @property
    def status(self):
        return GradingStatus(self.ex, self.stud)

    @property
    def extra_projects(self):
        return list(self.ex.projects)

    @property
    def get_grading(self):
        t = yield self.taskDataPar
        return t.get_grading(self.stud, self.ex)

    def add_comment_str(self, context, message: str = "Empty Comment",
                        point_modifier: float = 0, return_message=True):
        c, new = Comment.get_or_create(message=message,
                                       point_modifier=point_modifier)

        s = self.add_comment(context, c, False) + new

        if return_message:
            return Message("{}\n -{!s}".format(c.message, s))
        return c, s

    def add_comment(self, context, comment: Comment, return_message=True):

        if isinstance(context, Grading):
            ce, added = GradingComment.get_or_create(grading=context,
                                                     comment=comment)
        elif isinstance(context, Category):
            ce, added = ExerciseComment.get_or_create(student=self.stud,
                                                      ex=self.ex,
                                                      cat=context,
                                                      comment=comment)
        else:
            raise TypeError(
                'Invalid context type: {}\nExpected {} or {}'.format(
                    type(context), Grading, Category))

        s = Comment.Created(added)

        if return_message:
            return Message("{}\n -{!s}".format(comment.message, s))
        return s

    def add_comment_nocontext(self, comment: Comment):
        cs = comment.get_contexts(self.ex, self.stud)
        if not cs:
            p = SumPar(self.Cancel, DataPar(Category), self.taskDataPar,
                       prompt='No contexts found for this exercise. '
                              'Pick a context:')
            if p == self.Cancel.cancel:
                raise StopIteration
            context = yield p
            if isinstance(context, Task):
                context = context.get_grading(self.stud, self.ex)
        else:
            context = cs[0]
            if len(cs) > 1:
                context = yield ListPar(list(cs), 'context')
            else:
                yield Message('found context: {}'.format(context))

        if isinstance(context, Grading) and not context.status:
            yield Message('starting grading on {}'.format(context.task))
            context.processed = True
            context.save()

        yield self.add_comment(context, comment)

    def del_comment(self, comment: Comment, context=None):
        if not isinstance(context, BaseModel):
            context = comment.get_contexts(self.ex, self.stud)[0]

        try:
            if isinstance(context, Grading):
                gc = GradingComment.get(grading=context,
                                        comment=comment)
            elif isinstance(context, Category):
                gc = ExerciseComment.get(student=self.stud, ex=self.ex,
                                         cat=context,
                                         comment=comment)
            else:
                raise TypeError('invalid context: {}'.format(context))
        except DoesNotExist:
            return Message(
                "comment '{}' is not part of {}".format(comment.message,
                                                        context))
        else:
            deleted = gc.delete_instance()
            notstr = '' if deleted else 'not '
            return Message(
                "comment '{}' was {}deleted from {}".format(
                    comment.message,
                    notstr,
                    context))

    def comment_loop(self, comment_selector, add_action,
                     comment_delete_selector=None):
        if comment_delete_selector is None:
            comment_delete_selector = comment_selector
        choice_par = SumPar(self.CommentOption,
                            DataPar(Comment.message, comment_selector()),
                            prompt='Would you like to add a Comment?')  # \n{cmds}
        while True:
            choice = yield choice_par
            if choice == self.CommentOption.no_more_comments:
                break
            elif choice == self.CommentOption.delete_comments:
                choice = yield EnumPar(self.Cancel) | DataPar(Comment.message,
                                                              comment_delete_selector())
                if choice == self.Cancel.cancel:
                    continue

                yield self.del_comment(choice, add_action)

                if not choice.used and (
                        yield ask('Comment "{}" is no longer used. '
                                  'delete it?'.format(choice))):
                    if not choice.delete_instance():
                        yield Message('deletion failed.')
            else:
                if choice == self.CommentOption.add_new_comment:
                    choice = yield AddNewComment
                elif choice == self.CommentOption.pick_from_all_comments:
                    choice = yield EnumPar(self.Cancel) | Comment.message
                    if choice == self.Cancel.cancel:
                        continue
                yield Message('Comment added.')
                yield {'Comment': choice}
                if isinstance(add_action, type):
                    yield add_action
                else:
                    print(
                        'addaction: {}\nchoice: {}'.format(add_action,
                                                           type(choice)))
                    if callable(add_action):
                        add_action = yield add_action(choice)
                    yield self.add_comment(add_action, choice)

                    # print('grading: {} ({}, type: {})'.format(str(grading), grading,
                    #                                          type(grading)))

    class CommentOption(AddEnum):
        add_new_comment = 0
        pick_from_all_comments = 1
        no_more_comments = 2
        delete_comments = 3

    @property
    def msg(self):
        return "{} for Student {} ({},asp-{})".format(self.ex.Name,
                                                      self.stud.name,
                                                      self.stud.user,
                                                      self.stud.rep)

    @property
    def short(self):
        return "{}, ex {}".format(self.stud.user, self.ex.nr)

    def svn_cmd(self, cmd, *args, **kwargs):
        rep = kwargs.get('rep', self.repository)

        yield from svn_cmd(rep, cmd, *args)

    def __str__(self):
        return "<{}: {}>".format(self.__name__, self.msg)


class TaskAction(StudExAction):
    _parameters = [
        DataPar(Task.name),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = self.par_vals['Task']
        self.grading = self.task.get_grading(self.stud, self.ex)


class ConfirmGrading(StudExAction, FileWriter):
    _parameters = [EnumPar(YesNo, 'rewrite', 'rewrite existing files '
                                             '(otherwise might contain '
                                             'invisible comments)?')]

    def execute(self):
        if self.par_vals.rewrite:
            yield dict(show_invisible_comments=False)
            self._output = yield OutputGrading
            yield from self.write()
        else:
            yield from self.exec_copy()

        yield from self.exec_commit()

    @property
    def file(self):
        return self.paths.grading

    @property
    def output(self):
        return self._output

    @property
    def diff_file(self):
        return self.paths.grading_pre

    def exec_copy(self):
        source = self.paths.grading_pre.absolute
        ci_msg = 'grading for {Name}'.format(**self.format_vars)
        yield Message('copy from \n"{}"\n\t '
                      'to \n"{}"'.format(source, self.file.absolute))

        copy2(source, self.file.absolute)
        # self.repository.update()

    def exec_commit(self):
        target = self.file.absolute
        ci_msg = 'grading for {Name}'.format(**self.format_vars)
        yield from self.svn_cmd('revert', target)
        try:
            yield from self.svn_cmd('add', target)
        except SVNErrorCode.ILLEGAL_TARGET.errcls:
            do_commit = yield ask('{} was already added to repository. '
                                  'Do you want to commit the file '
                                  'anyways?'.format(target))
        else:
            do_commit = True

        if do_commit:
            yield Message("Committing grading for {} "
                          "with message: {}".format(self.stud.name, ci_msg))
            try:
                yield from self.svn_cmd('ci', '-m', ci_msg,
                                        target)
            except SVNError as e:
                yield Message(str(e))


class ConfirmExGrading(ExAction):
    _parameters = [EnumPar(YesNo, 'rewrite', 'rewrite existing files '
                                             '(otherwise might contain '
                                             'invisible comments)?')]

    def execute(self):
        sl = list(Student)
        for s in sl:
            yield dict(Student=s)
            yield ConfirmGrading


class CommitPreGradings(ExAction):
    def execute(self):
        target = self.paths.grading_pre_folder

        if not target.exists:
            yield Message(
                "{} doesn't exist: {}".format(target.name,
                                              target.absolute))
            raise StopIteration

        yield from svn_cmd(self.internal_repository, 'ci', '-m',
                           'gradings for {}'.format(self.ex.Name),
                           target.absolute)


class WriteGrading(StudExAction, FileWriter):
    @property
    def file(self):
        return self.paths.grading_pre

    @property
    def output(self):
        return self._output

    def execute(self):
        if not self.status.completed:
            yield Message(
                'Correction incomplete.\n{}'.format(self.status.progress))

        self._output = yield OutputGrading
        yield from self.write()


class WriteCommitGrading(WriteGrading):
    def execute(self):
        yield from super().execute()
        yield from self.ask_commit()


class WriteGradings(ExAction):
    def execute(self):
        for s in Student:
            status = GradingStatus(self.ex, s)
            if status.completed:
                yield dict(Student=s, show_invisible_comments=False)
                yield WriteGrading
            else:
                yield Message(
                    'Correction incomplete.\n{}'.format(status.progress))

        if (yield ask('commit all files?')):
            yield from self.other_action(CommitPreGradings).execute()


class UpdateGradingsReport(Action, FileWriter):
    ENTRY_STRING = "Exercise{nr}: {}"
    SPLITTER = re.compile("(^\*\*\* \w+, \w+ \(\w+\d{2}\)$)", re.M)
    PARSE_NAME = "^\*\*\* {last_name}, {first_name} \(" + Directory.REP_STRING + "\)$"
    PARSE_ENTRY = "^" + ENTRY_STRING + "$"

    @property
    def file(self):
        return Directory.grading_report

    @property
    def output(self):
        return self._output

    def execute(self):
        text = open_file(self.file.absolute)
        text_split = self.SPLITTER.split(text)

        for s in Student:
            stud_pat = self.PARSE_NAME.format(**s.format_vars)
            for i, text_slice in enumerate(text_split):
                if re.match(stud_pat, text_slice, re.M):
                    self._process_student(s, i + 1, text_split)

        # print(''.join(text_split))
        self._output = ''.join(text_split)
        yield from self.write()
        yield from self.ask_commit()

    def _process_student(self, s, i, text_split):
        for e in Exercise:
            text_slice = text_split[i]
            gs = GradingStatus(e, s)
            e_str = str(gs.total_points)
            if not gs.started:
                continue
            elif not gs.completed:
                e_str += ' [grading {}]'.format(gs.status.name.lower())
            e_par = self.PARSE_ENTRY.format('.*', **e.format_vars)
            e_rpl = self.ENTRY_STRING.format(e_str, **e.format_vars)

            text_split[i] = re.sub(e_par, e_rpl, text_slice, flags=re.M)


class OpenGrading(StudExAction):
    @property
    def file(self):
        return self.paths.grading_pre

    def execute(self):
        if not self.file.exists:
            yield Message(
                'no grading exists for {}.'.format(self.msg))
        else:
            p_text = self.file.open()
            p_text.wait()


class OutputGrading(StudExAction):
    INDENT = "    "
    show_invisible_comments = True

    def execute(self):
        if not all(c.visible for c in self.status.all_comments):
            self.show_invisible_comments = yield EnumPar(YesNo, 'show_invisible_comments',
                                                         'include invisible comments?', remember=True)
        return "\n".join(self.content_string)

    @property
    def content_string(self):
        yield "achieved points: {} of {} possible points".format(
            max(self.status.total_points, 0),
            self.ex.max_points)
        # yield "grading key:"

        if self.status.ex_comments.count():
            yield "\n\nGeneral Comments:"
            for cat in Category:
                comments = self.status.ex_comments.where(
                    ExerciseComment.cat == cat)
                if comments.count():
                    yield cat.name
                    for c in comments:
                        yield from wrap_indent(c, " - ")

        yield "\n"
        for g in self.status.general_gradings:
            yield from self.grading_string(g)
        yield "\n\n\n"
        for g in self.status.specific_gradings:
            yield from self.grading_string(g)

    def grading_string(self, g):
        indent = g.task.visible_depth * self.INDENT
        if g.visible:
            yield indent + str(g) + (' [grading {}]'.format(
                g.status.name) if not g.completed else '')
        for gc in g.comments:
            c = gc.comment
            if c.visible or self.show_invisible_comments:
                yield from wrap_indent(c, indent + ' - ')
                # for c in [gc.comment for gc in g.comments if gc.comment.visible]:


class AllExComments(ExAction):
    INDENT = "    "

    def execute(self):
        return "\n".join(self.content_string)

    @property
    def content_string(self):
        for c in self.ex.all_comments:
            yield from wrap_indent(c, "-- ")


class MoveComments(Action):
    _parameters = [Comment]

    def execute(self):
        source = self.par_vals.Comment
        # source = yield from self.pick_from_model(Comment)

        found = ', '.join(list(source.categories) +
                          list(t.name for t in source.tasks))
        yield Message('Found uses: {}'.format(found))
        target_type = yield ListPar([Category, Task], 'target_type')
        target = yield DataPar(target_type)
        moved = source.move_to(target)
        yield Message(str(moved))


class DeleteComments(Action):
    _parameters = [Comment]

    class Options(IntEnum):
        no = 0
        yes = 1
        view_usages = 2

    def execute(self):
        c = self.par_vals.Comment
        if c.used:
            proceed = yield EnumPar(self.Options,
                                    prompt='The comment has {} usages.'.format(
                                        c.used))  # source = yield from self.pick_from_model(Comment)
            if not proceed:
                raise StopIteration
            if proceed == self.Options.view_usages:
                for ec in c.ex_coms:
                    ecs = "{} {}: {}".format(ec.ex, ec.student, ec.category)
                    yield Message(ecs)
                for cg in c.gradings:
                    g = cg.grading
                    yield Message("{} {}: {}".format(g.true_ex, g.student, g))
                if not (yield ask('proceed with deletion?')):
                    raise StopIteration

        if c.delete_instance(recursive=True):
            yield Message("comment {} has been deleted.".format(c))
        else:
            yield Message("Something went wrong.")


class MergeComments(Action):
    _parameters = [Comment, DataPar(Comment, name='Comment2')]

    class Options(IntEnum):
        no = 0
        yes = 1
        view_usages = 2

    def execute(self):
        source, target = self.par_vals.Comment, self.par_vals.Comment2
        for c in [source, target]:
            if not c.used:
                yield Message('The comment is not used: "{}"'.format(c))
                raise StopIteration

        save_list = []
        for ec in source.ex_coms:
            ecs = "{} {}: {}".format(ec.ex, ec.student, ec.category)
            save_list.append((ec, ecs))
        for cg in source.gradings:
            g = cg.grading
            save_list.append((cg, "{} {}: {}".format(g.true_ex, g.student, g)))

        for d, s in save_list:
            d.comment = target
            yield Message(
                '[{}] {}'.format('SAVED' if d.save() else 'FAILED', s))

        if source.used:
            yield Message('"{}"\nThe comment is used '
                          '{} times now.'.format(source, source.used))
        else:
            yield Message('Sucessfully '
                          'merged\n"{}" with \n"{}".'.format(source, target))


class RenameGrading(StudExAction):
    _parameters = [TypePar(str, "filename")]

    def execute(self):
        target = Directory.grading_pre.absolute.format(**self.format_vars)
        baseDir = path.dirname(target)
        source = path.join(baseDir, self.par_vals['filename'].format(
            **self.format_vars))
        print('rename from "{}" to "{}"'.format(source, target))
        yield from self.svn_cmd('rename', source, target)


class UpdateRepo(StudExAction):
    def execute(self):
        yield Message(
            'Updating Repository {} to date {}'.format(
                self.repository.path,
                self.ex.due.strftime(
                    '%x %X')))

        yield from self.svn_cmd('revert', '-R', self.repository.path)

        upd_cmd = ['update', '-r', '{' + self.ex.due.isoformat() + '}',
                   self.repository.path]
        yield from self.svn_cmd(*upd_cmd)

        try:
            last_ex = Exercise.get(nr=self.ex.nr - 1)
            svn_log = self.repository.log_default(last_ex.due, self.ex.due)
        except:
            svn_log = self.repository.log_default(datetime(2000, 1, 1),
                                                  self.ex.due)

        for c in svn_log:
            yield Message(str(c))


class UpdateRepToDate(Action):
    _parameters = [Student.user, DatePar(name='date')]

    def execute(self):
        repo = Directory.student.stud_ex_path(self.par_vals.Student,
                                              Exercise()).repository
        date = self.par_vals['date']

        yield Message(
            'Updating Repository {} to date {}'.format(repo.path,
                                                       date.strftime(
                                                           '%x %X')))

        upd_cmd = ['update', '-r', '{' + date.isoformat() + '}',
                   repo.path]
        yield from svn_cmd(repo, *upd_cmd)

        limit = 3
        svn_log = repo.log_default(datetime(2000, 1, 1), date)
        yield Message('last {} commit messages:'.format(limit))

        for c in list(svn_log)[-3:]:
            yield Message("{:08d} ({} by {}):   {}".format(c.revision,
                                                           c.date.strftime(
                                                               '%x %X'),
                                                           c.author,
                                                           c.msg))


class BackupDB(Action):
    def execute(self):
        ts = int(datetime.now().timestamp())
        target = backup_pattern.format(ts)
        copy2(db.database, target)
        yield Message('created backup of database: {}'.format(target))


class BackupRestoreDB(Action):
    _parameters = [PathPar('.', 'backup', backup_matcher)]

    def execute(self):
        if not path.isfile(self.par_vals.backup):
            yield Message(
                "invalid file: '{}'".format(self.par_vals.backup))
            raise StopIteration

        copy2(self.par_vals.backup, db.database)
        yield Message(
            'restored backup of database: {}'.format(self.par_vals.backup))


class OpenProject(StudExAction):
    def execute(self):
        self.other_action(OpenReport).execute()

        yield from self.get_current_project()

        as_proj = self.paths.ProjectPath.open_if_exists()


class OpenReport(StudExAction):
    def execute(self):
        try:
            self.paths.Report.open_if_exists()
        except:
            raise StudentError('No Report File')


class OpenExFiles(ExAction):
    def execute(self):
        self.paths.old_mistakes.open_if_exists()
        self.paths.exercise_pdf.open_if_exists()
        note_file = self.paths.grading_notes

        if note_file.create_if_not_exists():
            yield Message('Created {}'.format(note_file.absolute))
        note_file.open()


class GradeTask(TaskAction):
    def execute(self):
        if self.grading.status == GradingStatus.Values.OPEN:
            if self.task.always_processed:  # Todo: test this
                self.grading.processed = True
            else:
                self.grading.processed = (
                    yield ask(
                        'was the task processed at all by the student?'))

        # if g.completed:
        #     yield Message("completed task '{}'.".format(t.name))

        self.grading.save()
        # return g


class ReopenTask(StudExAction):
    def execute(self):
        ex_constraint = (Grading.ex == self.ex) | Grading.ex.is_null()
        g = yield DataPar(lambda g: g.task.name,
                          Grading.select().join(Task).where(
                              Grading.student == self.stud,
                              ex_constraint,
                              Task.hidden_question == False,
                              Grading.completed == True))
        g.completed = False
        g.save()
        yield Message(
            'Reopened Task {} for {}'.format(g.task.name, self.msg))


class EditGradingStatus(StudExAction):
    class Options(Enum):
        @EnumMember
        def failed(self, g):
            return g.make_failed()

        @EnumMember
        def uncomplete(self, g):
            return g.uncomplete()

        @EnumMember
        def processed(self, g):
            return g.make_processed()

    _parameters = [Options]

    def execute(self):
        g = yield from self.get_grading

        res = self.par_vals.Options.value(self, g)
        res = list(res)
        succeeded = sum(res)

        yield Message('Marked {} Tasks (source {}) as {} '
                      'for {}'.format(succeeded, g.task.name,
                                      self.par_vals.Options.name,
                                      self.msg))
        if not all(res):
            yield Message('{} Failed.'.format(len(res) - succeeded))


class CompleteExercise(StudExAction):
    def execute(self):
        yield Message(str(self.status))
        for g in self.status.gradings:
            if g.completed:
                continue
            yield Message(str(g))
            g.completed = True
            g.save()


class CompleteGrading(StudExAction):
    def execute(self):
        yield Message(str(self.status))
        while True:
            gradings = [g for g in self.status.gradings if not g.completed]
            if not gradings:
                yield Message('There are no gradings left to close.')
                break
            choice = yield SumPar(ListPar(gradings, 'Grading'),
                                  self.Cancel)
            if choice == self.Cancel.cancel:
                break
            if choice.complete():
                yield Message('Completed Grading {} '
                              '(and all its subtasks)'.format(choice))

        yield Message(str(self.status))


class CommentTask(StudExAction):
    def execute(self):
        grading = yield from self.get_grading
        task = grading.task
        if not grading.status:
            save_grading = task.always_processed or (yield ask(
                'Grading for task "{}" is still open. '
                'Do you want to start it now?'.format(task.name)))
            if not save_grading:
                raise StopIteration
            grading.processed = True
            grading.save()
            yield Message('grading for task "{}" ({}) started.'.format(
                task.name,
                self.stud.name))
        dgc = lambda: Comment.select().join(GradingComment).where(
            GradingComment.grading == grading)
        yield from self.comment_loop(grading.existing_comments, grading,
                                     dgc)


class CommentTasks(StudExAction):
    def execute(self):
        while True:
            yield CommentTask

            if not (yield ask('comment another task?')):
                break


class AddExistingComment(StudExAction):
    def execute(self):
        c = yield from self.pick_from_model(Comment)
        yield from self.add_comment_nocontext(c)


class ConsequentialError(ExAction):
    @property
    def warn_par(self):
        return SumPar(Exercise.select().where(Exercise.nr > self.ex.nr),
                      Action.Cancel, name='warn',
                      prompt='Pick an exercise, from which on '
                             'consequential errors will no longer be '
                             'forgiven. The student will get a '
                             'corresponding warning.')

    @property
    def minEx_par(self):
        return DataPar(Exercise,
                       Exercise.select().where(Exercise.nr < self.ex.nr),
                       prompt='looking for consequential errors starting with'
                              'Exercise..',
                       name='minEx',
                       remember=True)


class ForgiveConsequentialError(ConsequentialError, StudExAction):
    _parameters = [
        EnumPar(YesNo, 'single', 'Ask for every single comment?')]

    COMPENSATION_MSG = "Compensation for consequential errors:"
    WARNING_MSG = "Be sure to not repeat the mistakes"
    MSG_SEP = "\n* "

    def include_comment(self, c: Comment):
        if c.point_modifier == 0:
            return False
        used_in = c.used_in_Exercises(self.stud)
        if self.ex not in used_in:
            return False
        minEx = yield self.minEx_par
        if not any(minEx.nr <= n.nr < self.ex.nr for n in used_in):
            return False
        if self.par_vals.single:
            return (yield ask('forgive {}?'.format(c)))
        return True

    def execute(self):
        compensation_comments = Comment.select().where(
            Comment.message.startswith(
                self.COMPENSATION_MSG) | Comment.message.startswith(
                self.WARNING_MSG))

        compensated = [c for c in compensation_comments if
                       c.used_for(self.stud, self.ex)]

        if compensated:
            if (yield EnumPar(YesNo, 'redo', 'it seems {} was already '
                                             'compensated. Redo compensation?'
                                             ''.format(self.short))):
                for c in compensated:
                    yield self.del_comment(c)
            else:
                raise StopIteration

        conseq_comments = []
        for c in Comment:
            if (yield from self.include_comment(c)):
                conseq_comments.append(c)

        more_comments = [c for c in Comment if
                         c.used_for(self.stud, self.ex) and
                         c not in conseq_comments and
                         c.point_modifier]
        if more_comments and (yield ask('add other comments '
                                        'as consequential errors manually?')):
            for c in more_comments:
                if (yield ask('forgive {}?'.format(c))):
                    conseq_comments.append(c)

        if not len(conseq_comments):
            yield Message('No consequential errors ' + self.msg)
            raise StopIteration

        conseq_points = sum(c.point_modifier for c in conseq_comments)

        conseq_msgs = [
            "{} {}".format(StrFn.truncate(c.message), -c.point_modifier)
            for c in conseq_comments]

        msg = self.COMPENSATION_MSG + self.MSG_SEP + self.MSG_SEP.join(
            conseq_msgs)

        yield self.add_comment_str(Category.FutureHints, msg,
                                   -conseq_points)

        warn = yield self.warn_par
        if warn:
            msg = self.WARNING_MSG + ' - otherwise from {} on,' \
                                     ' the points will be deducted ' \
                                     'again.'.format(warn.Name)
            yield self.add_comment_str(Category.FutureHints, msg)


class ForgiveConsequentialExercise(ConsequentialError):
    _parameters = [p.make_remember() for p in
                   ForgiveConsequentialError._parameters] + \
                  [EnumPar(YesNo, 'redo', 'Redo compensated Students?',
                           remember=True)]

    def execute(self):
        yield self.warn_par.make_remember()
        yield self.minEx_par

        slist = list(enumerate(Student))
        for i, s in slist:
            yield {'Student': s}
            yield Message("{}. Forgiving {}..".format(i, s))
            yield ForgiveConsequentialError


class AddNewComment(Action):
    _parameters = [TypePar(str, "message",
                           "Please enter a comment for the student to justify the point modification"),
                   TypePar(float, "point_modifier",
                           "how many points should be subtracted if this comment applies?"),
                   ]

    def execute(self):
        c = Comment()
        c.message = self.par_vals['message']
        c.point_modifier = self.par_vals['point_modifier']
        c.save(force_insert=True)
        return c


class AddGradingComment(StudExAction):
    _parameters = [Grading.id, Comment.message]

    def execute(self):
        assoc = GradingComment()
        assoc.grading = self.par_vals['Grading']
        assoc.comment = self.par_vals['Comment']
        assoc.save()


class AddExComment(StudExAction):
    def execute(self):
        yield from self.comment_loop(
            lambda: self.ex.existint_others(self.stud),
            self.get_cat, lambda: self.ex.existing_comments(self.stud))

    def get_cat(self, comment):
        cats = comment.categories
        if not cats.count():
            return DataPar(Category.name, lookup=False)
        # elif cats.count() == 1:
        #    return cats.get()
        else:
            return DataPar(Category.name, cats, lookup=False)


class CleanUnusedComments(Action):
    def execute(self):
        cs = Comment.get_unused_noconst()
        if not cs:
            yield Message('No unused Comments.')
        else:
            ans = yield EnumPar(self.Proceed,
                                prompt='There are {} unused comments. What '
                                       'would you like to do?'.format(
                                    len(cs)))
            for c in cs:
                yield from self._clean(c, ans)

    def _clean(self, c, ans):
        cmsg = ' comment {}:\n{}'.format(c.id, c.message)
        if ans == self.Proceed.ViewAll:
            yield Message(c)
        if ans == self.Proceed.DeleteOneByOne:
            if (yield ask('delete {}?'.format(cmsg))):
                if not c.delete_instance():
                    yield Message('deletion failed.')
        if ans == self.Proceed.DeleteAll:
            d = c.delete_instance()
            yield Message(('deleted' if d else 'Could not delete') + cmsg)

    class Proceed(Enum):
        ViewAll = 0
        DeleteOneByOne = 1
        DeleteAll = 2


class CorrectStudentExercise(StudExAction):
    def execute(self):

        yield Message(
            'Correct {}'.format(self.msg))

        do_grade = yield from self.do_grading()

        if do_grade:
            if (yield ask('Revert and Update repository?')):
                yield UpdateRepo

            yield Message(
                "Correcting {}\n{}".format(self.msg, self.status.progress))

            yield Validate

            if (yield ask('Open Project and Report?')):
                yield OpenProject

            yield from self.open_gradings()

        yield Message("General comments on {}".format(self.ex.Name))

        yield AddExComment

        out = yield OutputGrading
        yield Message(out)

        # ex_finished_msg = 'Is the correction of {} for {} completed?'.format(
        #    self.ex.Name, self.stud.name)
        # finished = yield ask(ex_finished_msg)
        # if finished:
        #    grading.completed = finished
        #    grading.save()

        yield Message("Correction of {} {}".format(self.msg, self.status))

    def do_grading(self):
        if self.status.completed:
            grade = yield ask('All Tasks on {} are completely graded.'
                              'do you want to go regrade?'.format(
                self.msg))
            if grade:
                for g in self.status.gradings:
                    g.completed = False
                    g.save()
            else:
                return False

        return True

    def open_gradings(self):

        hidden_questions = Task.select().where(Task.ex == self.ex,
                                               Task.hidden_question)
        if hidden_questions.count():
            yield Message("Iterating standard questions ...")
            for tq in hidden_questions:
                yield from self.exec_task(tq)

        open_gradings = yield EnumPar(self.OpenGradings,
                                      'Open Corrections for all Tasks in order now?')
        manually = open_gradings == self.OpenGradings.manually
        if (open_gradings):

            yield Message("Correcting exercise-specific Tasks ...")

            # for t in self.ex.tasks:
            for t in Task.select().where(Task.ex == self.ex,
                                         ~Task.hidden_question):
                yield from self.exec_task(t, manually)

            yield Message("Correcting general Tasks ...")

            for t in Task.general_tasks():
                yield from self.exec_task(t, manually)

    def exec_task(self, t: Task, manually):
        yield {'Task': t}
        yield Message('Task "{}"'.format(t.name))

        grading = t.get_grading(self.stud, self.ex)

        if grading.completed:
            yield Message("task '{}' was complete.".format(t.name))
            raise StopIteration

        if not manually:
            if not grading.status:
                grading.processed = True
                grading.save()
                yield Message('opened grading {}'.format(grading))
            raise StopIteration

        do_comment_loop = True

        if grading.status == GradingStatus.Values.OPEN:
            if t.always_processed:  # Todo: test this
                grading.processed = True
            else:
                ans = yield EnumPar(self.CompletedOption,
                                    prompt='was the task processed at all?')
                grading.processed = bool(ans)
                do_comment_loop = ans == self.CompletedOption.yes_and_add_comment

        yield {'Grading': grading}

        if grading.is_dirty():
            grading.save()

        if do_comment_loop:
            yield from self.comment_loop(grading.existing_comments,
                                         AddGradingComment)

            # if not grading.completed:
            #    task_finished_msg = 'Is the correction of task {} completed?'.format(
            #            t.name)
            #    grading.completed = yield EnumPar(YesNo, prompt=task_finished_msg)
            #    grading.save()

            # yield Message(
            #        "{} task '{}'.".format(
            #                'completed' if grading.completed else 'postponed',
            #                t.name))

    class OpenGradings(IntEnum):
        no = 0
        yes = 1
        manually = 2

    class CompletedOption(BoolEnum):
        no = 0
        yes = 1
        yes_and_add_comment = 2


class CheckAllComments(StudExAction):
    POSTPONE_LIMIT = 1000000

    def execute(self):
        l = list(self.ex.all_comments)
        ll, progress = len(l), 1
        yield Message('Found {} Comments for {}'.format(ll, self.ex))

        for c in l:
            if c.used_for(self.stud, self.ex):
                progress += 1
                yield Message(
                    'Comment "{}" is already used.'.format(c.message))
                continue

            proceed = yield EnumPar(self.Proceed,
                                    prompt='({}/{}) "{}" - Does the comment '
                                           'apply?'.format(progress, ll,
                                                           c))

            if proceed.postpone:
                if len(l) < self.POSTPONE_LIMIT:
                    l.append(c)
                else:
                    raise ActionError('postponed too often!')
            else:
                progress += 1

            if proceed.edit:
                c.message = yield TypePar(str, lookup=False)
                c.save()
                yield Message('edited comment: {}'.format(c))
            if proceed.add:
                yield from self.add_comment_nocontext(c)

        out = yield OutputGrading
        yield Message(out)

    class Proceed(Enum):
        no = (0, 0, 0)
        yes = (1, 0, 0)
        yes_and_edit = (1, 1, 0)
        postpone = (0, 0, 1)

        def __init__(self, add, edit, postpone):
            self.add = add
            self.edit = edit
            self.postpone = postpone


class CorrectExercise(ExAction):
    def execute(self):
        yield Message('Correcting exercise {}'.format(self.ex.Name))

        yield CreateExerciseQuestions

        progress = GradingStatus.Partition(
            (s.name, GradingStatus(self.ex, s).status) for s in Student)
        yield Message(progress)
        for s in Student:
            yield {'Student': s}
            yield CorrectStudentExercise
        # p_pdf.wait()
        yield Message(progress)


class CreateExerciseQuestions(ExAction):
    def execute(self):
        ex_pdf = self.paths.exercise_pdf.open_if_exists()
        old_mistakes = self.paths.old_mistakes.open_if_exists()

        created = 0
        yield Message('Create a series of exercise standart questions:')
        while (yield ask('create another question?')):
            qstr = yield TypePar(str,
                                 prompt='Enter the question you want to '
                                        'answer when grading {} for any '
                                        'student.'.format(self.ex.Name))
            parent = yield self.taskDataPar
            try:
                Task.create(ex=self.ex, question=qstr, name=qstr,
                            parent=parent, hidden_question=True)
                # TaskQuestion.create(ex=self.ex, question=qstr)
                created += 1
            except IntegrityError as e:
                yield Message(e)

        yield Message(
            'Created {} standart questions for {}.'.format(created,
                                                           self.ex.Name))


class RenameApp(StudExAction):
    DYNAMIC_START = r'@string/'
    S_TAG = 'string'

    def execute(self):
        yield from self.get_current_project()

        StringsXML = self.paths.StringsXML
        if not StringsXML.exists:
            raise StudentError(
                'could not find {}'.format(StringsXML.absolute),
                self.stud)
        self.xml_strings = StringsXML.read_file().getElementsByTagName(
            self.S_TAG)

        Manifest = self.paths.Manifest
        app_node = Manifest.read_file().getElementsByTagName('application')[0]
        activities = app_node.getElementsByTagName('activity')
        nodes = list(filter(self._is_launcher_activity, activities))
        if nodes:
            msg = 'Found {} Launcher activities in Manifest.'
            yield Message(msg.format(len(nodes)))
        nodes.append(app_node)

        results = set(self._replace_node_attribute(n) for n in nodes)
        dynamic = [r[0] for r in results]

        if not all(dynamic):
            Manifest.save_file()
        if any(dynamic):
            StringsXML.save_file()

        for dyn, str_name, old_name in results:
            typ_name = 'String' if dyn else 'Manifest'
            typ_name = '{} {}'.format(typ_name, str_name)
            if old_name == self.app_name:
                msg = '{} was correct ({})'.format(typ_name, self.app_name)
            else:
                msg = 'replaced {} "{}" with "{}"'.format(typ_name, old_name,
                                                          self.app_name)
            yield Message(msg)

    def _is_launcher_activity(self, a):
        return any('LAUNCHER' in cat.getAttribute('android:name') for cat in
                   a.getElementsByTagName('category'))

    def _replace_node_attribute(self, node, attribute='android:label'):
        dynamic = node.getAttribute(attribute).startswith(self.DYNAMIC_START)
        if dynamic:
            str_name = node.getAttribute(attribute)[len(self.DYNAMIC_START):]

            strings = self.xml_strings
            app_name_node = next(s for s in strings if
                                 s.getAttribute('name') == str_name)
            old_name = app_name_node.firstChild.wholeText
            app_name_node.firstChild.replaceWholeText(self.app_name)

        else:
            old_name = node.getAttribute(attribute)
            node.setAttribute(attribute, self.app_name)
            str_name = attribute

        return dynamic, str_name, old_name

    @property
    def app_name(self):
        name = self.current_project.name if self.current_project else self.stud.name
        return "asp-{:02d} ex{} {}".format(self.stud.rep, self.ex.nr, name)


class Validate(StudExAction):
    # _parameters = [EnumPar(YesNo, name="delete_ignored",
    #                        prompt='delete ignored files that were erroneousely comitted?)]

    MIN_SDK_SUPPORT = 22

    Ignored_files = [
        '.idea/workspace.xml',
        '.gradle',
        'build/',
        '.idea/libraries/',
        'app/build/',
        'local.properties',
        '.DS_Store',
    ]

    IGNORE_ERROR_SPECIFIC = "{item} was committed."
    FORGIVE_COMMENT = "{item} is not mandatory for {ex} but keep it in mind for the other exercises."

    def execute(self):

        old = self.current_project
        extra = self.extra_projects
        for p in [None] + extra:
            yield from self.set_current_project(p)
            # self.current_project = p

            if p:
                do_val = yield ask(
                    'Validating extra Project {}?'.format(p.name))
                if not do_val:
                    continue
            elif extra:
                yield Message('Validating Main Project.')

            yield from self.check_present()
            if p is None:
                yield from self.check_report()
            yield from self.check_ignored()

            yield from self.check_project_settings()

        yield from self.set_current_project(old)
        # self.current_project = old

    def check_project_settings(self):
        if self.paths.GradleBuildApp.exists:
            gradleBuildTxt = open_file(self.paths.GradleBuildApp.absolute)
            m = re.search(r'minSdkVersion (\d+)', gradleBuildTxt)
            if not m:
                yield Message('check {}, cannot find minSdkVersion.'
                              ''.format(self.paths.GradleBuildApp.Name))
            else:
                minSdkVersion = int(m.group(1))
                if minSdkVersion > self.MIN_SDK_SUPPORT:
                    # TODO: decide for future, if ,-0.5
                    msg = 'Your minSdkVersion is set to {}. I am confined to ' \
                          'test your solution with the emulator, since the ' \
                          'Nexus 7, which I would prefer, is only running ' \
                          '{}.'.format(minSdkVersion, self.MIN_SDK_SUPPORT)
                    yield self.add_comment_str(Category.ProjectStructure,
                                               msg)
                else:
                    yield Message(
                        'minSdkVersion: {}'.format(minSdkVersion))

                    # minSdkVersion 22

        if self.paths.Manifest.exists:
            yield RenameApp

    def check_present(self):

        always_ask = bool(self.current_project)
        yield from self.path_input(self.paths.ProjectPath,
                                   Comment.ProjectPathMissing, ask=always_ask)
        if not self.paths.ProjectPath.exists:
            if self.current_project is None:
                self.fail_exercise()

        for path_alias in ['IdeaProjectFolder',
                           'AppDir',
                           'Manifest',
                           'ProjectIml',
                           'AppIml',
                           'PackageFolders',
                           'GradleSettings',
                           'GradleBuild',
                           'GradleBuildApp', ]:
            try:
                failure = getattr(Comment, path_alias + 'Missing')
            except:
                failure = None
            yield from self.path_input(self.paths[path_alias], failure)

            if not self.paths[path_alias].exists:
                if self.current_project is None:
                    if (yield ask('fail whole exercise?')):
                        self.fail_exercise()

    def check_report(self):
        yield from self.path_input(self.paths.Report)
        if not self.paths.Report.exists:
            # yield self.comment_project_structure(Comment.ReportReminder)
            self.status.make_failed(Task.Report)
            if (yield ask('add a reminder comment?')):
                # Task.Report.
                yield self.add_comment(Category.ProjectStructure,
                                       comment=Comment.ReportReminder)
        else:
            g = Task.Report.get_grading(self.stud, self.ex)
            g.processed = True
            g.save()

    def fail_exercise(self):
        self.status.make_failed(*self.status.root_tasks)
        raise StudentError(
            '{} - {} will be graded as "not edited" and the student will '
            'not receive any points.'.format(
                self.paths.ProjectPath.status_string, self.ex.Name),
            self.stud)

    def path_input(self, d: CustomStudenPath, failure=None, forgive=None,
                   ask=False):
        yield Message(d.status_string)
        if d.exists and not ask:
            return d
        while True:
            early_mistaktes = CustomStudenPath.select().where(
                CustomStudenPath.directory == d.directory,
                CustomStudenPath.student == d.student)
            early_par = DataPar(CustomStudenPath.custom_path,
                                early_mistaktes)
            new_path = yield SumPar(self.RepairOption, early_par,
                                    prompt="{}\n\tdoesn't exist\nEnter a new "
                                           "path, select an existing, or "
                                           "simply don't correct the path."
                                           "".format(d.absolute))

            if isinstance(new_path, CustomStudenPath):
                new_path = new_path.custom_path
            if isinstance(new_path, self.RepairOption):
                if new_path == self.RepairOption.correct_path:
                    d.custom_path = yield PathPar(d.parent.absolute,
                                                  d.Name)
                    d.fix_path()
                else:
                    if new_path == self.RepairOption.dont_correct_path_but_forgive:
                        msg = forgive or self.FORGIVE_COMMENT.format(
                            item=d.directory.name,
                            ex=self.ex.Name)
                        yield self.add_ex_comment(msg,
                                                  Category.FutureHints)
                    break

            if d.exists:
                d.save()
                break

        comment = failure if failure and not d.status else d.status.comment
        rel_original = CustomStudenPath(directory=d.directory,
                                        student=self.stud,
                                        ex=self.ex).relative_to_repo
        d_name = "{} {}".format(d.directory.name, d.directory.type.name)
        comment = comment.format(name=d_name, rel_original=rel_original,
                                 rel_custom=d.relative_to_repo)
        yield self.add_ex_comment(comment)

    def add_ex_comment(self, comment, category=Category.ProjectStructure):
        if comment is None:
            return
        if isinstance(comment, str):
            add_comment = self.add_comment_str
        elif isinstance(comment, Comment):
            add_comment = self.add_comment
        else:
            raise TypeError(
                'Expected string or Comment, got {}'.format(type(comment)))

        return add_comment(category, comment)

    def check_ignored(self):
        errors = []
        paths = []

        for p in self.Ignored_files:
            mpath = self.paths.ProjectPath + p
            # print('checking "{}" ...'.format(mpath.absolute))

            try:
                r = mpath.get_svn_info()
            except ValueError:
                continue
            else:
                if r['wcinfo_schedule'] == 'delete':
                    print('{} was committed, but is '
                          'deleted for correction'.format(p))
                else:
                    errors.append(
                        self.IGNORE_ERROR_SPECIFIC.format(item=p))
                    paths.append(mpath.absolute)

        if len(errors):
            yield self.add_comment(Category.ProjectStructure,
                                   Comment.CommittedIgnoredItems)

            del_ig = yield ask('do you want to delete (svn rm) the '
                               'ignored files that were erroneousely '
                               'committed? (recommended)')
            if del_ig:
                try:
                    for p in paths:
                        yield from self.svn_cmd('rm', p)
                except SVNError as e:
                    print(e)
                    if e.svn_errcode == SVNErrorCode.UNVERSIONED_RESOURCE:
                        force = yield ask('Want to repeat the '
                                          'deletion with "svn rm --force" ?')
                        if force:
                            for p in paths:
                                yield from self.svn_cmd('--force', 'rm', p)
                except Exception as e:
                    print("plemm plemm???")
                    print(e)

        else:
            yield Message('No ignored items were erroneously committed.')
        for msg in errors:
            yield self.add_ex_comment(msg)

    class RepairOption(BoolEnum):
        dont_correct_path = 0
        dont_correct_path_but_forgive = 1
        correct_path = 2


class ResetExerciseGrading(StudExAction):
    def execute(self):
        msg = "\n    ".join("-{}: {}".format(n, q.execute())
                            for n, q in self.queries.items())
        yield Message('Deleted:\n' + msg)

    @property
    def queries(self):
        return {
            'Exercise Comments': ExerciseComment.delete().where(
                ExerciseComment.student == self.stud,
                ExerciseComment.ex == self.ex),
            'General Task Gradings': Grading.delete().where(
                Grading.student == self.stud,
                Grading.ex == self.ex),
            'Specific Task Gradings': Grading.delete().where(
                Grading.student == self.stud,
                Grading.task << self.ex.tasks),
        }


class ResetExercisePathGrading(ResetExerciseGrading):
    @property
    def queries(self):
        r = super().queries
        r.update({'Custom Paths': CustomStudenPath.delete().where(
            CustomStudenPath.student == self.stud,
            CustomStudenPath.ex == self.ex),
        })
        return r


class ScrapeExercise(Action):
    baseurl = "http://proglang.informatik.uni-freiburg.de/teaching/androidpracticum/"

    lastDigits = re.compile(r".*\D+(\d+)(\.pdf)?$", re.I)

    def get_link(self, me):
        href = me('a').attr('href')
        return href if href else me.html()

    def scrape_table(self, table):
        table_data = []
        rows = table.find('tr').items()
        r0 = next(rows).items('th')
        keys = [c.text() for c in r0]
        for tr in rows:
            vals = [self.get_link(e) for e in
                    tr.items('td')]  # .map(get_link)
            # print(list(vals.items()))
            table_data.append(dict(zip(keys, vals)))

        return table_data

    def execute(self):

        from pyquery import PyQuery
        from urllib.request import urlretrieve
        from datetime import datetime
        base = PyQuery(url=self.baseurl)

        linktable = base.find('table')

        tdata = self.scrape_table(linktable)

        years = [int(n['Name'][:-1]) for n in tdata if 'Name' in n and re.match(r'\d{4}\/', n['Name'])]
        default_choice = datetime.now().year
        #yield Message("available {} current {}".format(years, currentyear))

        if default_choice not in years:
            default_choice -= 1

        year = None
        if default_choice in years:
            if (yield ask('Year {} ?'.format(default_choice))):
                year = default_choice
        if not year:
            year = yield ListPar(years, 'scrape_year')

        url = "{}{}/".format(self.baseurl, year)

        d = PyQuery(url=url)
        d.make_links_absolute()

        table = d('h3:contains("Exercise")').next_all('table')
        created_ex, no_ex_msg = [], "No Exercises found."

        for l in self.scrape_table(table):
            try:
                date_format = '%d.%m.%Y'
                pdf_url = l['Sheet']
                published = datetime.strptime(l['Date'], date_format)
                due_date = datetime.strptime(l['Due date'] + ' 12',
                                             date_format + ' %H')
                ex = int(self.lastDigits.match(pdf_url).group(1))
                ex, created = Exercise.create_or_get(nr=ex, due=due_date,
                                                     published=published)

                no_ex_msg = 'No new exercises found. Last: {}'.format(
                    ex.Name)

                if not created:
                    continue
                created_ex.append(ex)
                save_to = Directory.exercise_pdf.absolute
                save_to = save_to.format(**ex.format_vars)
                urlretrieve(pdf_url, save_to)
            except Exception as e:
                print('failed to process ' + str(l))
                print(e)

        if created_ex:
            print(
                'created {}'.format(', '.join(e.Name for e in created_ex)))
        else:
            print(no_ex_msg)


class ParseAction(ExAction):
    POINTS = r"\(\+?(?P<max_points>\d+) [Pp]oints\)"
    TASK = r"(?P<name>.+?)\s*({})?$".format(POINTS)
    HEADINGS = {
        'ex': r"Exercise Sheet (?P<nr>\d+) {}".format(POINTS),
        'misc': r"Miscellaneous",
        'bonus': r"Extra {}".format(POINTS),
    }

    @staticmethod
    def split_match(pattern, string, max_split=0, flags=re.M):
        matches = re.split(pattern, string, max_split, flags)[1:]
        return list(zip(matches[::2], matches[1::2]))

    def parse_task(self, raw_task):
        parsed_task = re.match(self.TASK, raw_task).groupdict()
        t = Task()
        t.ex = self.par_vals['Exercise']
        t.name = parsed_task['name']
        if parsed_task['max_points']:
            t.max_points = int(parsed_task['max_points'])
        return t

    def parse_tasks(self, raw_tasks, bonus=False):
        r = []
        for task_raw, subtasks_raw in raw_tasks:
            t = self.parse_task(task_raw)
            if bonus:
                t.max_points = 0
            t.save()
            r.append(t)

            subtasks = self.split_match(r"^-\s+(.*\S)\s*$", subtasks_raw)
            for ss, _ in subtasks:
                st = self.parse_task(ss)
                st.parent = t
                st.save()
                r.append(st)

        return r

    def parse_heading(self, raw_heading):
        for k, rexp in self.HEADINGS.items():
            m = re.match(rexp, raw_heading)
            if m:
                return k, m

    def execute(self):
        self.internal_repository.update()
        p = self.paths.grading_key.absolute
        text = open_file(p)
        sections = self.split_match(r"^(.*)\n=+$", text)
        len_sections = len(sections)
        if not len_sections:
            raise FileParsingError('contains no sections', p)
        if len_sections not in (1, 2):
            yield Message('File has {} sections'.format(len_sections))

        hd_matches = [re.match(pat, txt_pair[0]) for pat, txt_pair in
                      zip(self.HEADINGS, sections)]

        headings = {}
        for raw_heading, s in sections:
            try:
                k, m = self.parse_heading(raw_heading)
            except TypeError:
                raise FileParsingError('Heading mismatch', p)
            if k in headings:
                raise FileParsingError('Multiple "{}" headings.'.format(k), p)
            headings[k] = m.groupdict(), s

        if not 'ex' in headings:
            raise FileParsingError('Found no section for regular '
                                   'exercise tasks.', p)

        exnr = headings['ex'][0]['nr']
        msg = "File {} contains infos for Exercise nr {}".format(p, exnr)
        assert self.ex.nr == int(exnr), msg

        tasknum = self.ex.tasks.count()
        if tasknum and (yield ask('Delete ({}) tasks from '
                                  '{}?'.format(tasknum, self.ex.Name))):
            d_rows = Task.delete().where(Task.ex == self.ex).execute()
            yield Message("deleted {} Tasks".format(d_rows))

        max_points = int(headings['ex'][0]['max_points'])

        raw_tasks = self.split_match(r"^\*\*\*\s*(.*)$", headings['ex'][1])
        tasks = self.parse_tasks(raw_tasks)

        if 'bonus' in headings:
            raw_bonus = self.split_match(r"^\*\*\*\s*(.*)$",
                                         headings['bonus'][1])
            self.parse_tasks(raw_bonus, True)

        assert self.ex.max_points == max_points, "Summed up points of subtasks ({}) " \
                                                 "must be equal to total Exercise " \
                                                 "Points ({})".format(
            self.ex.max_points, max_points)

        yield Message("Parsed {} tasks.".format(self.ex.tasks.count()))


class ActionError(Exception):
    pass


class ParameterError(Exception):
    pass


class FileParsingError(ActionError):
    def __init__(self, message, file):
        super().__init__(message)
        self.file = file


class StudentError(ActionError):
    def __init__(self, message, student):
        super().__init__(message)
        self.student = student

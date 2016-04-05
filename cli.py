import rlcompleter
import readline
from abc import ABCMeta, abstractmethod
from actions import Action, Message, Level, ActionError, StudentError
from parameter import Parameter, PathPar, Delete, GetVariable, ListPar
from peewee import Field, OperationalError
from collections import MutableMapping
from types import GeneratorType
from ascii import AsciiBox, AsciiFrame, AsciiFrameDecoration
from ascii import wrap_indent
import path_completer
import traceback


class ENVIRONMENT(MutableMapping):
    def __init__(self, parent=None, *args, **kwargs):
        self.store = dict()
        self.parent = parent if isinstance(parent, type(self)) else self.store
        self.update(dict(*args, **kwargs))  # use the free update to set keys

    def __getitem__(self, key):
        try:
            return self.store[key]
        except KeyError:
            return self.parent[key]

    def __setitem__(self, key, value):
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    @property
    def flat_dict(self):
        d = {} if self.root else self.parent.flat_dict
        d.update(self.store)
        return d

    @property
    def root(self):
        return self.parent is self.store

    @property
    def depth(self):
        return 0 if self.root else self.parent.depth + 1

    def __contains__(self, item):
        return item in self.store or item in self.parent

    def __str__(self):
        return ("\t" * self.depth) + str(self.store) + (
            "\n" + str(self.parent) if self.depth else '')


def completer(lst):
    m = 'startswith' if len(lst) < 20 else '__contains__'

    def cpl(text, state):
        options = [i for i in lst if getattr(i, m)(text)]
        if state < len(options):
            return options[state]
        else:
            return None

    return cpl


def parse_and_bind():
    if 'libedit' in readline.__doc__:
        readline.parse_and_bind("bind ^I rl_complete")
        # rl_parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")


# from actions import StudExAction
class CMD_META(ABCMeta):
    def __lshift__(self, other):
        return CLI_CMD.inject(other) >> self


class CLI_CMD(metaclass=CMD_META):
    ROOT_ENV = ENVIRONMENT()

    class NoValue:
        pass

    def __init__(self):
        self.parent = None
        self._local_env = None

    def make_local_env(self):
        self._local_env = ENVIRONMENT(self.env_parent)

    @property
    def env_parent(self):
        return self.parent.environment if self.parent else self.ROOT_ENV

    @property
    def environment(self):
        if self._local_env is not None:
            return self._local_env
        else:
            return self.env_parent

    @abstractmethod
    def execute(self):
        raise NotImplementedError('To be overridden!')

    def mute(self):
        return MUTE_CMD(self)

    def inject_child(self, cmd):
        cmd = self.inject(cmd)
        cmd.parent = self
        if cmd.META.new_scope:
            cmd.make_local_env()

        return cmd

    def inject_kv(self, key, value=None):
        return self.inject_child(key if value is None else value)

    @classmethod
    def inject(cls, cmd):
        if isinstance(cmd, type):
            if issubclass(cmd, Action):
                return A_CMD(cmd)
        if isinstance(cmd, CLI_CMD):
            return cmd
        elif isinstance(cmd, str):
            return LIT_CMD(cmd)
        elif isinstance(cmd, Message):
            return MESSAGE_CMD(cmd)
        elif isinstance(cmd, dict):
            return ASSIGN_CMD(**cmd)
        elif isinstance(cmd, Delete):
            return ENVIRONMENT_DEL_CMD(*cmd.keys)
        elif isinstance(cmd, GetVariable):
            return ENVIRONMENT_CMD(cmd.name)
        elif isinstance(cmd, GeneratorType):
            return GEN_CMD(cmd)
        elif isinstance(cmd, tuple):
            return PAR_CMD(*cmd)

        try:
            cmd = Parameter.inject(cmd)
            return PAR_INP(cmd)
        except:
            pass

        raise TypeError('invalid command value: {} '
                        '(type: {})'.format(cmd, type(cmd)))

    def __add__(self, other):
        return PAR_CMD(self, other)

    def __radd__(self, other):
        return PRINT_CMD(LIT_CMD(other)) + self

    def __mul__(self, other):
        if not isinstance(other, int):
            raise ValueError('{} should be of type integer'.format(other))
        return PAR_CMD(*(other * [self]))

    def __rshift__(self, other):
        return SEQ_CMD(self, other)

    @classmethod
    def print(cls, **kwargs):
        return PRINT_CMD(cls(**kwargs))

    class META:
        new_scope = False


class F_CMD(CLI_CMD):
    def __init__(self, f, *args, **kwargs):
        super().__init__()
        assert callable(f), TypeError('{} is not callable'.format(f))
        self.f = f
        self.args = list(args)
        self.kwargs = dict(kwargs)

    def execute(self):
        args = PAR_CMD(*self.args).execute()
        kwargs = {k: self.inject_child(v).execute() for k, v in
                  self.kwargs.items()}
        r = self.call(args, kwargs)
        if isinstance(r, CLI_CMD):
            r = r.execute()
        return r

    def call(self, args, kwargs):
        return self.f(*args, **kwargs)

    def apply(self, *args, **kwargs):
        self.args += args
        self.kwargs.update(kwargs)


class M_CMD(F_CMD):
    def call(self, args, kwargs):
        return self.f(self, *args, **kwargs)


class A_CMD(CLI_CMD):
    def __init__(self, action):
        super().__init__()
        self.action = action

    def execute(self):
        for p in self.action.parameters():
            self.inject_child(p.make_remember()).execute()

        try:
            r = self.action.init_exec(**self.environment.flat_dict)
            if isinstance(r, GeneratorType):
                r = self.inject_child(r).execute()

            return r
        except OperationalError as e:
            m = Message('A Database Problem occurred: {}'
                        .format(e), level=Level.ERROR)
            self.inject(m).execute()
        except StudentError as e:
            m = Message('Student {name} ({rep}) has failed: {}'
                        .format(e, **e.student.format_vars), level=Level.ERROR)
            self.inject(m).execute()
        except ActionError as e:
            m = Message('failed to execute action {}\n\n{}'.format(
                self.action, str(e)), level=Level.ERROR)
            self.inject(m).execute()

    @staticmethod
    def nonEmptyString(s):
        r = str(s)
        if r:
            return r
        else:
            raise ValueError('empty string')

    class META:
        new_scope = True


class PROC_INP(CLI_CMD):
    def __init__(self, proc: type = int, prompt=">"):
        super().__init__()
        self.prompt = prompt
        self.proc = proc

    def execute(self):
        r = None
        while True:
            try:
                r = self.proc(input(self.prompt))
                break
            except (ValueError, TypeError) as e:
                print(e)
        return r


class PAR_CMD(CLI_CMD):
    def __init__(self, *args):
        super().__init__()
        self.commands = args

    def execute(self):
        return [self.inject_child(c).execute() for c in self.commands]

    class META:
        new_scope = True


class SEQ_CMD(CLI_CMD):
    def __init__(self, initial_cmd, *cmd_funcs):
        super().__init__()
        self.cmd_funcs = list(cmd_funcs)
        try:
            initial_cmd = self.inject_child(initial_cmd)
            initial_cmd_func = lambda: initial_cmd
        except TypeError as e:
            assert callable(initial_cmd), e
            initial_cmd_func = initial_cmd
        self.cmd_funcs.insert(0, initial_cmd_func)

    def execute(self):
        vals = []
        for c in self.cmd_funcs:
            res = c(*vals)
            vals.append(self.inject_child(res).execute())

        return vals[-1]


class GEN_CMD(CLI_CMD):
    def __init__(self, generator: GeneratorType):
        super().__init__()
        self.generator = generator

    def execute(self):
        r = []
        subcommand = NOOP_CMD()
        while True:
            send_val = self.inject_child(subcommand).execute()
            r.append(send_val)
            try:
                subcommand = self.generator.send(send_val)
            except StopIteration:
                break
        return r[1:]

    class META:
        new_scope = True


class NOOP_CMD(CLI_CMD):
    def __init__(self):
        super().__init__()

    def execute(self):
        pass


class MUTE_CMD(CLI_CMD):
    def __init__(self, cmd):
        super().__init__()
        self.cmd = cmd

    def execute(self):
        self.cmd.execute()


class CHOSE_CMD(CLI_CMD):
    def __init__(self, init_prompt="available commands:\n\t {cmds}", **kwargs):
        super().__init__()
        self.commands = kwargs
        self.prompt = init_prompt

    def execute(self):
        try:
            choice = COMP_INP(self.commands.keys(), self.prompt).execute()
            return self.inject_child(self.commands[choice]).execute()
        except KeyboardInterrupt:
            return MESSAGE_CMD('Keyboard Interrupt').execute()
        except Exception as e:
            MESSAGE_CMD(str(e)).execute()
            traceback.print_exception(type(e), e, None)
            # MESSAGE_CMD(traceback.print_stack()).execute()

    @classmethod
    def exit(cls, **kwargs):
        kwargs.update(exit=EXIT_CMD())
        return WHILE_CMD(cls(**kwargs))


class PRINT_CMD(CLI_CMD):
    def __init__(self, cmd):
        super().__init__()
        self.cmd = self.inject_child(cmd)

    def execute(self):
        print(self.cmd.execute())


class MESSAGE_CMD(CLI_CMD):
    frames = {
        Level.ERROR: AsciiFrame(2, deco=AsciiFrameDecoration.Hash),
        Level.INFO: AsciiFrame(),
        Level.LOG: AsciiFrame(0),
        Level.WARNING: AsciiFrame(deco=AsciiFrameDecoration.Dotted),
    }

    def __init__(self, msg):
        super().__init__()
        self.msg = Message.inject(msg)

    def execute(self):
        msg = "\n".join(wrap_indent(self.msg.msg))
        p_str = str(AsciiBox(msg, frame=self.frames[self.msg.level]))
        print(p_str)
        if self.msg.confirm:
            input('Press Enter To Continue...')


class WHILE_CMD(CLI_CMD):
    def __init__(self, cmd, exit=lambda x: x == EXIT_CMD):
        super().__init__()
        self.cmd = cmd
        self.exit = exit

    def execute(self):
        old = None
        while True:
            r = self.cmd.execute()
            if self.exit(r):
                break
            old = r
        return old


class EXIT_CMD(CLI_CMD):
    def execute(self):
        return EXIT_CMD


class LIT_CMD(CLI_CMD):
    def __init__(self, string):
        super().__init__()
        self.string = string

    def execute(self):
        return self.string


class COMP_INP(CLI_CMD):
    def __init__(self, commands,
                 init_prompt="enter one of the following:\n\t {cmds}",
                 prompt=">"):
        super().__init__()
        self.commands = commands
        self.str_commands = ", ".join("'{0}'".format(n) for n in self.commands)
        self.init_prompt = init_prompt
        self.prompt = prompt

    def valid(self, inp):
        for c in self.commands:
            tc = type(c)
            r = tc(inp)
            if r == c:
                return c
        txt = "invalid input: '{}'\nmust be one of \n\t{}"
        raise ValueError(txt.format(inp, self.str_commands))

    def execute(self):
        parse_and_bind()
        readline.set_completer(completer(self.commands))
        # readline.get_line_buffer()
        print(self.init_prompt.format(cmds=self.str_commands))
        return PROC_INP(self.valid, self.prompt).execute()


class PAR_INP(CLI_CMD):
    def __init__(self, parameter: Parameter,
                 prompt=">"):
        super().__init__()
        self.parameter = Parameter.inject(parameter)
        self.prompt = prompt

    def execute(self):
        if self.parameter.lookup:
            try:
                return self.environment[self.parameter.name]
            except:
                pass
        v = self.get_input()
        if self.parameter.remember:
            self.environment[self.parameter.name] = v
        return v

    def get_input(self):
        prompt_kwargs = {'cmds': ''}
        if self.parameter.options:
            options = self.parameter.options
            prompt_kwargs['cmds'] = ", ".join("'{0}'".
                                              format(n) for n in options)
            parse_and_bind()
            readline.set_completer_delims('')
            readline.set_completer(completer(options))
        elif isinstance(self.parameter, PathPar):
            parse_and_bind()
            readline.set_completer_delims(' \t\n;')
            cpl = path_completer.Completer(self.parameter.root,
                                           self.parameter.pattern)
            readline.set_completer(cpl.complete)
        try:
            print_text = self.parameter.prompt_text.format(**prompt_kwargs)
        except KeyError:
            print_text = "[FORMATTING FAILED] " + self.parameter.prompt_text
        print(print_text)
        return PROC_INP(self.parameter.check, self.prompt).execute()

    def __hash__(self):
        return self.parameter.__hash__()

    def __eq__(self, other):
        return self.parameter == other.parameter or self.parameter == other

    class ALL:
        name = "ALL"


class ASSIGN_CMD(CLI_CMD):
    def __init__(self, **mapping):
        super().__init__()
        self.mapping = mapping

    def execute(self):
        self.environment.update(self.mapping)
        return self.mapping


class ENVIRONMENT_CMD(CLI_CMD):
    def __init__(self, key=CLI_CMD.NoValue, default=CLI_CMD.NoValue):
        super().__init__()
        self.get_args = [v for v in (key, default) if v is not CLI_CMD.NoValue]

    def execute(self):
        if self.get_args:
            return self.environment.get(*self.get_args)
        return self.environment


ENV_KEY_PARAMETER = ENVIRONMENT_CMD() >> (
    lambda e: ListPar(e.keys()) if e.keys() else 'Environment empty.')

PRINT_ENV_CMD = ENVIRONMENT_CMD() >> (
    lambda d: str({k: str(v) for k, v in d.items()})) >> MESSAGE_CMD


class ENVIRONMENT_DEL_CMD(CLI_CMD):
    def __init__(self, *keys):
        super().__init__()
        self.keys = [Parameter.inject(k).name for k in keys]

    def execute(self):
        for key in set(self.keys).intersection(self.environment):
            print('deleted variable {}'.format(key))
            del self.environment[key]


            # class ASSIGN_PAR_CMD(CLI_CMD):

# def __init__(self, key, value=None):
#         super().__init__()
#         if isinstance(key, Field):
#             key = DataPar(key)
#         self.key = key
#         self.value = value
#
#     def execute(self):
#         written = self.key in self.environment
#         eval_val = self.inject_kv(self.key, self.value).execute()
#         self.environment[self.key] = eval_val
#         return written

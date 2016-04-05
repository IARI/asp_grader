import os
import re
import readline

COMMANDS = ['extra', 'extension', 'stuff', 'errors',
            'email', 'foobar', 'foo']
RE_SPACE = re.compile('.*\s+$', re.M)


class Completer(object):
    def __init__(self, cwd=".", pattern=r".*"):
        self.cwd = cwd
        self.pattern = pattern

    def _listdir(self, root="."):
        "List directory 'root' appending the path separator to subdirs."
        res = []
        for name in os.listdir(self.full_path(root)):
            if not re.match(self.pattern, name):
                continue
            path = os.path.join(root, name)
            if os.path.isdir(path):
                name += os.sep
            res.append(name)
        return res

    def full_path(self, path):
        return os.path.join(self.cwd, path)

    def _complete_path(self, path=None):
        "Perform completion of filesystem path."
        if not path:
            return self._listdir()
        full_path = self.full_path(path)
        dirname, rest = os.path.split(path)
        tmp = dirname if dirname else '.'
        res = [os.path.join(dirname, p)
               for p in self._listdir(tmp) if p.startswith(rest)]
        # more than one match, or single match which does not exist (typo)
        if len(res) > 1 or not os.path.exists(full_path):
            return res
        # resolved to a single directory, so return list of files below it
        if os.path.isdir(path):
            return [os.path.join(path, p) for p in self._listdir(path)]
        # exact file match terminates this completion
        return [path + ' ']

    def complete(self, text, state):
        "Generic readline completion entry point."
        buffer = readline.get_line_buffer()
        line = readline.get_line_buffer().split()
        ## show all commands
        # if not line:
        #    return [c + ' ' for c in COMMANDS][state]
        # account for last argument ending in a space
        # if RE_SPACE.match(buffer):
        #    line.append('')
        # resolve command to the implementation function
        # cmd = line[0].strip()
        # if cmd in COMMANDS:
        #    impl = getattr(self, 'complete_%s' % cmd)
        #    args = line[1:]
        #    if args:
        #        return (impl(args) + [None])[state]
        #    return [cmd + ' '][state]
        # results = [c + ' ' for c in COMMANDS if c.startswith(cmd)] + [None]
        results = self._complete_path(text)
        return results[state]

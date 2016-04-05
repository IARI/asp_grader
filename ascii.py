from enum import Enum
from math import ceil
import chardet
from textwrap import wrap


def open_file(path):
    with open(path, 'rb') as content_file:
        raw = content_file.read()
    result = chardet.detect(raw)
    enc = result["encoding"]
    print("{} has encoding {}".format(path, enc))
    return raw.decode(encoding=enc)



def add_article(s: str):
    return "a{} {}".format('n' if s[0].lower() in 'aeiou' else '', s)


WIDTH = 120


def wrap_indent(obj, indent=""):
    s_indent = ' ' * len(indent)
    for l in str(obj).splitlines():
        yield from wrap(l, WIDTH, initial_indent=indent,
                        subsequent_indent=s_indent)
        indent = s_indent


class AsciiAlign(Enum):
    Left = 0
    Right = 1
    Center = 0.5

    def format(self, string: str, length: int, pad_char=" "):
        pad_length = length - len(string)
        assert pad_length >= 0, ValueError('format length too short')
        pad_left = round(pad_length * self.value)
        pad_right = pad_length - pad_left
        return pad_left * pad_char + string + pad_right * pad_char


class AsciiArt(str):
    LINE_SEP = "\n"

    def __new__(cls, *args, **kwargs):
        empty = kwargs.pop('empty', False)
        obj = super().__new__(cls, *args, **kwargs)
        obj.__empty = empty
        return obj

    @property
    def line_width(self):
        return max(map(len, self.lines))

    @property
    def height(self):
        return len(self.lines)

    def block(self, alignment: AsciiAlign = AsciiAlign.Left):
        lw = self.line_width
        return self.inject(alignment.format(l, lw) for l in self.lines)

    @property
    def lines(self):
        return [] if self.__empty else self.split(self.LINE_SEP)

    def inject(self, lines):
        lines = list(lines)
        return AsciiArt(self.LINE_SEP.join(lines), empty=len(lines) == 0)

    # def wrap(self, width=120):
    #    return AsciiArt(wrap(self, width))

    def __rshift__(self, other):
        if isinstance(other, int):
            if other < 0:
                return self.inject(l[:other] for l in self.lines)
            other = other * " "
        return self.inject(l + other for l in self.lines)

    def __lshift__(self, other):
        if isinstance(other, int):
            if other < 0:
                return self.inject(l[-other:] for l in self.lines)
            other = " " * other
        other = AsciiArt(other)
        other = other * (1, ceil(self.height / other.height))

        return other + self

    def __rrshift__(self, other):
        return self << other

    def __rlshift__(self, other):
        return self >> other

    def __mul__(self, other):
        if not isinstance(other, tuple) or len(other) != 2:
            raise TypeError(
                'can only multiply AsciiArt with tuples of length 2')
        x, y = other
        l = self.inject([''] * y)
        for i in range(x):
            l += self
        return self.inject([l] * y)

    def __pow__(self, power, modulo=None):
        return self * (power, power)

    def __add__(self, other):
        return self.inject(k + l for k, l in zip(self.lines, other.lines))
        # from itertools import zip_longest
        # return self.inject((k or "") + (l or "")
        #                    for k, l in zip_longest(self.lines, other.lines))

    def __and__(self, other):
        return self.inject(self.lines + other.lines)


class AsciiFrameDecoration(Enum):
    Normal = ('=', '|', '#')
    Hash = ('#')
    NoCorner = ('=', '|')
    Dashed = ('-', '!')
    Dotted = ('.', ':')

    def __init__(self, wedge_h: str,
                 wedge_v: str = None,
                 corner: str = None):
        self.wedge_h = AsciiArt(wedge_h)
        self.wedge_v = AsciiArt(wedge_v or wedge_h)
        self.corner = AsciiArt(corner or wedge_h)


class AsciiFrame:
    def __init__(self, width=1, padding=0, margin=0,
                 deco: AsciiFrameDecoration = AsciiFrameDecoration.Normal):
        self.width_h = width
        self.width_v = width
        self._padding_h = padding
        self._padding_v = padding
        self._margin_h = margin
        self._margin_v = margin
        self._set_h_v_option('_padding', padding)
        self._set_h_v_option('_margin', margin)
        self._set_h_v_option('width', width)
        self.deco = deco

    def _set_h_v_option(self, name, value):
        if isinstance(value, tuple):
            setattr(self, name + '_h', value[0])
            setattr(self, name + '_v', value[1])
        elif not isinstance(value, int):
            raise TypeError('margin must be of type int, or tuple.'
                            ' given {}'.format(type(value)))

    @property
    def margin_v(self):
        return AsciiArt() * (1, self._margin_v)

    @property
    def margin_h(self):
        return " " * self._margin_h

    @property
    def padding_v(self):
        return AsciiArt() * (1, self._padding_v)

    @property
    def padding_h(self):
        return " " * self._padding_h

    @property
    def corner(self):
        return self.deco.corner * (self.width_h, self.width_v)

    @property
    def frameelement_vertical(self):
        return self.deco.wedge_v * (self.width_h, 1)

    @property
    def frameelement_horizontal(self):
        return self.deco.wedge_h * (1, self.width_v)

    def encase(self, string):
        astring = AsciiArt(string).block()
        h_dim = 2 * self._padding_h + astring.line_width
        h_line = self.frameelement_horizontal * (h_dim, 1)
        h_line = self.corner + h_line + self.corner
        body = self.padding_v & (
            astring << self.padding_h >> self.padding_h) & self.padding_v
        body_vframe = body << self.frameelement_vertical >> self.frameelement_vertical
        return self.margin_v & h_line & body_vframe & h_line & self.margin_v


class AsciiBox:
    def __init__(self, message: str, heading: str = "",
                 frame: AsciiFrame = AsciiFrame()):
        self.message = message
        self.heading = heading
        self.frame = frame

    def __str__(self):
        return self.frame.encase(self.message)

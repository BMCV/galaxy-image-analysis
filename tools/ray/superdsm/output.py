import numpy as np
import sys

from IPython.display import clear_output


def is_jupyter_notebook():
    """Checks whether code is being executed in a Jupyter notebook.

    :return: ``True`` if code is being executed in a Jupyter notebook and ``False`` otherwise.
    """
    try:
        if get_ipython().__class__.__name__ == 'ZMQInteractiveShell':
            return True
    except NameError: pass
    return False


def get_output(out=None):
    """Returns a suitable :py:class:`~.Output` implementation.

    If ``out`` is None or set to ``'muted'``, then a :py:class:`~.JupyterOutput` object is retruned if code is being executed in a Jupyter notebook and a :py:class:`~.ConsoleOutput` object otherwise, and the returned object is set to *muted* in the latter case. In all other cases, ``out`` itself will be returned.

    .. runblock:: pycon

       >>> import superdsm.output
       >>> out1 = superdsm.output.get_output(None)
       >>> out1.muted
       >>> type(out1)
       >>> out2 = superdsm.output.get_output(out1)
       >>> out1 is out2
       >>> out3 = superdsm.output.get_output('muted')
       >>> type(out3)
       >>> out3.muted
    """
    kwargs = dict()
    if isinstance(out, str) and out == 'muted':
        out = None
        kwargs['muted'] = True
    if out is not None:
        return out
    if is_jupyter_notebook():
        return JupyterOutput(**kwargs)
    else:
        return ConsoleOutput(**kwargs)


class Text:

    PURPLE    = '\033[95m'
    CYAN      = '\033[96m'
    DARKCYAN  = '\033[36m'
    BLUE      = '\033[94m'
    GREEN     = '\033[92m'
    YELLOW    = '\033[93m'
    RED       = '\033[91m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    END       = '\033[0m'

    @staticmethod
    def style(text, style):
        return f'{style}{text}{Text.END}'


class Output:
    """Abstract base class of :py:class:`~.JupyterOutput` and :py:class:`~.ConsoleOutput`.

    Outputs are organized hierarchically, in the sense that each output has one or none *parent* ouput. To indicate this relationship, we say that an output is *derived* from its parent. If an output is muted, all its direct and indirect derivations will be muted too. To create a muted output, pass ``muted=True`` to the constructor of the output, or its :py:meth:`~.Output.derive` method.

    :param parent: The parent output (or ``None``).
    :param muted: ``True`` if this output should be muted and ``False`` otherwise.
    :param margin: The left indentation of this derived output (in number of whitespaces, with respect to the indentation of the parent output).
    """

    def __init__(self, parent=None, muted=False, margin=0):
        self._muted = muted
        self.parent = parent
        self.margin = margin

    @property
    def muted(self):
        """``True`` if the output has been muted and ``False`` otherwise.
        """
        return self._muted or (self.parent is not None and self.parent.muted)
    
    def derive(self, muted=False, maxlen=np.inf, margin=0):
        """Derives an output.

        :param muted: ``True`` if the derived output should be muted and ``False`` otherwise.
        :param maxlen: Maximum number of lines of the derived output.
        :param margin: The left indentation of the derived output (in number of whitespaces, with respect to the indentation of this output).
        :return: The derived output.
        """
        raise NotImplementedError()
    
    def intermediate(self, line, flush=True):
        """Display an intermediate line of text.

        Intermediate output is overwritten by the next invocation of the :py:meth:`~.intermediate` and :py:meth:`~.write` methods.

        :param line: Line of text to be displayed.
        :param flush: ``True`` if the output should be displayed immediately and ``False`` otherwise.
        """
        raise NotImplementedError()
    
    def write(self, line):
        """Outputs a line of text permantently (as opposed to intermediate output).

        Previous intermiedate output is overwritten.

        :param line: Line of text to be displayed.
        """
        raise NotImplementedError()


class JupyterOutput(Output):
    """Implements the :py:class:`~.Output` class for Jupyter-based applications.

    :param parent: The parent output (or ``None``).
    :param maxlen: Maximum number of allowed lines (older lines of text will be dropped if this number is exceeded).
    :param muted: ``True`` if this output should be muted and ``False`` otherwise.
    :param margin: The left indentation of this derived output (in number of whitespaces, with respect to the indentation of the parent output).
    """

    def __init__(self, parent=None, maxlen=np.inf, muted=False, margin=0):
        super(JupyterOutput, self).__init__(parent, muted, margin)
        assert margin >= 0
        self.lines     = []
        self.current   = None
        self.maxlen    = maxlen
        self.truncated = 0
    
    def derive(self, muted=False, maxlen=np.inf, margin=0):
        child = JupyterOutput(parent=self, maxlen=maxlen, muted=muted, margin=margin)
        if self.current is not None: child.lines.append(self.current)
        return child
    
    def clear(self, flush=False):
        """Removes all intermediate output.
        """
        clear_output(not flush)
        p_list = [self]
        while p_list[-1].parent is not None:
            p_list += [p_list[-1].parent]
        for p in p_list[::-1]:
            if p.truncated > 0: print('[...] (%d)' % self.truncated)
            for line in p.lines: print(line)
        self.current = None

    def truncate(self, offset=0):
        """Truncates output so that maximum number of lines is respected.

        Older lines of text are dropped, so that

        .. math:: \\text{number of retained lines} + \\text{offset} \\leq \\text{maximum number of lines allowed}.
        """
        if len(self.lines) + offset > self.maxlen:
            self.lines = self.lines[len(self.lines) + offset - self.maxlen:]
            self.truncated += 1
    
    def intermediate(self, line, flush=True):
        if self.muted: return
        line = ' ' * self.margin + line
        self.truncate(offset=+1)
        self.clear()
        self.current = line
        print(line)
        if flush: sys.stdout.flush()
    
    def write(self, line, keep_current=False):
        if self.muted: return
        if keep_current and self.current is not None: self.lines.append(self.current)
        line = ' ' * self.margin + line
        self.lines.append(line)
        self.truncate()
        self.clear()


class ConsoleOutput(Output):
    """Implements the :py:class:`~.Output` class for terminal-based applications.
    """

    def __init__(self, muted=False, parent=None, margin=0):
        super(ConsoleOutput, self).__init__(parent, muted, margin)
        self._intermediate_line_length = 0

    def intermediate(self, line):
        if not self.muted:
            _line = ' ' * self.margin + line
            print(self._finish_line(_line), end='\r')
            self._intermediate_line_length = len(_line)
            sys.stdout.flush()

    def _finish_line(self, line):
        return line + ' ' * max((0, self._intermediate_line_length - len(line)))
    
    def write(self, line):
        if not self.muted:
            lines = line.split('\n')
            if len(lines) == 1:
                sys.stdout.write("\033[K")
                print(' ' * self.margin + line)
            else:
                for line in lines: self.write(line)
    
    def derive(self, muted=False, margin=0):
        assert margin >= 0
        return ConsoleOutput(muted, self, self.margin + margin)
        
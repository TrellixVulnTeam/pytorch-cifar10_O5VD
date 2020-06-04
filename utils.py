import collections
import os
import os.path as path
import sys
import time

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.


def progress_bar(chrono, current, total, msg=None):
    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    msg_format = '  Step: %s | Tot: %s' \
                 % (format_time(chrono.last('step_time')),
                    format_time(chrono.total('step_time')))

    if msg:
        msg_format += ' | %s' % msg

    sys.stdout.write(msg_format)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg_format) - 3):
        sys.stdout.write(' ')

    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current == total - 1:
        sys.stdout.write('\n')
    sys.stdout.write('\r')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


class Chrono:
    def __init__(self):
        self.timings = collections.OrderedDict()

    def measure(self, what):
        return Timer(lambda t: self._done(what, t))

    def _done(self, what, t):
        self.timings.setdefault(what, []).append(t)

    def remove(self, what):
        self.timings[what] = []

    def last(self, what):
        return self.timings[what][-1]

    def total(self, what):
        return sum(self.timings[what])

    def avgtime(self, what, dropfirst=False):
        timings = self.timings[what]
        if dropfirst and len(timings) > 1:
            timings = timings[1:]
        return sum(timings) / len(timings)


class Timer:
    def __init__(self, donecb):
        self.cb = donecb

    def __enter__(self):
        self.t0 = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        t = time.time() - self.t0
        self.cb(t)


class Logger:
    def __init__(self, file):
        if path.exists(file):
            self.file = open(file, 'a')
        else:
            self.file = open(file, 'w')
            self.header()

    def header(self):
        self.file.write('epoch, time, learning_rate, tr_loss, tr_acc, val_loss, val_acc\n')
        self.flush()

    def write(self, log):
        self.file.write(log)
        self.flush()

    def flush(self):
        self.file.flush()

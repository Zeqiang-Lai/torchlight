import time
import shutil
from numpy import inf


def format_num(n, fmt='{0:.3g}'):
    f = fmt.format(n).replace('+0', '+').replace('-0', '-')
    n = str(n)
    return f if len(f) < len(n) else n


def text_divider(sym, text='', ncols=None):
    if ncols is None:
        ncols = shutil.get_terminal_size()[0]
    left = ncols // 2
    right = ncols - left
    divider = sym*(left-len(text)) + text + sym*right
    return divider


class Timer:
    def __init__(self):
        self._start_time = 0

    def tic(self):
        self._start_time = time.time()

    def tok(self):
        now = time.time()
        used = int(now - self._start_time)
        second = used % 60
        used = used // 60
        minutes = used % 60
        used = used // 60
        hours = used
        return "{}:{}:{}".format(hours, minutes, second)


class MetricTracker:
    def __init__(self):
        self._data = {}
        self.reset()

    def reset(self):
        self._data = {}

    def update(self, key, value, n=1):
        if key not in self._data.keys():
            self._data[key] = {'total': 0, 'count': 0}
        self._data[key]['total'] += value * n
        self._data[key]['count'] += n

    def avg(self, key):
        return self._data[key]['total'] / self._data[key]['count']

    def result(self):
        return {k: self._data[k]['total'] / self._data[k]['count'] for k in self._data.keys()}

    def summary(self):
        items = ['{}: {:.8f}'.format(k, v) for k, v in self.result().items()]
        return ' '.join(items)


class PerformanceMonitor:
    def __init__(self, mnt_mode, early_stop_threshold=0.1):
        self.mnt_mode = mnt_mode
        self.early_stop_threshold = early_stop_threshold

        assert self.early_stop_threshold > 0, 'early_stop_threshold should be greater than 0'
        assert self.mnt_mode in ['min', 'max']

        self.reset()

    def update(self, metric):
        improved = (self.mnt_mode == 'min' and metric <= self.mnt_best) or \
                   (self.mnt_mode == 'max' and metric >= self.mnt_best)
        self.best = False
        if improved:
            self.mnt_best = metric
            self.not_improved_count = 0
            self.best = True
        else:
            self.not_improved_count += 1

    def is_best(self):
        return self.best == True

    def should_early_stop(self):
        return self.not_improved_count > self.early_stop_threshold

    def reset(self):
        self.not_improved_count = 0
        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.best = False

    def state_dict(self):
        return {'not_improved_count': self.not_improved_count, 'mnt_best': self.mnt_best, 'best': self.best}

    def load_state_dict(self, states):
        self.not_improved_count = states['not_improved_count']
        self.mnt_best = states['mnt_best']
        self.best = states['best']

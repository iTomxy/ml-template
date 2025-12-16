try:
    import __builtin__ # Python 2
except ImportError:
    import builtins as __builtin__ # Python 3
from collections.abc import Iterable
import csv
import datetime
import fnmatch, functools
import glob
import itertools
import multiprocessing as mp
import os
import re
import shutil, socket, subprocess, sys
import time, timeit


def enum_product(*args):
    """enumerate the product of several arrays with corresponding indices
    usage:
    - `for (i1, i2), (v1, v2) in enum_product(array1, array2):`
    ref:
    - https://stackoverflow.com/questions/56430745/enumerating-a-tuple-of-indices-with-itertools-product
    """
    if len(args) == 1:
        iterator = enumerate(args[0])
    else:
        iterator = zip(
            itertools.product(*(range(len(x)) for x in args)),
            itertools.product(*args)
        )

    for _index, _data in iterator:
        yield _index, _data


class prog_bar:
    """simple progress bar with timing & ETA estimation
    Will temporarily overload the built-in `print` inside and restore when finish.
    Usage:
        ```
        for epoch in prog_bar(range(100), "Epoch"):
            for iter, (x, y) in prog_bar(enumerate(data_loader), "Iter"):
                # training
        ```
    Ref:
        - https://stackoverflow.com/questions/3002085/how-to-print-out-status-bar-and-percentage
        - https://stackoverflow.com/questions/550470/overload-print-python
    """

    n_objects       = 0
    builtin_print   = None
    prev_log        = ""

    def __init__(self, iter_obj, msg=None):
        """
        Input:
            - iter_obj: range, tuple, list, numpy.ndarray
            - msg: str, some message to show
        """
        self.eta    = True  # estimate Expected Time of Arrival
        self.msg    = msg + ": [ " if msg else "[ "

        if isinstance(iter_obj, range):
            self.current, self.stop, self.step = iter_obj.start, iter_obj.stop, iter_obj.step
        elif isinstance(iter_obj, Iterable):
            self.current, self.step = 0, 1
            try:
                self.stop = len(iter_obj)
            except TypeError:
                # TypeError: object of type ? has no len()
                self.stop = '?'
                self.eta  = False # unable to estimate ETA in this case
        else:
            raise NotImplemented

        self.iter_obj = iter_obj
        prog_bar.n_objects += 1
        if 1 == prog_bar.n_objects:
            # overload the built-in `print` temporarily
            prog_bar.builtin_print = __builtin__.print
            __builtin__.print = self.print

    def __del__(self):
        prog_bar.n_objects -= 1
        # restore the built-in `print`
        if 0 == prog_bar.n_objects:
            __builtin__.print = prog_bar.builtin_print
            print('')

    def __iter__(self):
        self.update(self.log())
        estim_time, momentum = 0, 0.9

        tic = timeit.default_timer()
        for i, x in enumerate(self.iter_obj):
            t = timeit.default_timer() - tic
            estim_time = momentum * t + (1 - momentum) * estim_time if i > 1 else t
            self.current += self.step
            if self.eta:
                eta = estim_time * (self.stop - self.current + 1)
                self.update(self.log(
                    str(datetime.timedelta(seconds=estim_time)),
                    str(datetime.timedelta(seconds=eta)) if i > 0 else "N/A"
                ))
            else:
                self.update(self.log(str(datetime.timedelta(seconds=estim_time))))

            tic = timeit.default_timer()
            yield x

        self.update(self.log())

    def log(self, estim_time=None, eta=None):
        """construct progress log string & return. Template:
            <msg>: [ <current iter> / <total iter>, <time per iter>/it, <ETA> ]
        estim_time, eta: None|str, formatted time string
        """
        s = self.msg + "{} / {}".format(self.current, self.stop)
        if estim_time is not None:
            s += ", " + estim_time + "/it"
            if eta is not None:
                s += ", ETA: " + eta
        return s + " ]"

    def update(self, new_log):
        # always update the global `prev_log`
        #   cuz the newest update always comes from the innest embedded loop
        prog_bar.builtin_print('\r' + new_log + ' ' * (len(prog_bar.prev_log) - len(new_log)), end="", flush=True)
        prog_bar.prev_log = new_log

    def print(self, *args, **kwargs):
        # clean the previous progress log first
        prog_bar.builtin_print("\r{}\r".format(' ' * len(prog_bar.prev_log)), end="")
        prog_bar.builtin_print(*args, **kwargs)
        # reprint the progress log at bottum
        prog_bar.builtin_print(prog_bar.prev_log, end="", flush=True)


def dict2csv(csv_file, dict_data):
    """write a dict to `csv_file`
    - dict_data: {(str) key name: (list) value list}
    """
    with open(csv_file, "w", newline='') as f:
        writer = csv.writer(f)
        for k in dict_data:
            v_list = [k]
            v_list.extend(list(dict_data[k]))
            writer.writerow(v_list)


def natural_sort_key(s, num_pattern=re.compile('([0-9]+)'), lower=False):
    """https://stackoverflow.com/questions/4836710/is-there-a-built-in-function-for-string-natural-sort"""
    if lower:
        return [int(text) if text.isdigit() else text.lower()
                for text in num_pattern.split(s)]
    else:
        return [int(text) if text.isdigit() else text#.lower()
                for text in num_pattern.split(s)]


def free_port():
    """find an available port, useful for PyTorch DDP
    Ref: https://www.cnblogs.com/mayanan/p/15997892.html
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as tcp:
        tcp.bind(("", 0))
        _, port = tcp.getsockname()
    return port


def sort_gpu():
    """sort GPUs descendingly by free memory
    Return:
        gpu_ids: List[int], GPU IDs
        free_mem: List[int], available memory of each GPU
    """
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'])
        free_memory = [int(x) for x in output.decode('utf-8').strip().split('\n')]
        gpus = list(enumerate(free_memory))
        gpus = sorted(gpus, key=lambda t: t[1], reverse=True)
        gpu_ids, free_mem = list(zip(*gpus))
        return gpu_ids, free_mem
    except Exception as e:
        return [], []


def gpus_type():
    """detect types of each GPU based on the `nvidia-smi` command"""
    gpu_types = {}
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], encoding="utf-8")
        gpus = output.strip().split("\n")
        gpu_types = {i: gpu for i, gpu in enumerate(gpus)}
        print("GPU types:", gpu_types)
    except FileNotFoundError:
        print("`nvidia-smi` is not installed or no NVIDIA GPU found.")
    except Exception as e:
        print(f"Error: {e}")

    return gpu_types


def rm_empty_dir(root_dir):
    """remove empty directories recursively, including `root_dir`"""
    # avoid invalid path at first call
    if not os.path.isdir(root_dir):
        return
    # clean sub-folders
    for fd in os.listdir(root_dir):
        fd = os.path.join(root_dir, fd)
        if os.path.isdir(fd):
            rm_empty_dir(fd)
    # clean itself
    if len(os.listdir(root_dir)) == 0:
        os.rmdir(root_dir)


def backup_files(backup_root, src_root='.', white_list=[], black_list=[], ignore_symlink=False):
    """Back-up files (e.g. codes) by copying recursively, selecting files based on white & black list.
    Only files match one of the white patterns will be candidates, and will be ignored if
    match any black pattern. I.e. black list is prioritised over white list.

    Potential alternative: shutil.copytree

    Example (back-up codes in a Python project):
    ```python
    backup_files(
        "./logs/1st-run/backup_code",
        white_list=["*.py", "scripts/*.sh"],
        black_list=["logs"],
    )
    ```

    Input:
        backup_root: root folder to back-up file
        src_root: str = '.', path to the root folder to search
        white_list: List[str] = [], file pattern/s to back-up
        black_list: List[str] = [], file/folder pattern/s to ignore
        ignore_symlink: bool = False, ignore (i.e. don't back-up & search) symbol link to file/folder
    """
    assert os.path.isdir(src_root), src_root
    assert not os.path.isdir(backup_root), f"* Back-up folder already exists: {backup_root}"
    assert isinstance(white_list, (list, tuple)) and len(white_list) > 0

    # resolve `~` and make them absolute path to servive
    # the working directory changing later
    src_root = os.path.expanduser(src_root)
    backup_root = os.path.realpath(os.path.expanduser(backup_root))

    # Separate iterms in black_list with explicit `/` or `\` suffix and
    # let them only apply on folder filtering
    general_bl, dir_bl = [], []
    for s in black_list:
        if s.endswith('/') or s.endswith('\\'):
            dir_bl.append(s)
        else:
            general_bl.append(s)

    # rm `./` prefix, or it will cause stupid matching failure like:
    #     fnmatch.fnmatch("./utils/misc.py", "utils/*") # <- got False
    # but works for:
    #     fnmatch.fnmatch("utils/misc.py", "utils/*") # <- got True
    # Also rm '/' or '\' suffix.
    white_list = [os.path.relpath(s) for s in white_list]
    # black_list = [os.path.relpath(s) for s in black_list]
    general_bl = [os.path.relpath(s) for s in general_bl]
    dir_bl = [os.path.relpath(s) for s in dir_bl]

    def _check(_s, _list):
        """check if `_s` matches any listed pattern"""
        _s = os.path.relpath(_s)
        for _pat in _list:
            if fnmatch.fnmatch(_s, _pat):
                return True
        return False

    cwd = os.getcwd() # full path
    os.chdir(src_root)

    Q = ['.']
    while len(Q) > 0:
        fd, Q = os.path.relpath(Q[0]), Q[1:]

        is_link = os.path.islink(fd)
        if is_link and ignore_symlink:
            continue

        _dest = os.path.join(backup_root, fd)
        if os.path.isfile(fd):
            if _check(fd, white_list) and not _check(fd, general_bl):
                os.makedirs(os.path.dirname(_dest), exist_ok=True)
                if is_link:
                    os.symlink(os.path.realpath(fd), _dest)
                else:
                    shutil.copy(fd, _dest)
        elif not (
            _check(fd, general_bl) or _check(os.path.basename(fd), general_bl) or
            _check(fd, dir_bl) or _check(os.path.basename(fd), dir_bl)
        ):
            if is_link:
                os.makedirs(os.path.dirname(_dest), exist_ok=True)
                os.symlink(os.path.realpath(fd), _dest, target_is_directory=True)
            else:
                Q.extend([os.path.join(fd, x) for x in os.listdir(fd)])

    os.chdir(cwd) # return to current working dir on finish
    # rm_empty_dir(backup_root)


def backup_by_renaming(*fd_list, suffix="bak"):
    """Back-up files or folders by renaming them."""
    for fd in fd_list:
        fd = os.path.abspath(os.path.expanduser(fd))
        if not os.path.exists(fd):
            print("No such file or folder:", fd)
            continue

        # ensure unique back-up file/folder name
        multiplicity = 0
        suf = suffix
        while os.path.exists(fd + suf):
            multiplicity += 1
            suf = suffix + str(multiplicity)

        os.rename(fd, fd + suf)


def textcolor(text, color, bg=False, bright=False):
    """color text in terminal using ANSI escape codes
    Input:
        text: str to be colored
        color: can be
            - int[3]: [R, G, B] in [0, 255], e.g. (255, 0, 0) for red
            - str<#RRGGBB>: hex/html color string, e.g. "#00FF00" for green
            - str: predefined color name, e.g. "blue"
        bg: bool = False, apply to background instead of foreground
        bright: bool = False, use bright variant for named colors
    Output:
        str: ANSI colored text
    """

    # Standard color names mapping
    color_codes = {
        'black': 0, 'red': 1, 'green': 2, 'yellow': 3,
        'blue': 4, 'magenta': 5, 'cyan': 6, 'white': 7
    }

    # Determine ANSI code
    if isinstance(color, (list, tuple)) and len(color) == 3:
        # RGB mode: [r, g, b]
        r, g, b = color
        base_code = 48 if bg else 38
        return "\033[{};2;{};{};{}m{}\033[0m".format(base_code, r, g, b, text)

    elif isinstance(color, str) and color.startswith('#'):
        # Hex color: #RRGGBB
        hex_color = color.lstrip('#')
        if len(hex_color) != 6:
            raise ValueError("Hex color must be 6 characters (RRGGBB)")

        try:
            r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            base_code = 48 if bg else 38
            return "\033[{};2;{};{};{}m{}\033[0m".format(base_code, r, g, b, text)
        except ValueError:
            raise ValueError("Invalid hex color format")

    elif isinstance(color, str) and color.lower() in color_codes:
        # Named color
        color_num = color_codes[color.lower()]
        if bright:
            # Bright colors: 90-97 (fg) or 100-107 (bg)
            base_code = 100 if bg else 90
        else:
            # Standard colors: 30-37 (fg) or 40-47 (bg)
            base_code = 40 if bg else 30

        return "\033[{}m{}\033[0m".format(base_code + color_num, text)

    raise ValueError("Invalid color format: {}".format(color))


class MPExecutor:
    """General-purpose multi-processing executor that dynamically create processes to run jobs

    Built-in alternative: concurrent.futures.ProcessPoolExecutor
    """
    def __init__(self):
        self.p_list = []

    def __del__(self):
        self.join()

    def __call__(self, f, *args, **kwargs):
        """
        Args:
            f: callable: function to call
            args, kwargs: arguments to pass to `f'
        Returns:
            p: the process that runs the job
        """
        assert callable(f)
        p = mp.Process(target=f, args=args, kwargs=kwargs)
        p.start()
        self.p_list.append(p)
        # p.join() # do NOT join here
        return p

    def join(self):
        for p in self.p_list:
            if p.is_alive():
                p.join()

        self.p_list.clear()


if __name__ == "__main__":
    data = {"a": (1, 2, 3), "b": [4, 5, 6]}
    dict2csv("test.csv", data)

    X = ([1, 2], [3, 4], [5, 6], [7, 8])
    def _generator(_X):
        for _x in _X:
            yield _x
    print("before prog_bar:", id(print))
    for epoch in prog_bar("abc", "Epoch"):
        print('Epoch:', epoch, id(print))
        for i in prog_bar(range(15, 7, -2), "range"):
            print("\trange:", i, id(print))
            time.sleep(0.5)
            for x in prog_bar(X, "X" * 30):
                print("\t\tX:", x, id(print))
                time.sleep(0.2)
        for x in prog_bar(_generator(X), "iterator"):
            print("\titerator:", x, id(print))
            time.sleep(1)
    print("after prog_bar:", id(print))

    import json
    logger = get_logger("dynamic", log_file="dynamic.json", log_console=False, fmt="%(message)s")
    logger.info(json.dumps({"time": time.asctime(), "acc": 0.78}))

    backup_files("log/backup_codes", white_list=["*.py", "*.sh"], black_list=["log/"])

    print(textcolor("Hello World", [255, 0, 0]))
    print(textcolor("Hello World", "#00FF00"))
    print(textcolor("Hello World", "blue"))
    print(textcolor("Hello World", "red", bg=True))
    print(textcolor("Hello World", "green", bright=True))

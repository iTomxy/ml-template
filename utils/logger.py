import os, logging, sys


def get_logger(
    name,
    log_file='',
    # fmt='[%(asctime)s] - {%(filename)s:%(lineno)d} - %(levelname)s - %(message)s',
    fmt="{{'time': %(asctime)s, 'file': %(filename)s, 'lineno': %(lineno)d, 'level': %(levelname)s, 'msg': %(message)s}}",
    datefmt='%Y-%m-%d %H:%M:%S',
    log_level=logging.INFO,
    log_file_mode='a',
):
    """using built-in logging module
    https://blog.csdn.net/weixin_39278265/article/details/115203933
    Args:
        name: str, globally unique logger name, e.g. `__file__`
        log_file: str = '', log to file if provided
        fmt: str, logging message format
        datefmt: str, date format
        log_level: can be logging.NOTSET|DEBUG|INFO|WARNING|ERROR|CRITICAL
        log_file_mode: str = 'a', in {'a', 'w'}. If log to file, set the writing mode.
    Return:
        logger: logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    formatter = logging.Formatter(fmt, datefmt=datefmt)
    handlers = []
    # terminal output to stdout: debug, info
    h = logging.StreamHandler(sys.stdout)
    h.setLevel(logging.DEBUG)
    h.addFilter(lambda record: record.levelno <= logging.INFO)
    handlers.append(h)
    # terminal output to stderr: warning, error, critical
    h = logging.StreamHandler(sys.stderr)
    h.setLevel(logging.WARNING)
    handlers.append(h)
    # file output
    if log_file:
        os.makedirs(os.path.dirname(log_file) or '.', exist_ok=True)
        h = logging.FileHandler(log_file, mode=log_file_mode)
        h.setLevel(log_level)
        handlers.append(h)

    for h in handlers:
        # h.setLevel(log_level)
        h.setFormatter(formatter)
        logger.addHandler(h)

    return logger

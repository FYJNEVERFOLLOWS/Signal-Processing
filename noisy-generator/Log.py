import logging, os

LOG_FORMAT = "%(message)s"

class Log():
    def __init__(self, clean = False):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(LOG_FORMAT)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        self.logger.addHandler(ch)

    def log(self, *args):
        s = ''
        for i in args:
            s += (str(i) + ' ')

        logging.debug(s)

log = Log(True)

def init_logger(log_file, log_level=logging.INFO):
    log_format = logging.Formatter("[\033[032m%(asctime)s\033[0m %(levelname)s] %(module)s.%(funcName)s %(message)s")
    logger = logging.getLogger()
    logger.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.NOTSET)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger
# -*- coding:utf-8 -*-

import logging
from logging import handlers, FileHandler


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info',
                 fmt='%(asctime)s - %(levelname)s: %(message)s'):
        self.filename = filename
        self.level = level
        self.logger = logging.getLogger(filename)
        self.format_str = logging.Formatter(fmt)  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(self.level))  # 设置日志级别
        self.log2file()
        self.log2screen()

    def log2file(self):
        sh = logging.StreamHandler()  # 往屏幕上输出
        sh.setFormatter(self.format_str)  # 设置屏幕上显示的格式
        self.logger.addHandler(sh)  # 把对象加到logger里

    def log2screen(self):
        th = logging.FileHandler(filename=self.filename, encoding='utf-8')  # 往文件里写入#指定间隔时间自动生成文件的处理器
        th.setFormatter(self.format_str)  # 设置文件里写入的格式
        self.logger.addHandler(th)  # 把对象加到logger里


if __name__ == '__main__':
    log = Logger('./all.log', level='info')
    log.logger.info('this is a test')

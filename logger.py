import logging
from os.path import dirname, abspath
import os
import colorlog 
import time

class TrainLogger():
    def __init__(
        self,
        logger_name: str,
        std_level=logging.INFO,
        file_level=logging.DEBUG,
        std_out: bool=True,
        save_info: bool=True,
        output_dir: str | None=None,
        file_name: str | None=None,
    ) -> None:
        """训练日志类

        参数:
        - logger_name (str): 日志名称, 用于区分不同模块
        - std_level (logging level): 控制台输出日志级别, 默认为 INFO
        - file_level (logging level): 文件输出日志级别, 默认为 DEBUG
        - std_out (bool): 是否输出到控制台, 默认为 True
        - save_info (bool): 是否保存日志到文件, 默认为 True
        - output_dir (str | None): 输出目录, 默认为 None, 此时使用默认日志目录
        - file_name (str | None): 日志文件名, 默认为 None, 此时使用日期记录
        """
        datefmt = "%Y-%m-%d %H:%M:%S"
        # 日期格式化, 年-月-日 时:分:秒

        if std_out:   # 输出到控制台
            std_logfmt = "[%(asctime)s.%(msecs)03d] [%(levelname)s]: %(log_color)s%(message)s"
            # 构建标准格式

            self.stdout_logger = logging.getLogger('{}_std'.format(logger_name))
            self.stdout_logger.setLevel(std_level)
            # 创建 logger 实例

            log_colors_config = {
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red'
            }   # 日志颜色配置

            formatter = colorlog.ColoredFormatter(
                fmt=std_logfmt,
                datefmt=datefmt,
                log_colors=log_colors_config,
            )   # 彩色日志格式标准化

            sh = logging.StreamHandler()
            sh.setLevel(std_level)
            sh.setFormatter(formatter)
            self.stdout_logger.addHandler(sh)
            # 绑定 formatter, 按彩色格式输出

        if save_info:   # 保存到文件
            file_logfmt = "[%(asctime)s.%(msecs)03d] [%(levelname)s]: %(message)s"
            # 去掉颜色字段

            self.file_logger = logging.getLogger('{}_file'.format(logger_name))
            self.file_logger.setLevel(file_level)
            # 创建文件专用 logger, 设置日志级别为 file_level

            if output_dir is not None:   # 指定项目根目录
                base_dir = os.path.join(output_dir, 'logs')   # 指定目录
            else:
                base_dir = os.path.join(dirname(dirname(abspath(__file__))), 'logs')   # 获取上级目录的绝对路径

            if not os.path.exists(base_dir):   # 检查目录是否存在, 不存在则创建
                os.makedirs(base_dir, exist_ok=True)

            if file_name is not None:   # 确定日志文件名
                log_file = file_name
            else:                       # 未指定文件名, 则使用日期记录
                log_file = os.path.join(base_dir, f"{logger_name}-{time.strftime('%Y%m%d')}.log")

            fh = logging.FileHandler(filename=log_file, mode='a', encoding='utf-8')
            fh.setLevel(file_level)
            # 创建文件处理器

            save_formatter =  logging.Formatter(
                fmt=file_logfmt,
                datefmt=datefmt,
                )
            fh.setFormatter(save_formatter)
            self.file_logger.addHandler(fh)
            # 绑定格式器并添加到 logger

    def info(self, message: str, std_out: bool=True, save_to_file: bool=True) -> None:
        """输出 INFO 日志
        参数:
        - message (str): 日志消息
        - std_out (bool): 是否输出到控制台, 默认为 True
        - save_to_file (bool): 是否保存到文件, 默认为 True
        """
        if std_out:
            self.stdout_logger.info(message)
        if save_to_file:
            self.file_logger.info(message)

    def debug(self, message: str, std_out: bool=True, save_to_file: bool=True) -> None:
        """输出 DEBUG 日志
        参数:
        - message (str): 日志消息
        - std_out (bool): 是否输出到控制台, 默认为 True
        - save_to_file (bool): 是否保存到文件, 默认为 True
        """
        if std_out:
            self.stdout_logger.debug(message)
        if save_to_file:
            self.file_logger.debug(message)

    def warning(self, message: str, std_out: bool=True, save_to_file: bool=True) -> None:
        """输出 WARNING 日志
        参数:
        - message (str): 日志消息
        - std_out (bool): 是否输出到控制台, 默认为 True
        - save_to_file (bool): 是否保存到文件, 默认为 True
        """
        if std_out:
            self.stdout_logger.warning(message)
        if save_to_file:
            self.file_logger.warning(message)

    def error(self, message: str, std_out: bool=True, save_to_file: bool=True) -> None:
        """输出 ERROR 日志
        参数:
        - message (str): 日志消息
        - std_out (bool): 是否输出到控制台, 默认为 True
        - save_to_file (bool): 是否保存到文件, 默认为 True
        """
        if std_out:
            self.stdout_logger.error(message)
        if save_to_file:
            self.file_logger.error(message)

if __name__ == "__main__":
    logger = TrainLogger(
        logger_name="TEST",
        std_level=logging.DEBUG,
        file_level=logging.DEBUG,
        std_out=True,
        save_info=True,
        output_dir=None,
        file_name=None,
    )

    logger.info("This is an info message.", std_out=True, save_to_file=True)
    logger.debug("This is a debug message.", std_out=True, save_to_file=True)
    logger.warning("This is a warning message.", std_out=True, save_to_file=True)
    logger.error("This is an error message.", std_out=True, save_to_file=True)
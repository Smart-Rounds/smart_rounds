from core.settings import settings
import logging

class BaseService:
    def __init__(self):
        self.settings = settings
        logging.basicConfig(level=getattr(logging, settings.log_level))
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__}")


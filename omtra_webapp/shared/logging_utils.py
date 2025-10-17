import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
import os


def setup_logging(
    level: str = "INFO",
    structured: bool = True,
    log_file: Optional[Path] = None
) -> logging.Logger:
    """Setup structured logging"""
    
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    if structured:
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage']:
                log_entry[key] = value
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


def log_job_event(
    logger: logging.Logger,
    job_id: str,
    event: str,
    **kwargs
):
    """Log a job-related event with structured data"""
    logger.info(
        f"Job event: {event}",
        extra={
            'job_id': job_id,
            'event': event,
            **kwargs
        }
    )


def log_api_request(
    logger: logging.Logger,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    **kwargs
):
    """Log an API request with structured data"""
    logger.info(
        f"API request: {method} {path} - {status_code}",
        extra={
            'method': method,
            'path': path,
            'status_code': status_code,
            'duration_ms': duration_ms,
            **kwargs
        }
    )


def log_file_upload(
    logger: logging.Logger,
    filename: str,
    size: int,
    sha256: str,
    client_ip: Optional[str] = None,
    **kwargs
):
    """Log a file upload event"""
    logger.info(
        f"File uploaded: {filename}",
        extra={
            'event': 'file_upload',
            'file_name': filename,  # Use 'file_name' instead of 'filename' to avoid conflict
            'size': size,
            'sha256': sha256,
            'client_ip': client_ip,
            **kwargs
        }
    )


# Initialize logging based on environment
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
STRUCTURED_LOGGING = os.getenv('STRUCTURED_LOGGING', 'true').lower() == 'true'

# Create global logger
logger = setup_logging(
    level=LOG_LEVEL,
    structured=STRUCTURED_LOGGING
)

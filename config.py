"""
Configuration module for Ollama Mirror Server

Handles configuration loading from environment variables, config files, and defaults.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pydantic import validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


@dataclass
class CacheSettings:
    """Cache-related configuration"""
    directory: Path
    retention_days: int = 30
    cleanup_interval: int = 3600  # seconds
    max_size_gb: Optional[float] = None
    compress_blobs: bool = False


@dataclass
class ServerSettings:
    """Server-related configuration"""
    host: str = "0.0.0.0"
    port: int = 11434
    workers: int = 1
    reload: bool = False
    access_log: bool = True


@dataclass
class UpstreamSettings:
    """Upstream registry configuration"""
    url: str = "https://registry.ollama.ai"
    timeout: int = 300
    max_concurrent_downloads: int = 3
    retry_attempts: int = 3
    retry_delay: float = 1.0


@dataclass
class SecuritySettings:
    """Security-related configuration"""
    api_key: Optional[str] = None
    allowed_models: Optional[list] = None
    blocked_models: Optional[list] = None
    rate_limit_requests: int = 100
    rate_limit_window: int = 60


@dataclass
class LoggingSettings:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[Path] = None
    max_file_size: str = "10MB"
    backup_count: int = 5


class OllamaMirrorConfig(BaseSettings):
    """Main configuration class using Pydantic for validation"""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 11434
    workers: int = 1
    reload: bool = False

    # Cache settings
    cache_dir: Path = Path("./cache")
    cache_retention_days: int = 30
    cache_cleanup_interval: int = 3600
    cache_max_size_gb: Optional[float] = None
    cache_compress_blobs: bool = False

    # Upstream settings
    upstream_url: str = "https://registry.ollama.ai"
    upstream_timeout: int = 300
    max_concurrent_downloads: int = 3
    retry_attempts: int = 3
    retry_delay: float = 1.0

    # Security settings
    api_key: Optional[str] = None
    allowed_models: Optional[str] = None  # Comma-separated list
    blocked_models: Optional[str] = None  # Comma-separated list
    rate_limit_requests: int = 100
    rate_limit_window: int = 60

    # Logging settings
    log_level: str = "INFO"
    log_file: Optional[Path] = None
    log_max_file_size: str = "10MB"
    log_backup_count: int = 5

    # Feature flags
    enable_api_proxy: bool = True
    enable_metrics: bool = True
    enable_web_ui: bool = False

    class Config:
        env_prefix = "OLLAMA_MIRROR_"
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"

    @validator('cache_dir')
    def validate_cache_dir(cls, v):
        """Ensure cache directory is a Path object"""
        if isinstance(v, str):
            return Path(v)
        return v

    @validator('log_file')
    def validate_log_file(cls, v):
        """Ensure log file is a Path object if provided"""
        if v is not None and isinstance(v, str):
            return Path(v)
        return v

    @validator('upstream_url')
    def validate_upstream_url(cls, v):
        """Ensure upstream URL doesn't end with slash"""
        return v.rstrip('/')

    @validator('log_level')
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v

    def get_allowed_models(self) -> Optional[list]:
        """Parse allowed models from comma-separated string"""
        if self.allowed_models:
            return [model.strip() for model in self.allowed_models.split(',')]
        return None

    def get_blocked_models(self) -> Optional[list]:
        """Parse blocked models from comma-separated string"""
        if self.blocked_models:
            return [model.strip() for model in self.blocked_models.split(',')]
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "server": {
                "host": self.host,
                "port": self.port,
                "workers": self.workers,
                "reload": self.reload
            },
            "cache": {
                "directory": str(self.cache_dir),
                "retention_days": self.cache_retention_days,
                "cleanup_interval": self.cache_cleanup_interval,
                "max_size_gb": self.cache_max_size_gb,
                "compress_blobs": self.cache_compress_blobs
            },
            "upstream": {
                "url": self.upstream_url,
                "timeout": self.upstream_timeout,
                "max_concurrent_downloads": self.max_concurrent_downloads,
                "retry_attempts": self.retry_attempts,
                "retry_delay": self.retry_delay
            },
            "security": {
                "api_key": "***" if self.api_key else None,
                "allowed_models": self.get_allowed_models(),
                "blocked_models": self.get_blocked_models(),
                "rate_limit_requests": self.rate_limit_requests,
                "rate_limit_window": self.rate_limit_window
            },
            "logging": {
                "level": self.log_level,
                "file": str(self.log_file) if self.log_file else None,
                "max_file_size": self.log_max_file_size,
                "backup_count": self.log_backup_count
            },
            "features": {
                "enable_api_proxy": self.enable_api_proxy,
                "enable_metrics": self.enable_metrics,
                "enable_web_ui": self.enable_web_ui
            }
        }


def load_config(config_file: Optional[Path] = None) -> OllamaMirrorConfig:
    """Load configuration from environment variables and optional config file"""

    # Start with default config
    config = OllamaMirrorConfig()

    # Load from config file if provided
    if config_file and config_file.exists():
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)

            # Update config with file values
            for section, values in file_config.items():
                for key, value in values.items():
                    env_key = f"OLLAMA_MIRROR_{section.upper()}_{key.upper()}"
                    if env_key not in os.environ:
                        os.environ[env_key] = str(value)

            # Recreate config to pick up the new environment variables
            config = OllamaMirrorConfig()
            logger.info(f"Loaded configuration from {config_file}")

        except Exception as e:
            logger.warning(f"Failed to load config file {config_file}: {e}")

    return config


def save_config(config: OllamaMirrorConfig, config_file: Path):
    """Save current configuration to file"""
    try:
        config_dict = config.to_dict()

        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

        logger.info(f"Configuration saved to {config_file}")

    except Exception as e:
        logger.error(f"Failed to save configuration to {config_file}: {e}")
        raise


def setup_logging(config: OllamaMirrorConfig):
    """Setup logging based on configuration"""
    import logging.handlers

    # Create logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.log_level))

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Setup formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if config.log_file:
        try:
            config.log_file.parent.mkdir(parents=True, exist_ok=True)

            # Parse max file size
            size_mb = 10  # default
            if config.log_max_file_size.endswith('MB'):
                size_mb = int(config.log_max_file_size[:-2])

            file_handler = logging.handlers.RotatingFileHandler(
                config.log_file,
                maxBytes=size_mb * 1024 * 1024,
                backupCount=config.log_backup_count
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

            logger.info(f"Logging to file: {config.log_file}")

        except Exception as e:
            logger.warning(f"Failed to setup file logging: {e}")


def validate_config(config: OllamaMirrorConfig) -> list:
    """Validate configuration and return list of issues"""
    issues = []

    # Check cache directory
    try:
        config.cache_dir.mkdir(parents=True, exist_ok=True)
        if not config.cache_dir.is_dir():
            issues.append(
                f"Cache directory is not accessible: {config.cache_dir}")
    except Exception as e:
        issues.append(f"Cannot create cache directory {config.cache_dir}: {e}")

    # Check port availability
    if not (1 <= config.port <= 65535):
        issues.append(f"Invalid port number: {config.port}")

    # Check upstream URL
    if not config.upstream_url.startswith(('http://', 'https://')):
        issues.append(f"Invalid upstream URL: {config.upstream_url}")

    # Check log file directory if specified
    if config.log_file:
        try:
            config.log_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Cannot create log directory: {e}")

    # Check cache size limit
    if config.cache_max_size_gb is not None and config.cache_max_size_gb <= 0:
        issues.append("Cache max size must be positive")

    return issues


def get_sample_config() -> Dict[str, Any]:
    """Get a sample configuration dictionary"""
    return {
        "server": {
            "host": "0.0.0.0",
            "port": 11434,
            "workers": 1,
            "reload": False
        },
        "cache": {
            "directory": "./cache",
            "retention_days": 30,
            "cleanup_interval": 3600,
            "max_size_gb": None,
            "compress_blobs": False
        },
        "upstream": {
            "url": "https://registry.ollama.ai",
            "timeout": 300,
            "max_concurrent_downloads": 3,
            "retry_attempts": 3,
            "retry_delay": 1.0
        },
        "security": {
            "api_key": None,
            "allowed_models": None,
            "blocked_models": None,
            "rate_limit_requests": 100,
            "rate_limit_window": 60
        },
        "logging": {
            "level": "INFO",
            "file": None,
            "max_file_size": "10MB",
            "backup_count": 5
        },
        "features": {
            "enable_api_proxy": True,
            "enable_metrics": True,
            "enable_web_ui": False
        }
    }


# Global config instance (will be initialized in main module)
config: Optional[OllamaMirrorConfig] = None


def get_config() -> OllamaMirrorConfig:
    """Get the global configuration instance"""
    global config
    if config is None:
        config = load_config()
    return config

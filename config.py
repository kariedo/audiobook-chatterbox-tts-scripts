#!/usr/bin/env python3
"""
Configuration Management System for Chatterbox TTS
Supports TOML files, environment variables, and CLI arguments with proper precedence
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field


try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # Fallback for older Python versions
    except ImportError:
        tomllib = None


@dataclass
class TTSConfig:
    """TTS-specific configuration"""
    voice_file: Optional[str] = None
    exaggeration: float = 0.8
    cfg_weight: float = 0.8
    pitch_shift: float = 0.0
    max_workers: int = 2
    memory_cleanup_interval: int = 5
    debug_memory: bool = False


@dataclass
class AudioConfig:
    """Audio processing configuration"""
    mp3_enabled: bool = False
    mp3_bitrate: str = "128k"
    remove_wav: bool = False
    split_minutes: int = 5
    smart_split: bool = True
    silence_threshold: float = -35.0
    min_silence_duration: float = 0.3


@dataclass
class ProcessingConfig:
    """Text processing configuration"""
    max_chunk_chars: int = 200
    min_chunk_chars: int = 50
    limit_minutes: Optional[int] = None
    validate_audio: bool = True
    regeneration_attempts: int = 4
    ending_detection: bool = True


@dataclass
class OutputConfig:
    """Output and metadata configuration"""
    output_dir_template: str = "{filename}"
    metadata_artist: Optional[str] = None
    metadata_album: Optional[str] = None
    metadata_genre: str = "Audiobook"
    preserve_structure: bool = True


@dataclass
class ChatterboxConfig:
    """Complete Chatterbox configuration"""
    tts: TTSConfig = field(default_factory=TTSConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


class ConfigError(Exception):
    """Configuration-related errors"""
    pass


class ConfigManager:
    """Manages configuration from multiple sources with proper precedence"""
    
    def __init__(self, config_file: Optional[Path] = None):
        self.config_file = config_file
        self.config = ChatterboxConfig()
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file, environment, respecting precedence"""
        
        # 1. Start with defaults (already set in dataclasses)
        
        # 2. Load from config file if specified or found
        config_data = {}
        if self.config_file and self.config_file.exists():
            config_data = self._load_toml_file(self.config_file)
        elif not self.config_file:
            # Look for default config files
            default_locations = [
                Path.cwd() / "chatterbox_config.toml",
                Path.cwd() / "config.toml",
                Path.home() / ".config" / "chatterbox" / "config.toml",
            ]
            
            for location in default_locations:
                if location.exists():
                    self.config_file = location
                    config_data = self._load_toml_file(location)
                    logging.info(f"📁 Using config file: {location}")
                    break
        
        # 3. Apply config file values
        if config_data:
            self._apply_config_data(config_data)
        
        # 4. Apply environment variables (highest precedence before CLI)
        self._apply_environment_variables()
    
    def _load_toml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load TOML configuration file"""
        if tomllib is None:
            raise ConfigError(
                "TOML support not available. Install with: pip install tomli"
            )
        
        try:
            with open(file_path, 'rb') as f:
                return tomllib.load(f)
        except Exception as e:
            raise ConfigError(f"Failed to load config file {file_path}: {e}")
    
    def _apply_config_data(self, data: Dict[str, Any]):
        """Apply configuration data from file"""
        
        # Apply TTS settings
        if 'tts' in data:
            tts_data = data['tts']
            for key, value in tts_data.items():
                if hasattr(self.config.tts, key):
                    # Validate value
                    validated_value = self._validate_config_value('tts', key, value)
                    setattr(self.config.tts, key, validated_value)
                else:
                    logging.warning(f"Unknown TTS config key: {key}")
        
        # Apply Audio settings
        if 'audio' in data:
            audio_data = data['audio']
            for key, value in audio_data.items():
                if hasattr(self.config.audio, key):
                    validated_value = self._validate_config_value('audio', key, value)
                    setattr(self.config.audio, key, validated_value)
                else:
                    logging.warning(f"Unknown audio config key: {key}")
        
        # Apply Processing settings
        if 'processing' in data:
            processing_data = data['processing']
            for key, value in processing_data.items():
                if hasattr(self.config.processing, key):
                    validated_value = self._validate_config_value('processing', key, value)
                    setattr(self.config.processing, key, validated_value)
                else:
                    logging.warning(f"Unknown processing config key: {key}")
        
        # Apply Output settings
        if 'output' in data:
            output_data = data['output']
            for key, value in output_data.items():
                if hasattr(self.config.output, key):
                    validated_value = self._validate_config_value('output', key, value)
                    setattr(self.config.output, key, validated_value)
                else:
                    logging.warning(f"Unknown output config key: {key}")
    
    def _apply_environment_variables(self):
        """Apply configuration from environment variables"""
        env_mappings = {
            # TTS settings
            'CHATTERBOX_VOICE': ('tts', 'voice_file'),
            'CHATTERBOX_EXAGGERATION': ('tts', 'exaggeration', float),
            'CHATTERBOX_CFG_WEIGHT': ('tts', 'cfg_weight', float),
            'CHATTERBOX_PITCH_SHIFT': ('tts', 'pitch_shift', float),
            'CHATTERBOX_MAX_WORKERS': ('tts', 'max_workers', int),
            'CHATTERBOX_DEBUG_MEMORY': ('tts', 'debug_memory', bool),
            
            # Audio settings
            'CHATTERBOX_MP3_ENABLED': ('audio', 'mp3_enabled', bool),
            'CHATTERBOX_MP3_BITRATE': ('audio', 'mp3_bitrate'),
            'CHATTERBOX_REMOVE_WAV': ('audio', 'remove_wav', bool),
            'CHATTERBOX_SPLIT_MINUTES': ('audio', 'split_minutes', int),
            'CHATTERBOX_SMART_SPLIT': ('audio', 'smart_split', bool),
            
            # Processing settings
            'CHATTERBOX_MAX_CHUNK_CHARS': ('processing', 'max_chunk_chars', int),
            'CHATTERBOX_LIMIT_MINUTES': ('processing', 'limit_minutes', int),
            'CHATTERBOX_VALIDATE_AUDIO': ('processing', 'validate_audio', bool),
            'CHATTERBOX_REGENERATION_ATTEMPTS': ('processing', 'regeneration_attempts', int),
            
            # Output settings
            'CHATTERBOX_METADATA_ARTIST': ('output', 'metadata_artist'),
            'CHATTERBOX_METADATA_ALBUM': ('output', 'metadata_album'),
            'CHATTERBOX_METADATA_GENRE': ('output', 'metadata_genre'),
        }
        
        for env_var, mapping in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                section = mapping[0]
                key = mapping[1]
                type_converter = mapping[2] if len(mapping) > 2 else str
                
                try:
                    # Convert environment variable value
                    if type_converter == bool:
                        converted_value = env_value.lower() in ('true', '1', 'yes', 'on')
                    elif type_converter in (int, float):
                        converted_value = type_converter(env_value)
                    else:
                        converted_value = env_value
                    
                    # Apply to config
                    config_section = getattr(self.config, section)
                    setattr(config_section, key, converted_value)
                    logging.debug(f"Applied env var {env_var}={env_value} to {section}.{key}")
                    
                except (ValueError, TypeError) as e:
                    logging.warning(f"Invalid environment variable {env_var}={env_value}: {e}")
    
    def _validate_config_value(self, section: str, key: str, value: Any) -> Any:
        """Validate configuration values"""
        
        # TTS validations
        if section == 'tts':
            if key == 'exaggeration' and not (0.1 <= value <= 2.0):
                raise ConfigError(f"TTS exaggeration must be between 0.1 and 2.0, got {value}")
            elif key == 'cfg_weight' and not (0.1 <= value <= 1.0):
                raise ConfigError(f"TTS cfg_weight must be between 0.1 and 1.0, got {value}")
            elif key == 'pitch_shift' and not (-12.0 <= value <= 12.0):
                raise ConfigError(f"TTS pitch_shift must be between -12.0 and 12.0 semitones, got {value}")
            elif key == 'max_workers' and not (1 <= value <= 16):
                raise ConfigError(f"TTS max_workers must be between 1 and 16, got {value}")
        
        # Audio validations
        elif section == 'audio':
            if key == 'mp3_bitrate' and value not in ['64k', '96k', '128k', '160k', '192k', '256k', '320k']:
                raise ConfigError(f"Audio mp3_bitrate must be one of 64k-320k, got {value}")
            elif key == 'split_minutes' and not (1 <= value <= 60):
                raise ConfigError(f"Audio split_minutes must be between 1 and 60, got {value}")
        
        # Processing validations
        elif section == 'processing':
            if key == 'max_chunk_chars' and not (50 <= value <= 1000):
                raise ConfigError(f"Processing max_chunk_chars must be between 50 and 1000, got {value}")
            elif key == 'min_chunk_chars' and not (10 <= value <= 200):
                raise ConfigError(f"Processing min_chunk_chars must be between 10 and 200, got {value}")
            elif key == 'regeneration_attempts' and not (1 <= value <= 10):
                raise ConfigError(f"Processing regeneration_attempts must be between 1 and 10, got {value}")
        
        return value
    
    def apply_cli_args(self, args):
        """Apply command-line arguments (highest precedence)"""
        
        # Map CLI args to config sections
        cli_mappings = {
            'voice': ('tts', 'voice_file'),
            'workers': ('tts', 'max_workers'),
            'exaggeration': ('tts', 'exaggeration'),
            'cfg_weight': ('tts', 'cfg_weight'),
            'pitch_shift': ('tts', 'pitch_shift'),
            'debug_memory': ('tts', 'debug_memory'),
            
            'mp3': ('audio', 'mp3_enabled'),
            'mp3_bitrate': ('audio', 'mp3_bitrate'),
            'remove_wav': ('audio', 'remove_wav'),
            'split_minutes': ('audio', 'split_minutes'),
            'smart_split': ('audio', 'smart_split'),
            
            'limit_minutes': ('processing', 'limit_minutes'),
            'max_chunk_chars': ('processing', 'max_chunk_chars'),
            'validate_audio': ('processing', 'validate_audio'),
            'regeneration_attempts': ('processing', 'regeneration_attempts'),
            
            'tag': ('output', 'metadata_album'),  # CLI --tag maps to album
        }
        
        for cli_arg, (section, key) in cli_mappings.items():
            if hasattr(args, cli_arg):
                value = getattr(args, cli_arg)
                
                # For boolean arguments, only apply if True (meaning it was specified)
                # For other arguments, only apply if not None (meaning it was specified)
                should_apply = False
                if isinstance(value, bool):
                    # Only apply True boolean values (when flag was specified)
                    should_apply = value
                elif value is not None:
                    # Apply non-None values for other types
                    should_apply = True
                
                if should_apply:
                    # Validate the value
                    validated_value = self._validate_config_value(section, key, value)
                    
                    # Apply to config
                    config_section = getattr(self.config, section)
                    setattr(config_section, key, validated_value)
                    logging.debug(f"Applied CLI arg --{cli_arg}={value} to {section}.{key}")
    
    def get_effective_config(self) -> Dict[str, Any]:
        """Get the current effective configuration as a dictionary"""
        return {
            'tts': {
                'voice_file': self.config.tts.voice_file,
                'exaggeration': self.config.tts.exaggeration,
                'cfg_weight': self.config.tts.cfg_weight,
                'pitch_shift': self.config.tts.pitch_shift,
                'max_workers': self.config.tts.max_workers,
                'memory_cleanup_interval': self.config.tts.memory_cleanup_interval,
                'debug_memory': self.config.tts.debug_memory,
            },
            'audio': {
                'mp3_enabled': self.config.audio.mp3_enabled,
                'mp3_bitrate': self.config.audio.mp3_bitrate,
                'remove_wav': self.config.audio.remove_wav,
                'split_minutes': self.config.audio.split_minutes,
                'smart_split': self.config.audio.smart_split,
                'silence_threshold': self.config.audio.silence_threshold,
                'min_silence_duration': self.config.audio.min_silence_duration,
            },
            'processing': {
                'max_chunk_chars': self.config.processing.max_chunk_chars,
                'min_chunk_chars': self.config.processing.min_chunk_chars,
                'limit_minutes': self.config.processing.limit_minutes,
                'validate_audio': self.config.processing.validate_audio,
                'regeneration_attempts': self.config.processing.regeneration_attempts,
                'ending_detection': self.config.processing.ending_detection,
            },
            'output': {
                'output_dir_template': self.config.output.output_dir_template,
                'metadata_artist': self.config.output.metadata_artist,
                'metadata_album': self.config.output.metadata_album,
                'metadata_genre': self.config.output.metadata_genre,
                'preserve_structure': self.config.output.preserve_structure,
            }
        }
    
    def save_config(self, file_path: Path):
        """Save current configuration to TOML file"""
        config_dict = self.get_effective_config()
        
        toml_content = self._dict_to_toml(config_dict)
        
        try:
            with open(file_path, 'w') as f:
                f.write(toml_content)
            logging.info(f"💾 Configuration saved to {file_path}")
        except Exception as e:
            raise ConfigError(f"Failed to save config to {file_path}: {e}")
    
    def _dict_to_toml(self, config_dict: Dict[str, Any]) -> str:
        """Convert configuration dictionary to TOML format"""
        lines = [
            "# Chatterbox TTS Configuration File",
            "# Generated automatically - edit as needed",
            "",
        ]
        
        for section_name, section_data in config_dict.items():
            lines.append(f"[{section_name}]")
            
            for key, value in section_data.items():
                if value is None:
                    lines.append(f"# {key} = \"<not set>\"")
                elif isinstance(value, bool):
                    lines.append(f"{key} = {str(value).lower()}")
                elif isinstance(value, str):
                    lines.append(f'{key} = "{value}"')
                else:
                    lines.append(f"{key} = {value}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def print_effective_config(self):
        """Print the current effective configuration"""
        config_dict = self.get_effective_config()
        
        print("📋 Effective Configuration:")
        print("=" * 50)
        
        for section_name, section_data in config_dict.items():
            print(f"\n[{section_name}]")
            for key, value in section_data.items():
                if value is None:
                    print(f"  {key}: <not set>")
                else:
                    print(f"  {key}: {value}")
        
        if self.config_file:
            print(f"\n📁 Config file: {self.config_file}")
        else:
            print(f"\n📁 Config file: <using defaults>")


def create_default_config() -> ConfigManager:
    """Create a config manager with default settings"""
    return ConfigManager()


def load_config(config_file: Optional[Union[str, Path]] = None) -> ConfigManager:
    """Load configuration from file or defaults"""
    if config_file:
        config_file = Path(config_file)
    return ConfigManager(config_file)
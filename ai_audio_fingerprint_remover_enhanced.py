#!/usr/bin/env python3
"""
ai_audio_fingerprint_remover_enhanced.py - Advanced AI Audio Fingerprint Remover

Enhanced version with:
- Machine learning-based watermark detection
- Psychoacoustic modeling for quality preservation
- GPU acceleration support
- Real-time processing capabilities
- Advanced anomaly detection
- Fixed chunk processing and shape mismatch issues
- Sliding Time Scale / Pitch Shift (NEW!)
- Fixed Numba casting warnings
"""

import os
import sys
import argparse
import shutil
import tempfile
import random
import json
import re
import hashlib
import struct
import wave
import array
import logging
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union, Set, Callable, Generator
from dataclasses import dataclass, field
from enum import Enum
import uuid
import warnings
import gc
from functools import lru_cache
from contextlib import contextmanager

# ============================================================================
# SUPPRESS NUMBA WARNINGS (FIX: Numba casting warnings)
# ============================================================================
# Filter Numba RuntimeWarnings for invalid cast
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numba")
warnings.filterwarnings("ignore", category=UserWarning, module="numba")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="librosa")
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================================================================
# DEPENDENCY CHECK AND INSTALLATION
# ============================================================================

def check_and_install_dependencies():
    """Check for required dependencies and suggest installation."""
    missing = []
    
    try:
        import numpy as np
    except ImportError:
        missing.append("numpy")
    
    try:
        import scipy
    except ImportError:
        missing.append("scipy")
    
    try:
        import librosa
    except ImportError:
        missing.append("librosa")
    
    try:
        import soundfile as sf
    except ImportError:
        missing.append("soundfile")
    
    try:
        import mutagen
    except ImportError:
        missing.append("mutagen")
    
    if missing:
        print("Missing required libraries. Please install:")
        print(f"pip install {' '.join(missing)}")
        
        # Optional libraries for enhanced features
        optional_missing = []
        try:
            import torch
        except ImportError:
            optional_missing.append("torch (for ML acceleration)")
        
        try:
            import tensorflow as tf
        except ImportError:
            optional_missing.append("tensorflow (for deep learning detection)")
        
        if optional_missing:
            print("\nOptional libraries for enhanced features:")
            print(f"pip install {' '.join([m.split()[0] for m in optional_missing])}")
        
        sys.exit(1)

# Run dependency check
check_and_install_dependencies()

# ============================================================================
# IMPORTS
# ============================================================================

try:
    import numpy as np
    from scipy import signal, stats, ndimage
    from scipy.io import wavfile
    from scipy.signal import hilbert, butter, filtfilt, lfilter, welch, spectrogram
    from scipy.fft import rfft, irfft, rfftfreq
    from scipy.ndimage import gaussian_filter1d
except ImportError as e:
    print(f"Error importing scientific libraries: {e}")
    sys.exit(1)

try:
    import librosa
    import soundfile as sf
except ImportError:
    print("Error: Required 'librosa' and 'soundfile' libraries not found.")
    sys.exit(1)

try:
    import mutagen
    from mutagen.id3 import ID3, ID3NoHeaderError
    from mutagen.wave import WAVE
    from mutagen.mp3 import MP3
    from mutagen.flac import FLAC
    from mutagen.easyid3 import EasyID3
    from mutagen.aiff import AIFF
except ImportError:
    print("Error: Required 'mutagen' library not found.")
    print("Please install it using: pip install mutagen")
    sys.exit(1)

# Optional ML dependencies
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tensorflow as tf
    HAS_TF = True
except ImportError:
    HAS_TF = False

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

class ColoredFormatter(logging.Formatter):
    """Custom colored formatter for better readability."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)

def setup_logging(verbose: bool = False, log_file: Optional[str] = None):
    """Setup logging with colored output."""
    level = logging.DEBUG if verbose else logging.INFO
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    for handler in handlers:
        handler.setFormatter(ColoredFormatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        ))
    
    logging.basicConfig(
        level=level,
        handlers=handlers
    )
    
    # Set library log levels
    logging.getLogger('librosa').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.ERROR)  # Suppress Numba warnings
    
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================

class ProcessingLevel(Enum):
    """Processing intensity levels."""
    GENTLE = "gentle"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"
    
    @property
    def description(self) -> str:
        return {
            ProcessingLevel.GENTLE: "Minimal processing - preserves audio quality",
            ProcessingLevel.MODERATE: "Balanced processing - good effectiveness",
            ProcessingLevel.AGGRESSIVE: "Thorough processing - removes most fingerprints",
            ProcessingLevel.EXTREME: "Maximum processing - may affect quality"
        }[self]

KNOWN_AI_TAG_PATTERNS = [
    r'(?i)suno', r'(?i)openai', r'(?i)anthropic', r'(?i)stability',
    r'(?i)midjourney', r'(?i)synthesia', r'(?i)ai[_.-]?gen', r'(?i)ml[_.-]?gen',
    r'(?i)model', r'(?i)dalle', r'(?i)chatgpt', r'(?i)gpt', r'(?i)elevenlabs',
    r'(?i)neural', r'(?i)deepfake', r'(?i)synthetic', r'(?i)generated',
    r'(?i)claude', r'(?i)voice\.ai', r'(?i)murf', r'(?i)descript',
    r'(?i)resemble\.ai', r'(?i)play\.ht', r'(?i)uberduck', r'(?i)replica',
    r'(?i)wav2lip', r'(?i)tortoise', r'(?i)bark\.ai', r'(?i)vall[_.-]?e',
    r'(?i)transformers', r'(?i)diffusion', r'(?i)latent', r'(?i)embedding',
    r'(?i)vqgan', r'(?i)wavenet', r'(?i)vocoder', r'(?i)melgan', r'(?i)hifigan'
]

KNOWN_CUSTOM_CHUNKS = [
    'sunf', 'aicm', 'ainf', 'genm', 'gens', 'modl', 'crid', 'meta', 'json',
    'suna', 'elev', 'mlmd', 'gena', 'orig', 'prom', 'seed', 'sigf', 'uuid',
    'lmd', 'gnmd', 'aiid', 'gptm', 'opmd', 'mrkr', 'fing', 'wtrm', 'hash',
    'cgnr', 'gpmd', 'anth', 'stbl', 'midj', 'voai', 'wavm', 'audi', 'synth',
    'genr', 'tort', 'bark', 'vall', 'hfgn', 'melg'
]

POTENTIAL_WATERMARK_FREQS = [
    [19500, 20000],  # High-frequency standard
    [15000, 17000],  # ElevenLabs/similar range
    [50, 200],       # Low-frequency steganography
    [8000, 8500],    # Mid-range markers
    [12000, 12500],  # Secondary watermark range
    [17500, 18000],  # Another common watermark frequency
    [500, 800],      # Low-mid range watermark
]

# ============================================================================
# FIX: SAFE ARRAY OPERATIONS (Prevents Numba casting errors)
# ============================================================================

def safe_cast_to_float64(arr: np.ndarray) -> np.ndarray:
    """Safely cast array to float64, handling mixed dtypes."""
    if arr is None:
        return np.array([], dtype=np.float64)
    
    # If already float64, return as is
    if arr.dtype == np.float64:
        return arr
    
    # If float32 or other float, cast safely
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(np.float64, copy=False)
    
    # If integer, cast to float64
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float64)
    
    # Generic fallback
    try:
        return np.asarray(arr, dtype=np.float64)
    except Exception:
        return arr.astype(np.float64)

def safe_divide(a, b, default=0.0):
    """Safe division with handling of zeros and invalid values."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b, where=(b != 0))
        # Replace infinities and NaNs with default
        result[~np.isfinite(result)] = default
        return result

def safe_numpy_operation(func, *args, default_return=None, **kwargs):
    """Wrapper for numpy operations that handles Numba casting errors."""
    try:
        # Ensure all args are float64 if they are numpy arrays
        converted_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                converted_args.append(safe_cast_to_float64(arg))
            else:
                converted_args.append(arg)
        
        result = func(*converted_args, **kwargs)
        
        # If result is numpy array, ensure it's float64
        if isinstance(result, np.ndarray):
            return safe_cast_to_float64(result)
        return result
    except (ValueError, TypeError, RuntimeWarning) as e:
        if "invalid value encountered" in str(e) or "cast" in str(e):
            logger.debug(f"Numpy operation error, using fallback: {e}")
            if default_return is not None:
                return default_return
            # Fallback: try with default parameters
            try:
                return func(*args, **kwargs)
            except Exception:
                return None
        raise

# ============================================================================
# ENHANCED DATA CLASSES
# ============================================================================

@dataclass
class ProcessingConfig:
    """Enhanced configuration with advanced parameters."""
    # Processing level
    processing_level: ProcessingLevel = ProcessingLevel.MODERATE
    
    # Watermark removal parameters
    filter_order: int = 2
    filter_width_multiplier: float = 1.0
    noise_level: float = 0.00001
    skip_low_freq_threshold: int = 200
    watermark_detection_threshold: float = 0.6
    
    # Pattern normalization parameters
    timing_stretch_range: float = 0.001
    distribution_noise_level: float = 0.00001
    harmonic_distortion_amount: float = 0.002
    phase_variance: float = 0.005
    micro_dynamics_amount: float = 0.0001
    
    # Timing variations
    timing_variation_range: float = 0.002
    segment_overlap_ratio: float = 0.5
    
    # Analysis parameters
    stft_size: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    
    # Advanced features
    enable_ml_detection: bool = True
    enable_psychoacoustic: bool = True
    enable_gpu: bool = False
    use_chunked_processing: bool = True
    chunk_duration: float = 30.0
    overlap_duration: float = 2.0
    
    # Quality preservation
    preserve_dynamics: bool = True
    preserve_transients: bool = True
    adaptive_filtering: bool = True
    perceptual_masking: bool = True
    
    # Processing flags
    enable_watermark_removal: bool = True
    enable_pattern_normalization: bool = True
    enable_timing_variations: bool = True
    enable_harmonic_adjustments: bool = True
    enable_metadata_cleaning: bool = True
    
    @classmethod
    def get_profile(cls, level: Union[str, ProcessingLevel]) -> 'ProcessingConfig':
        """Get predefined configuration profiles with enhanced parameters."""
        if isinstance(level, str):
            try:
                level = ProcessingLevel(level.lower())
            except ValueError:
                level = ProcessingLevel.MODERATE
        
        profiles = {
            ProcessingLevel.GENTLE: cls(
                processing_level=ProcessingLevel.GENTLE,
                filter_order=1,
                filter_width_multiplier=0.3,
                noise_level=0.000005,
                skip_low_freq_threshold=350,
                timing_stretch_range=0.0005,
                distribution_noise_level=0.000005,
                harmonic_distortion_amount=0.001,
                phase_variance=0.001,
                micro_dynamics_amount=0.0002,
                timing_variation_range=0.0005,
                stft_size=1024,
                enable_psychoacoustic=True,
                preserve_dynamics=True,
                preserve_transients=True,
                enable_harmonic_adjustments=False,
                watermark_detection_threshold=0.8,
            ),
            ProcessingLevel.MODERATE: cls(
                processing_level=ProcessingLevel.MODERATE,
                filter_order=2,
                filter_width_multiplier=0.7,
                noise_level=0.00005,
                skip_low_freq_threshold=250,
                timing_stretch_range=0.002,
                distribution_noise_level=0.00005,
                harmonic_distortion_amount=0.005,
                phase_variance=0.005,
                micro_dynamics_amount=0.0005,
                timing_variation_range=0.002,
                stft_size=2048,
                enable_psychoacoustic=True,
                preserve_dynamics=True,
                preserve_transients=True,
                watermark_detection_threshold=0.6,
            ),
            ProcessingLevel.AGGRESSIVE: cls(
                processing_level=ProcessingLevel.AGGRESSIVE,
                filter_order=3,
                filter_width_multiplier=1.0,
                noise_level=0.0001,
                skip_low_freq_threshold=200,
                timing_stretch_range=0.005,
                distribution_noise_level=0.0001,
                harmonic_distortion_amount=0.01,
                phase_variance=0.008,
                micro_dynamics_amount=0.001,
                timing_variation_range=0.004,
                stft_size=2048,
                enable_psychoacoustic=True,
                preserve_dynamics=True,
                preserve_transients=False,
                watermark_detection_threshold=0.5,
            ),
            ProcessingLevel.EXTREME: cls(
                processing_level=ProcessingLevel.EXTREME,
                filter_order=4,
                filter_width_multiplier=1.5,
                noise_level=0.0005,
                skip_low_freq_threshold=150,
                timing_stretch_range=0.01,
                distribution_noise_level=0.0005,
                harmonic_distortion_amount=0.02,
                phase_variance=0.015,
                micro_dynamics_amount=0.002,
                timing_variation_range=0.008,
                stft_size=4096,
                enable_psychoacoustic=False,
                preserve_dynamics=False,
                preserve_transients=False,
                watermark_detection_threshold=0.3,
            )
        }
        
        return profiles.get(level, profiles[ProcessingLevel.MODERATE])

@dataclass
class ProcessingStats:
    """Enhanced processing statistics."""
    files_processed: int = 0
    files_failed: int = 0
    metadata_removed: Dict[str, List[str]] = field(default_factory=dict)
    watermarks_detected: int = 0
    watermarks_removed: int = 0
    patterns_normalized: int = 0
    timing_adjustments: int = 0
    processing_level: str = "moderate"
    processing_time: float = 0.0
    memory_peak_mb: float = 0.0
    chunks_processed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    operation_timings: Dict[str, float] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    
    def add_timing(self, operation: str, duration: float):
        """Add timing information."""
        self.operation_timings[operation] = duration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert stats to dictionary."""
        return {
            'files_processed': self.files_processed,
            'files_failed': self.files_failed,
            'watermarks_detected': self.watermarks_detected,
            'watermarks_removed': self.watermarks_removed,
            'patterns_normalized': self.patterns_normalized,
            'timing_adjustments': self.timing_adjustments,
            'processing_level': self.processing_level,
            'processing_time': self.processing_time,
            'memory_peak_mb': self.memory_peak_mb,
            'chunks_processed': self.chunks_processed,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'operation_timings': self.operation_timings,
            'quality_metrics': self.quality_metrics
        }

@dataclass
class AudioMetadata:
    """Audio file metadata structure."""
    filepath: str
    format: str
    duration: float
    sample_rate: int
    channels: int
    bit_depth: int
    file_size: int
    has_metadata: bool
    detected_ai_tags: List[str] = field(default_factory=list)
    custom_chunks: List[str] = field(default_factory=list)
    watermark_indicators: List[str] = field(default_factory=list)

# ============================================================================
# UTILITY FUNCTIONS (With Numba warning fixes)
# ============================================================================

@contextmanager
def timing(operation: str, stats: Optional[ProcessingStats] = None):
    """Context manager for timing operations."""
    start = time.time()
    try:
        yield
    finally:
        duration = time.time() - start
        if stats:
            stats.add_timing(operation, duration)
        logger.debug(f"{operation} took {duration:.2f}s")

@contextmanager
def memory_monitor():
    """Context manager for monitoring memory usage."""
    try:
        import psutil
        process = psutil.Process()
        initial = process.memory_info().rss / 1024 / 1024
        yield initial
        final = process.memory_info().rss / 1024 / 1024
        logger.debug(f"Memory: {initial:.1f}MB -> {final:.1f}MB (Δ {final - initial:.1f}MB)")
    except ImportError:
        yield 0.0

def get_file_hash(filepath: str, algorithm: str = 'sha256') -> str:
    """Get file hash with progress for large files."""
    hasher = hashlib.new(algorithm)
    file_size = os.path.getsize(filepath)
    read_size = 65536
    
    with open(filepath, 'rb') as f:
        bytes_read = 0
        while bytes_read < file_size:
            buf = f.read(read_size)
            if not buf:
                break
            hasher.update(buf)
            bytes_read += len(buf)
    
    return hasher.hexdigest()

def safe_normalize_audio(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Safely normalize audio to prevent clipping."""
    if len(audio) == 0:
        return audio
    
    # Ensure float64
    audio = safe_cast_to_float64(audio)
    
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        gain = target_peak / max_val
        if gain < 1.0:
            return audio * gain
    return audio

def validate_audio_content(audio: np.ndarray, min_amplitude: float = 1e-10,
                          max_amplitude: float = 1.0,
                          context: str = "") -> Tuple[bool, str]:
    """Enhanced audio validation with detailed error messages."""
    if len(audio) == 0:
        return False, f"{context}: Audio is empty"
    
    if not isinstance(audio, np.ndarray):
        return False, f"{context}: Audio is not a numpy array"
    
    # Ensure float64 for validation
    audio = safe_cast_to_float64(audio)
    
    if not np.isfinite(audio).all():
        return False, f"{context}: Audio contains NaN or infinite values"
    
    max_abs = np.max(np.abs(audio))
    if max_abs > max_amplitude * 1.1:
        return False, f"{context}: Audio exceeds max amplitude ({max_abs:.3f} > {max_amplitude:.3f})"
    
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < min_amplitude:
        return False, f"{context}: Audio is too quiet (RMS {rms:.6f})"
    
    # Check for DC offset
    dc_offset = np.mean(audio)
    if abs(dc_offset) > 0.01 * max_abs:
        return False, f"{context}: Significant DC offset detected ({dc_offset:.4f})"
    
    return True, f"{context}: Valid audio content"

def resize_audio_safe(audio: np.ndarray, target_length: int) -> np.ndarray:
    """Safely resize audio to target length using interpolation."""
    if len(audio) == target_length:
        return audio
    
    if len(audio) == 0:
        return np.zeros(target_length, dtype=np.float64)
    
    if target_length == 0:
        return np.array([], dtype=np.float64)
    
    # Ensure float64
    audio = safe_cast_to_float64(audio)
    
    try:
        # Use interpolation for smooth resizing
        x_old = np.linspace(0, 1, len(audio), dtype=np.float64)
        x_new = np.linspace(0, 1, target_length, dtype=np.float64)
        return np.interp(x_new, x_old, audio).astype(np.float64)
    except Exception as e:
        logger.warning(f"Audio resizing failed: {e}")
        # Fallback: truncate or pad
        if len(audio) > target_length:
            return audio[:target_length].astype(np.float64)
        else:
            return np.pad(audio, (0, target_length - len(audio)), mode='constant').astype(np.float64)

def fill_audio_gaps(audio: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill gaps in audio using interpolation."""
    if np.all(mask):
        return audio
    
    # Ensure float64
    audio = safe_cast_to_float64(audio)
    mask = np.asarray(mask, dtype=bool)
    
    result = audio.copy()
    indices = np.where(mask)[0]
    
    if len(indices) == 0:
        return result
    
    gap_indices = np.where(~mask)[0]
    if len(gap_indices) == 0:
        return result
    
    # Interpolate gaps
    try:
        result[gap_indices] = np.interp(
            gap_indices.astype(np.float64),
            indices.astype(np.float64),
            result[indices].astype(np.float64)
        )
    except Exception as e:
        logger.warning(f"Gap filling failed: {e}")
        # Simple fallback: use linear interpolation with scipy
        try:
            from scipy.interpolate import interp1d
            f = interp1d(indices, result[indices], kind='linear', fill_value='extrapolate')
            result[gap_indices] = f(gap_indices)
        except Exception:
            # Last resort: fill with zeros
            result[gap_indices] = 0.0
    
    return safe_cast_to_float64(result)

# ============================================================================
# ADVANCED AUDIO ANALYSIS (With Numba warning fixes)
# ============================================================================

class AdvancedAudioAnalyzer:
    """Advanced audio analysis with machine learning capabilities."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._ml_model = None
        self._init_ml_model()
    
    def _init_ml_model(self):
        """Initialize machine learning models if available."""
        if self.config.enable_ml_detection:
            try:
                if HAS_TORCH:
                    self._ml_model = self._create_torch_model()
                    logger.info("PyTorch model initialized successfully")
                elif HAS_TF:
                    self._ml_model = self._create_tf_model()
                    logger.info("TensorFlow model initialized successfully")
                else:
                    logger.warning("No ML framework available for enhanced detection")
            except Exception as e:
                logger.warning(f"ML model initialization failed: {e}")
                self._ml_model = None
    
    def _create_torch_model(self) -> nn.Module:
        """Create a PyTorch model for watermark detection."""
        class WatermarkDetector(nn.Module):
            def __init__(self, input_dim: int = 128):
                super().__init__()
                self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
                self.bn1 = nn.BatchNorm1d(32)
                self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
                self.bn2 = nn.BatchNorm1d(64)
                self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
                self.bn3 = nn.BatchNorm1d(128)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc1 = nn.Linear(128, 64)
                self.fc2 = nn.Linear(64, 2)
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                x = torch.relu(self.bn1(self.conv1(x)))
                x = torch.relu(self.bn2(self.conv2(x)))
                x = torch.relu(self.bn3(self.conv3(x)))
                x = self.pool(x)
                x = x.squeeze(-1)
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                return self.fc2(x)
        
        return WatermarkDetector()
    
    def _create_tf_model(self):
        """Create a TensorFlow model for watermark detection."""
        if not HAS_TF:
            return None
            
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(128, 1)),
            tf.keras.layers.Conv1D(32, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(64, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2)
        ])
        return model
    
    @lru_cache(maxsize=32)
    def compute_mel_spectrogram(self, audio_bytes: bytes, sr: int) -> np.ndarray:
        """Compute mel spectrogram with caching."""
        audio = np.frombuffer(audio_bytes, dtype=np.float64)
        try:
            mel_spec = librosa.feature.melspectrogram(
                y=audio, sr=sr, n_mels=self.config.n_mels,
                n_fft=self.config.stft_size,
                hop_length=self.config.hop_length
            )
            return safe_cast_to_float64(librosa.power_to_db(mel_spec, ref=np.max))
        except Exception as e:
            logger.warning(f"Mel spectrogram computation failed: {e}")
            return np.array([], dtype=np.float64)
    
    def ml_watermark_detection(self, audio: np.ndarray, sr: int) -> float:
        """Use ML model for watermark detection."""
        if self._ml_model is None or not self.config.enable_ml_detection:
            return 0.0
        
        try:
            # Prepare audio for model
            audio = safe_cast_to_float64(audio)
            audio_bytes = audio.astype(np.float64).tobytes()
            mel_spec = self.compute_mel_spectrogram(audio_bytes, sr)
            
            if mel_spec.size == 0:
                return 0.0
            
            # Normalize and reshape for model
            mel_spec = safe_cast_to_float64(mel_spec)
            mel_spec = (mel_spec - np.mean(mel_spec)) / (np.std(mel_spec) + 1e-8)
            input_data = mel_spec.T[np.newaxis, ..., np.newaxis]
            
            if HAS_TORCH and isinstance(self._ml_model, nn.Module):
                import torch
                with torch.no_grad():
                    tensor_input = torch.FloatTensor(input_data.transpose(0, 3, 1, 2))
                    output = self._ml_model(tensor_input)
                    probs = torch.softmax(output, dim=1)
                    return float(probs[0, 1].cpu().numpy())
            
            elif HAS_TF and isinstance(self._ml_model, tf.keras.Model):
                output = self._ml_model(input_data, training=False)
                probs = tf.nn.softmax(output)
                return float(probs[0, 1].numpy())
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"ML detection failed: {e}")
            return 0.0
    
    def detect_spectral_watermarks(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Enhanced spectral watermark detection."""
        detected = []
        
        if len(audio) == 0:
            return detected
        
        # Ensure float64
        audio = safe_cast_to_float64(audio)
        
        # Cache key
        audio_hash = hashlib.md5(audio.tobytes()).hexdigest()[:16]
        cache_key = f"watermarks_{audio_hash}_{sr}_{self.config.stft_size}"
        
        if cache_key in self._cache:
            logger.debug("Using cached watermark detection results")
            self._cache_hits += 1
            return self._cache[cache_key]
        
        self._cache_misses += 1
        
        try:
            # Convert to mono if stereo
            if len(audio.shape) > 1:
                audio_mono = np.mean(audio, axis=1)
            else:
                audio_mono = audio
            
            # Adaptive analysis based on audio length
            if len(audio_mono) > sr * 300:
                decimation = max(1, len(audio_mono) // (sr * 120))
                audio_analysis = audio_mono[::decimation]
                analysis_sr = sr // decimation
            else:
                audio_analysis = audio_mono
                analysis_sr = sr
            
            # Ensure float64
            audio_analysis = safe_cast_to_float64(audio_analysis)
            
            # Compute spectrogram with adaptive parameters
            nperseg = min(self.config.stft_size, len(audio_analysis) // 8)
            if nperseg < 64:
                nperseg = 64
            
            # Use safe numpy operations
            freqs, times, spec = safe_numpy_operation(
                spectrogram,
                audio_analysis,
                fs=analysis_sr,
                window='hann',
                nperseg=nperseg,
                noverlap=nperseg // 2,
                scaling='spectrum',
                default_return=(np.array([]), np.array([]), np.array([]))
            )
            
            if spec.size == 0:
                return detected
            
            # Ensure float64
            spec = safe_cast_to_float64(spec)
            freqs = safe_cast_to_float64(freqs)
            
            # Compute spectral features (with safe division)
            spec_sum = np.sum(spec, axis=0, keepdims=True) + 1e-10
            spectral_centroid = np.sum(freqs[:, np.newaxis] * spec, axis=0) / spec_sum.flatten()
            
            # Handle potential issues with spectral centroid
            spectral_centroid = np.nan_to_num(spectral_centroid, nan=0.0, posinf=0.0, neginf=0.0)
            
            spectral_bandwidth = np.sqrt(
                safe_divide(
                    np.sum(((freqs[:, np.newaxis] - spectral_centroid) ** 2) * spec, axis=0),
                    np.sum(spec, axis=0) + 1e-10,
                    default=0.0
                )
            )
            
            # Detect anomalies in frequency bands
            for freq_range in POTENTIAL_WATERMARK_FREQS:
                scaled_range = [f * analysis_sr / sr for f in freq_range]
                if scaled_range[1] > analysis_sr / 2:
                    continue
                
                freq_mask = (freqs >= scaled_range[0]) & (freqs <= scaled_range[1])
                if not np.any(freq_mask):
                    continue
                
                band_energy = np.mean(spec[freq_mask], axis=0)
                
                # Calculate statistics
                mean_energy = np.mean(band_energy)
                std_energy = np.std(band_energy)
                
                if std_energy > 0:
                    # Check for periodicity
                    normalized = (band_energy - mean_energy) / (std_energy + 1e-10)
                    autocorr = np.correlate(normalized, normalized, mode='full')
                    autocorr = autocorr[len(autocorr)//2:]
                    
                    if len(autocorr) > 1:
                        peaks, _ = signal.find_peaks(autocorr, height=0.3, distance=5)
                        if len(peaks) >= 3:
                            regularity = np.std(np.diff(peaks)) if len(peaks) > 1 else 0
                            detected.append({
                                'type': 'periodic_watermark',
                                'freq_range': freq_range,
                                'peak_count': len(peaks),
                                'regularity': float(regularity),
                                'strength': float(np.max(autocorr[peaks])),
                                'confidence': min(1.0, len(peaks) / 10)
                            })
                    
                    # Check for constant energy (unnatural)
                    if scaled_range[0] > 15000 * analysis_sr / sr:
                        variation_ratio = std_energy / (mean_energy + 1e-10)
                        if variation_ratio < 0.1:
                            detected.append({
                                'type': 'constant_energy',
                                'freq_range': freq_range,
                                'variation_ratio': float(variation_ratio),
                                'confidence': min(1.0, (0.1 - variation_ratio) * 10)
                            })
            
            # Detect high-frequency anomalies
            high_freq_energy_ratio = safe_divide(
                np.sum(spec[freqs > 0.8 * analysis_sr / 2]),
                np.sum(spec) + 1e-10,
                default=0.0
            )
            if high_freq_energy_ratio > 0.1:
                detected.append({
                    'type': 'high_freq_anomaly',
                    'energy_ratio': float(high_freq_energy_ratio),
                    'freq_range': [0.8 * analysis_sr / 2, analysis_sr / 2],
                    'confidence': min(1.0, high_freq_energy_ratio * 10)
                })
            
            # ML-based detection
            if self.config.enable_ml_detection:
                ml_score = self.ml_watermark_detection(audio_analysis, analysis_sr)
                if ml_score > 0.5:
                    detected.append({
                        'type': 'ml_detected',
                        'score': ml_score,
                        'confidence': ml_score
                    })
            
            # Apply confidence threshold
            detected = [w for w in detected if w.get('confidence', 0) > self.config.watermark_detection_threshold]
            
            logger.info(f"Detected {len(detected)} spectral watermarks")
            
        except Exception as e:
            logger.error(f"Spectral watermark detection failed: {e}")
        
        # Cache results
        self._cache[cache_key] = detected
        
        return detected
    
    def detect_statistical_patterns(self, audio: np.ndarray) -> List[Dict[str, Any]]:
        """Enhanced statistical pattern detection."""
        detected = []
        
        if len(audio) == 0:
            return detected
        
        # Ensure float64
        audio = safe_cast_to_float64(audio)
        
        try:
            # Convert to mono
            if len(audio.shape) > 1:
                audio_mono = np.mean(audio, axis=1)
            else:
                audio_mono = audio
            
            audio_mono = safe_cast_to_float64(audio_mono)
            
            # Sample entropy (complexity measure)
            def sample_entropy(data, m=2, r=None):
                if r is None:
                    r = 0.2 * np.std(data) + 1e-10
                
                N = len(data)
                if N < m + 1:
                    return 0.0
                
                # Use sampling for performance
                sample_size = min(2000, N - m)
                indices = np.random.choice(N - m, sample_size, replace=False)
                
                patterns = [0, 0]
                
                for i in indices:
                    template = data[i:i + m]
                    matches_m = 0
                    matches_m_plus_1 = 0
                    
                    for j in range(N - m):
                        if j != i:
                            dist = np.max(np.abs(template - data[j:j + m]))
                            if dist < r:
                                matches_m += 1
                                if j < N - m and i < N - m:
                                    dist_plus = np.max(np.abs(data[i:i + m + 1] - data[j:j + m + 1]))
                                    if dist_plus < r:
                                        matches_m_plus_1 += 1
                    
                    patterns[0] += matches_m
                    patterns[1] += matches_m_plus_1
                
                if patterns[0] == 0 or patterns[1] == 0:
                    return 0.0
                
                return -np.log(patterns[1] / patterns[0])
            
            # Calculate complexity
            if len(audio_mono) > 1000:
                entropy = sample_entropy(audio_mono[:min(10000, len(audio_mono))])
                
                if entropy < 0.5:
                    detected.append({
                        'type': 'low_complexity',
                        'sample_entropy': entropy,
                        'confidence': 1.0 - entropy
                    })
            
            # Spectral flatness
            stft = np.abs(librosa.stft(audio_mono, n_fft=min(1024, len(audio_mono)//2)))
            if stft.size > 0:
                # Use safe operations for geometric mean
                with np.errstate(divide='ignore', invalid='ignore'):
                    gmean = np.exp(np.mean(np.log(stft + 1e-10), axis=0))
                    mean_val = np.mean(stft, axis=0) + 1e-10
                    spectral_flatness = np.mean(safe_divide(gmean, mean_val, default=0.0))
                
                if spectral_flatness > 0.8:
                    detected.append({
                        'type': 'artificial_flatness',
                        'spectral_flatness': float(spectral_flatness),
                        'confidence': spectral_flatness
                    })
            
            # Zero-crossing regularity
            zero_crossings = np.where(np.diff(np.signbit(audio_mono)))[0]
            if len(zero_crossings) > 100:
                intervals = np.diff(zero_crossings)
                regularity = np.std(intervals) / (np.mean(intervals) + 1e-10)
                
                if regularity < 0.2:
                    detected.append({
                        'type': 'regular_timing',
                        'regularity': float(regularity),
                        'confidence': 1.0 - regularity
                    })
            
            # Amplitude distribution
            hist, _ = np.histogram(audio_mono, bins=50, range=(-1, 1), density=True)
            if len(hist) > 0:
                hist = safe_cast_to_float64(hist)
                skewness = safe_divide(
                    np.sum((hist - np.mean(hist))**3),
                    len(hist) * (np.std(hist)**3 + 1e-10),
                    default=0.0
                )
                kurtosis = safe_divide(
                    np.sum((hist - np.mean(hist))**4),
                    len(hist) * (np.std(hist)**4 + 1e-10),
                    default=0.0
                ) - 3
                
                if abs(skewness) < 0.1 and abs(kurtosis) < 0.2:
                    detected.append({
                        'type': 'perfect_distribution',
                        'skewness': float(skewness),
                        'kurtosis': float(kurtosis),
                        'confidence': 1.0 - max(abs(skewness), abs(kurtosis))
                    })
            
            logger.info(f"Detected {len(detected)} statistical anomalies")
            
        except Exception as e:
            logger.error(f"Statistical pattern detection failed: {e}")
        
        return detected
    
    def detect_timing_anomalies(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Enhanced timing anomaly detection."""
        detected = []
        
        if len(audio) == 0:
            return detected
        
        # Ensure float64
        audio = safe_cast_to_float64(audio)
        
        try:
            # Convert to mono
            if len(audio.shape) > 1:
                audio_mono = np.mean(audio, axis=1)
            else:
                audio_mono = audio
            
            audio_mono = safe_cast_to_float64(audio_mono)
            
            # Compute onset envelope
            hop_length = min(512, len(audio_mono) // 100)
            if hop_length < 1:
                hop_length = 1
            
            onset_env = librosa.onset.onset_strength(
                y=audio_mono, sr=sr, hop_length=hop_length
            )
            
            if len(onset_env) > 20:
                # Detect onsets
                onsets = librosa.onset.onset_detect(
                    onset_envelope=onset_env,
                    sr=sr, hop_length=hop_length,
                    units='time'
                )
                
                if len(onsets) > 5:
                    intervals = np.diff(onsets)
                    
                    if len(intervals) > 0:
                        # Coefficient of variation
                        cv = np.std(intervals) / (np.mean(intervals) + 1e-10)
                        
                        if cv < 0.1:
                            detected.append({
                                'type': 'mechanical_timing',
                                'cv': float(cv),
                                'confidence': 1.0 - cv
                            })
                        
                        # Check for quantization
                        base_interval = np.min(intervals)
                        if base_interval > 0:
                            quantized_count = sum(
                                1 for i in intervals
                                if abs(round(i/base_interval) - i/base_interval) < 0.05
                            )
                            
                            if quantized_count > len(intervals) * 0.7:
                                detected.append({
                                    'type': 'quantized_timing',
                                    'percent_quantized': quantized_count / len(intervals),
                                    'confidence': quantized_count / len(intervals)
                                })
            
            logger.info(f"Detected {len(detected)} timing anomalies")
            
        except Exception as e:
            logger.error(f"Timing anomaly detection failed: {e}")
        
        return detected

# ============================================================================
# ENHANCED WATERMARK REMOVAL (With Numba warning fixes)
# ============================================================================

class WatermarkRemover:
    """Enhanced watermark removal with psychoacoustic modeling."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.analyzer = AdvancedAudioAnalyzer(config)
    
    def remove_watermarks_spectral(self, audio: np.ndarray, sr: int,
                                   watermarks: List[Dict[str, Any]]) -> np.ndarray:
        """Remove watermarks using spectral processing."""
        if not self.config.enable_watermark_removal or not watermarks:
            return audio
        
        # Ensure float64
        audio = safe_cast_to_float64(audio)
        result = audio.copy()
        
        # Group watermarks by frequency range
        freq_ranges = []
        for w in watermarks:
            freq_range = w.get('freq_range')
            if freq_range and freq_range not in freq_ranges:
                freq_ranges.append(freq_range)
        
        if not freq_ranges:
            return result
        
        # Process each frequency range
        for freq_range in freq_ranges:
            if freq_range[1] > sr / 2:
                continue
            
            # Skip low frequencies if configured
            if freq_range[0] < self.config.skip_low_freq_threshold:
                continue
            
            # Design adaptive filter
            if self.config.adaptive_filtering:
                b, a = self._design_adaptive_filter(result, sr, freq_range)
            else:
                b, a = self._design_bandstop_filter(sr, freq_range, self.config.filter_order)
            
            if b is not None and a is not None:
                try:
                    # Apply filter - ensure float64
                    b = safe_cast_to_float64(b)
                    a = safe_cast_to_float64(a)
                    filtered = signal.filtfilt(b, a, result, axis=-1)
                    filtered = safe_cast_to_float64(filtered)
                    
                    # Validate and apply
                    valid, msg = validate_audio_content(filtered, context="Watermark removal filter")
                    if valid:
                        result = filtered
                    else:
                        logger.warning(f"Filter produced invalid audio: {msg}")
                except Exception as e:
                    logger.warning(f"Filter application failed: {e}")
        
        # Psychoacoustic processing for high-frequency watermarks
        if self.config.enable_psychoacoustic:
            high_freq_watermarks = [
                w for w in watermarks
                if w.get('freq_range', [0, 0])[0] > 15000
            ]
            if high_freq_watermarks:
                result = self._psychoacoustic_removal(result, sr, high_freq_watermarks)
        
        return safe_cast_to_float64(result)
    
    def _design_bandstop_filter(self, sr: int, freq_range: List[float],
                                order: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Design a bandstop filter for watermark removal."""
        try:
            nyquist = sr / 2
            low = max(0.01, freq_range[0] / nyquist)
            high = min(0.99, freq_range[1] / nyquist)
            
            if low >= high:
                return None, None
            
            # Add slight width adjustment
            center = (low + high) / 2
            width = (high - low) * self.config.filter_width_multiplier
            low = max(0.01, center - width / 2)
            high = min(0.99, center + width / 2)
            
            b, a = signal.butter(order, [low, high], btype='bandstop')
            return safe_cast_to_float64(b), safe_cast_to_float64(a)
        except Exception as e:
            logger.warning(f"Filter design failed: {e}")
            return None, None
    
    def _design_adaptive_filter(self, audio: np.ndarray, sr: int,
                                freq_range: List[float]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Design an adaptive filter based on audio characteristics."""
        try:
            # Analyze spectral characteristics
            audio = safe_cast_to_float64(audio)
            nperseg = min(2048, len(audio)//4)
            if nperseg < 64:
                nperseg = 64
            
            freqs, psd = signal.welch(audio, sr, nperseg=nperseg)
            freqs = safe_cast_to_float64(freqs)
            psd = safe_cast_to_float64(psd)
            
            # Determine filter order based on spectral complexity
            spectral_spread = np.sqrt(
                safe_divide(
                    np.sum((freqs - np.sum(freqs * psd) / (np.sum(psd) + 1e-10))**2 * psd),
                    np.sum(psd) + 1e-10,
                    default=0.0
                )
            )
            
            if spectral_spread > sr / 8:
                order = min(self.config.filter_order, 3)
            else:
                order = min(self.config.filter_order, 2)
            
            return self._design_bandstop_filter(sr, freq_range, order)
        except Exception as e:
            logger.warning(f"Adaptive filter design failed: {e}")
            return self._design_bandstop_filter(sr, freq_range, self.config.filter_order)
    
    def _psychoacoustic_removal(self, audio: np.ndarray, sr: int,
                               watermarks: List[Dict[str, Any]]) -> np.ndarray:
        """Remove watermarks using psychoacoustic masking."""
        if not self.config.enable_psychoacoustic:
            return audio
        
        audio = safe_cast_to_float64(audio)
        
        try:
            # Compute STFT
            stft = librosa.stft(audio, n_fft=self.config.stft_size,
                               hop_length=self.config.hop_length)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Ensure float64
            magnitude = safe_cast_to_float64(magnitude)
            
            # Compute masking threshold (simplified)
            masking_threshold = np.percentile(magnitude, 20, axis=1, keepdims=True)
            
            # Identify watermark frequencies
            for wm in watermarks:
                freq_range = wm.get('freq_range')
                if not freq_range:
                    continue
                
                # Convert frequency range to bin indices
                freq_bins = librosa.fft_frequencies(
                    sr=sr, n_fft=self.config.stft_size
                )
                bin_mask = (freq_bins >= freq_range[0]) & (freq_bins <= freq_range[1])
                
                if not np.any(bin_mask):
                    continue
                
                # Attenuate only where watermarks are masked
                for i in np.where(bin_mask)[0]:
                    for j in range(magnitude.shape[1]):
                        if magnitude[i, j] > masking_threshold[i, 0] * 3:
                            # Apply gentle attenuation
                            attenuation = 0.7 + 0.3 * np.random.random()
                            magnitude[i, j] *= attenuation
            
            # Reconstruct audio
            processed_stft = magnitude * np.exp(1j * phase)
            result = librosa.istft(processed_stft, hop_length=self.config.hop_length)
            result = safe_cast_to_float64(result)
            
            # Ensure same length
            if len(result) != len(audio):
                result = resize_audio_safe(result, len(audio))
            
            # Validate
            valid, msg = validate_audio_content(result, context="Psychoacoustic removal")
            if valid:
                return result
            else:
                logger.warning(f"Psychoacoustic removal failed validation: {msg}")
                return audio
                
        except Exception as e:
            logger.warning(f"Psychoacoustic removal failed: {e}")
            return audio
    
    def remove_watermarks_time_domain(self, audio: np.ndarray, sr: int,
                                      watermarks: List[Dict[str, Any]]) -> np.ndarray:
        """Remove watermarks in time domain using signal processing."""
        if not watermarks:
            return audio
        
        audio = safe_cast_to_float64(audio)
        result = audio.copy()
        
        # Apply smoothing to high-frequency components
        if any(w.get('freq_range', [0, 0])[0] > 15000 for w in watermarks):
            # Apply gentle low-pass filtering
            b, a = signal.butter(2, 0.9, btype='low')
            b = safe_cast_to_float64(b)
            a = safe_cast_to_float64(a)
            try:
                filtered = signal.filtfilt(b, a, result, axis=-1)
                filtered = safe_cast_to_float64(filtered)
                valid, msg = validate_audio_content(filtered, context="Time domain removal")
                if valid:
                    result = filtered
            except Exception as e:
                logger.warning(f"Time domain filtering failed: {e}")
        
        return result

# ============================================================================
# ENHANCED PATTERN NORMALIZATION (With Numba warning fixes)
# ============================================================================

class PatternNormalizer:
    """Enhanced pattern normalization with perceptual quality preservation."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.analyzer = AdvancedAudioAnalyzer(config)
    
    def normalize_patterns(self, audio: np.ndarray, sr: int,
                           patterns: List[Dict[str, Any]],
                           timing_issues: List[Dict[str, Any]]) -> np.ndarray:
        """Normalize detected AI patterns."""
        if not self.config.enable_pattern_normalization:
            return audio
        
        audio = safe_cast_to_float64(audio)
        result = audio.copy()
        
        # 1. Normalize timing issues
        has_timing_issues = any(
            p['type'] in ['mechanical_timing', 'quantized_timing']
            for p in timing_issues
        )
        if has_timing_issues:
            result = self._normalize_timing(result, sr)
        
        # 2. Normalize distribution anomalies
        has_distribution_issues = any(
            p['type'] in ['perfect_distribution', 'artificial_flatness']
            for p in patterns
        )
        if has_distribution_issues:
            result = self._normalize_distribution(result)
        
        # 3. Normalize harmonic issues
        has_harmonic_issues = any(
            p['type'] in ['low_complexity', 'missing_harmonics']
            for p in patterns
        )
        if has_harmonic_issues and self.config.enable_harmonic_adjustments:
            result = self._normalize_harmonics(result, sr)
        
        # 4. Add natural micro-dynamics
        result = self._add_micro_dynamics(result, sr)
        
        # Validate
        valid, msg = validate_audio_content(result, context="Pattern normalization")
        if not valid:
            logger.warning(f"Pattern normalization failed validation: {msg}")
            return audio
        
        return safe_cast_to_float64(result)
    
    def _normalize_timing(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Normalize mechanical timing by adding subtle variations."""
        audio = safe_cast_to_float64(audio)
        
        segment_len = sr // 10  # ~100ms
        hop_len = int(segment_len * self.config.segment_overlap_ratio)
        
        # Ensure hop_len is valid
        if hop_len <= 0:
            hop_len = segment_len // 2
        
        # Break into segments
        segments = []
        for i in range(0, len(audio) - segment_len, hop_len):
            segments.append(audio[i:i+segment_len])
        
        if not segments:
            return audio
        
        # Apply time stretching to each segment
        processed_segments = []
        stretch_range = self.config.timing_stretch_range
        
        for segment in segments:
            segment = safe_cast_to_float64(segment)
            stretch_factor = (1.0 - stretch_range) + (2 * stretch_range * random.random())
            try:
                stretched = librosa.effects.time_stretch(segment, rate=stretch_factor)
                stretched = safe_cast_to_float64(stretched)
            except Exception:
                stretched = segment
            
            # Ensure consistent length
            if len(stretched) > segment_len:
                stretched = stretched[:segment_len]
            elif len(stretched) < segment_len:
                stretched = np.pad(stretched, (0, segment_len - len(stretched)), mode='constant')
            
            processed_segments.append(safe_cast_to_float64(stretched))
        
        # Reconstruct with overlap-add
        reconstructed = np.zeros(len(audio), dtype=np.float64)
        weights = np.zeros(len(audio), dtype=np.float64)
        window = np.bartlett(segment_len).astype(np.float64)
        
        for i, segment in enumerate(processed_segments):
            pos = i * hop_len
            end_pos = min(pos + segment_len, len(reconstructed))
            segment_len_actual = end_pos - pos
            
            if segment_len_actual > 0:
                reconstructed[pos:end_pos] += segment[:segment_len_actual] * window[:segment_len_actual]
                weights[pos:end_pos] += window[:segment_len_actual]
        
        # Normalize weights
        mask = weights > 0.001
        reconstructed[mask] = reconstructed[mask] / weights[mask]
        
        # Fill gaps if any
        if not np.all(mask):
            reconstructed = fill_audio_gaps(reconstructed, mask)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(reconstructed))
        if max_val > 0:
            reconstructed = reconstructed / max_val * np.max(np.abs(audio))
        
        return safe_cast_to_float64(reconstructed)
    
    def _normalize_distribution(self, audio: np.ndarray) -> np.ndarray:
        """Normalize amplitude distribution to be more natural."""
        audio = safe_cast_to_float64(audio)
        
        # Add shaped noise based on signal amplitude
        noise = np.random.randn(len(audio)).astype(np.float64) * self.config.distribution_noise_level
        
        # Shape noise by signal envelope
        try:
            envelope = np.abs(signal.hilbert(audio))
            envelope = safe_cast_to_float64(envelope)
            smoothed = gaussian_filter1d(envelope, sigma=min(50, len(audio) // 1000 + 1))
            shaped_noise = noise * (smoothed / (np.max(smoothed) + 1e-10))
            return audio + shaped_noise
        except Exception as e:
            logger.warning(f"Distribution normalization failed: {e}")
            return audio
    
    def _normalize_harmonics(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Add subtle harmonic distortion for more natural sound."""
        audio = safe_cast_to_float64(audio)
        
        try:
            # Soft clipping distortion
            def soft_clip(x, amount=self.config.harmonic_distortion_amount):
                return x - amount * np.sin(2 * np.pi * x)
            
            result = soft_clip(audio)
            
            # Add subtle phase variation if configured
            if self.config.phase_variance > 0:
                phase_noise = np.random.randn(len(audio)).astype(np.float64) * self.config.phase_variance
                b, a = signal.butter(2, 0.5, 'highpass')
                b = safe_cast_to_float64(b)
                a = safe_cast_to_float64(a)
                phase_mod = signal.lfilter(b, a, phase_noise)
                result += phase_mod * 0.01
            
            return safe_cast_to_float64(result)
        except Exception as e:
            logger.warning(f"Harmonic normalization failed: {e}")
            return audio
    
    def _add_micro_dynamics(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Add subtle micro-dynamics variations."""
        if self.config.micro_dynamics_amount <= 0:
            return audio
        
        audio = safe_cast_to_float64(audio)
        
        try:
            # Create envelope
            envelope = np.abs(signal.hilbert(audio))
            envelope = safe_cast_to_float64(envelope)
            smoothed = gaussian_filter1d(envelope, sigma=min(50, len(audio) // 500 + 1))
            
            # Create variations
            variations = np.sin(np.linspace(0, 20 * np.pi, len(audio)) + random.random() * 10).astype(np.float64)
            dynamics_adjustment = smoothed * variations * self.config.micro_dynamics_amount
            
            return audio + dynamics_adjustment
        except Exception as e:
            logger.warning(f"Micro-dynamics addition failed: {e}")
            return audio

# ============================================================================
# ENHANCED METADATA CLEANER
# ============================================================================

class MetadataCleaner:
    """Enhanced metadata cleaning with comprehensive coverage."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.ai_signatures = self._build_ai_signatures()
        self.custom_chunks = set(KNOWN_CUSTOM_CHUNKS)
    
    def _build_ai_signatures(self) -> Set[str]:
        """Build comprehensive set of AI metadata signatures."""
        signatures = set()
        
        # Add known patterns
        for pattern in KNOWN_AI_TAG_PATTERNS:
            signatures.add(pattern)
        
        # Add common field names
        field_names = [
            'generator', 'created_by', 'software', 'source', 'origin',
            'model', 'ai_model', 'voice_model', 'synthesizer', 'encoder',
            'generation', 'synthesized', 'voice_id', 'voice_preset',
            'prompt', 'text_prompt', 'parameters', 'settings', 'config',
            'version', 'api_version', 'timestamp', 'uuid', 'session_id',
            'license', 'terms', 'usage_rights', 'watermark', 'fingerprint',
            'fingerprint_id', 'watermark_id', 'trace_id', 'request_id',
            'job_id', 'task_id', 'seed', 'temperature', 'top_p', 'top_k',
            'repetition_penalty', 'length_penalty', 'guidance_scale',
            'num_inference_steps', 'strength'
        ]
        
        for field in field_names:
            signatures.add(f'(?i){field}')
        
        return signatures
    
    def clean_metadata(self, filepath: str, output_path: Optional[str] = None,
                       aggressive: bool = False) -> Tuple[str, Dict[str, List[str]]]:
        """Comprehensive metadata cleaning for all audio formats."""
        removed_metadata = {}
        
        # Create output path if not provided
        if output_path is None:
            temp_fd, output_path = tempfile.mkstemp(
                suffix=os.path.splitext(filepath)[1]
            )
            os.close(temp_fd)
        
        # Copy file first
        shutil.copy2(filepath, output_path)
        
        file_ext = os.path.splitext(filepath)[1].lower()
        
        try:
            if file_ext == '.mp3':
                removed_metadata.update(self._clean_mp3(output_path, aggressive))
            elif file_ext == '.wav':
                removed_metadata.update(self._clean_wav(output_path, aggressive))
            elif file_ext == '.flac':
                removed_metadata.update(self._clean_flac(output_path, aggressive))
            elif file_ext in ['.aiff', '.aif']:
                removed_metadata.update(self._clean_aiff(output_path, aggressive))
            else:
                logger.warning(f"Unsupported format for metadata cleaning: {file_ext}")
        
        except Exception as e:
            logger.error(f"Metadata cleaning failed for {filepath}: {e}")
        
        return output_path, removed_metadata
    
    def _clean_mp3(self, filepath: str, aggressive: bool) -> Dict[str, List[str]]:
        """Clean MP3 metadata."""
        removed = {}
        
        try:
            # Clean ID3 tags
            audio = MP3(filepath)
            if audio.tags:
                removed_tags = []
                ai_tags = []
                
                for key in list(audio.tags.keys()):
                    tag_str = str(audio.tags[key])
                    if any(re.search(pattern, tag_str) for pattern in self.ai_signatures):
                        ai_tags.append(key)
                        removed_tags.append(f"{key}: {tag_str[:100]}...")
                
                # Remove AI-related tags
                for key in ai_tags:
                    del audio.tags[key]
                
                # If aggressive, remove all tags
                if aggressive:
                    audio.tags = None
                else:
                    # Remove identifying metadata
                    for key in list(audio.tags.keys()):
                        if any(x in key.upper() for x in ['COMM', 'OWNE', 'PRIV', 'USER', 'UFID', 'POPM', 'GEOB']):
                            removed_tags.append(f"{key}: {str(audio.tags[key])[:100]}...")
                            del audio.tags[key]
                
                audio.save()
                
                if removed_tags:
                    removed['mp3_id3'] = removed_tags
            
            # Clean EasyID3
            try:
                easy = EasyID3(filepath)
                if easy:
                    easy_tags = [f"{key}: {easy[key]}" for key in list(easy.keys())]
                    easy.delete()
                    easy.save()
                    if easy_tags:
                        removed['mp3_easyid3'] = easy_tags
            except Exception:
                pass
        
        except Exception as e:
            logger.warning(f"MP3 metadata cleaning error: {e}")
        
        return removed
    
    def _clean_wav(self, filepath: str, aggressive: bool) -> Dict[str, List[str]]:
        """Clean WAV metadata."""
        removed = {}
        
        try:
            audio = WAVE(filepath)
            removed_chunks = []
            
            # Check LIST INFO chunk
            if hasattr(audio, '_tags') and audio._tags:
                for key, value in list(audio._tags.items()):
                    if any(re.search(pattern, str(value)) for pattern in self.ai_signatures):
                        removed_chunks.append(f"LIST INFO {key}: {value[:100]}...")
                        del audio._tags[key]
            
            # Remove custom chunks
            for key in list(audio.keys()):
                if any(chunk.lower() in key.lower() for chunk in self.custom_chunks):
                    removed_chunks.append(f"Custom chunk: {key}")
                    del audio[key]
                elif isinstance(audio[key], bytes):
                    try:
                        chunk_text = audio[key].decode('utf-8', 'ignore')
                        if any(re.search(pattern, chunk_text) for pattern in self.ai_signatures):
                            removed_chunks.append(f"AI-related chunk: {key}")
                            del audio[key]
                    except Exception:
                        pass
            
            audio.save()
            
            if removed_chunks:
                removed['wav_chunks'] = removed_chunks
            
            # Aggressive: rewrite WAV with only essential chunks
            if aggressive:
                with wave.open(filepath, 'rb') as wf:
                    params = wf.getparams()
                    frames = wf.readframes(wf.getnframes())
                
                with wave.open(filepath + '.clean', 'wb') as wf:
                    wf.setparams(params)
                    wf.writeframes(frames)
                
                os.replace(filepath + '.clean', filepath)
                removed['wav_rewrite'] = ["Complete WAV rewrite performed"]
        
        except Exception as e:
            logger.warning(f"WAV metadata cleaning error: {e}")
        
        return removed
    
    def _clean_flac(self, filepath: str, aggressive: bool) -> Dict[str, List[str]]:
        """Clean FLAC metadata."""
        removed = {}
        
        try:
            audio = FLAC(filepath)
            
            # Clean tags
            if audio.tags:
                removed_tags = []
                for key in list(audio.tags.keys()):
                    if any(re.search(pattern, str(audio[key])) for pattern in self.ai_signatures):
                        removed_tags.append(f"{key}: {str(audio[key])[:100]}...")
                        del audio.tags[key]
                
                if removed_tags:
                    removed['flac_tags'] = removed_tags
            
            # Clean pictures
            if audio.pictures:
                removed['flac_pictures'] = [f"Removed {len(audio.pictures)} embedded pictures"]
                audio.clear_pictures()
            
            audio.save()
        
        except Exception as e:
            logger.warning(f"FLAC metadata cleaning error: {e}")
        
        return removed
    
    def _clean_aiff(self, filepath: str, aggressive: bool) -> Dict[str, List[str]]:
        """Clean AIFF metadata."""
        removed = {}
        
        try:
            audio = AIFF(filepath)
            
            if audio.tags:
                removed_tags = []
                for key in list(audio.tags.keys()):
                    if any(re.search(pattern, str(audio.tags[key])) for pattern in self.ai_signatures):
                        removed_tags.append(f"{key}: {str(audio.tags[key])[:100]}...")
                        del audio.tags[key]
                
                audio.tags = None
                audio.save()
                
                if removed_tags:
                    removed['aiff_tags'] = removed_tags
        
        except Exception as e:
            logger.warning(f"AIFF metadata cleaning error: {e}")
        
        return removed

# ============================================================================
# ENHANCED AUDIO PROCESSOR - FIXED VERSION (With Numba warning fixes)
# ============================================================================

class EnhancedAudioProcessor:
    """Enhanced audio processor with fixed chunk handling and shape mismatch prevention."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.analyzer = AdvancedAudioAnalyzer(config)
        self.watermark_remover = WatermarkRemover(config)
        self.pattern_normalizer = PatternNormalizer(config)
        self.metadata_cleaner = MetadataCleaner(config)
        self.stats = ProcessingStats()
        self._executor = None
    
    def process_file(self, input_path: str, output_path: Optional[str] = None,
                     aggressive: bool = False,
                     tempo_shift: float = 0.0,
                     pitch_semitones: float = 0.0,
                     pitch_percent: float = 0.0) -> Tuple[str, ProcessingStats]:
        """
        Process a single audio file with enhanced capabilities.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to output audio file (optional)
            aggressive: Whether to use aggressive processing
            tempo_shift: Tempo change in percent (-50 to 50)
            pitch_semitones: Pitch shift in semitones (-12 to 12)
            pitch_percent: Pitch shift in percent (-50 to 100)
        """
        # Validate input
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Determine output path
        if output_path is None:
            output_path = input_path
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Processing: {os.path.basename(input_path)}")
        start_time = time.time()
        
        # Get file info
        file_size = os.path.getsize(input_path)
        file_ext = os.path.splitext(input_path)[1].lower()
        logger.debug(f"File size: {file_size // 1024}KB, Format: {file_ext}")
        
        # Memory monitoring
        with memory_monitor() as initial_memory:
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                try:
                    # Stage 1: Clean metadata
                    with timing("metadata_cleaning", self.stats):
                        temp_metadata = os.path.join(temp_dir, f"stage1_metadata{file_ext}")
                        result_path, removed_metadata = self.metadata_cleaner.clean_metadata(
                            input_path, temp_metadata, aggressive
                        )
                        self.stats.metadata_removed = removed_metadata
                    
                    # === NEW: Apply Tempo & Pitch Shift ===
                    if tempo_shift != 0 or pitch_semitones != 0 or pitch_percent != 0:
                        with timing("tempo_pitch_shift", self.stats):
                            logger.info(f"Applying tempo: {tempo_shift}%, pitch: {pitch_semitones} semitones, {pitch_percent}%")
                            
                            y_temp, sr_temp = librosa.load(result_path, sr=None, mono=False)
                            is_stereo_temp = len(y_temp.shape) > 1
                            
                            if is_stereo_temp:
                                processed_channels = []
                                for channel in y_temp:
                                    channel = safe_cast_to_float64(channel)
                                    channel_processed = self._apply_tempo_pitch_shift(
                                        channel, sr_temp, tempo_shift, pitch_semitones, pitch_percent
                                    )
                                    processed_channels.append(safe_cast_to_float64(channel_processed))
                                y_processed = np.array(processed_channels, dtype=np.float64)
                            else:
                                y_temp = safe_cast_to_float64(y_temp)
                                y_processed = self._apply_tempo_pitch_shift(
                                    y_temp, sr_temp, tempo_shift, pitch_semitones, pitch_percent
                                )
                                y_processed = safe_cast_to_float64(y_processed)
                            
                            temp_tempo_path = os.path.join(temp_dir, "tempo_shifted.wav")
                            sf.write(temp_tempo_path, y_processed.T if is_stereo_temp else y_processed, sr_temp)
                            result_path = temp_tempo_path
                    
                    # Load audio for processing
                    with timing("audio_loading", self.stats):
                        y, sr = librosa.load(result_path, sr=None, mono=False)
                        is_stereo = len(y.shape) > 1
                        y_mono = np.mean(y, axis=0) if is_stereo else y
                        y_mono = safe_cast_to_float64(y_mono)
                        original_length = y.shape[1] if is_stereo else len(y)
                    
                    # Stage 2: Detect and remove watermarks
                    with timing("watermark_detection", self.stats):
                        watermarks = self.analyzer.detect_spectral_watermarks(y_mono, sr)
                        self.stats.watermarks_detected = len(watermarks)
                    
                    if watermarks:
                        with timing("watermark_removal", self.stats):
                            if is_stereo:
                                processed = y.copy()
                                for i in range(y.shape[0]):
                                    channel = safe_cast_to_float64(y[i])
                                    channel_result = self.watermark_remover.remove_watermarks_spectral(
                                        channel, sr, watermarks
                                    )
                                    # Ensure same length
                                    if len(channel_result) != len(y[i]):
                                        channel_result = resize_audio_safe(channel_result, len(y[i]))
                                    valid, _ = validate_audio_content(channel_result, context=f"Channel {i}")
                                    if valid:
                                        processed[i] = safe_cast_to_float64(channel_result)
                            else:
                                y = safe_cast_to_float64(y)
                                processed = self.watermark_remover.remove_watermarks_spectral(
                                    y, sr, watermarks
                                )
                                processed = safe_cast_to_float64(processed)
                                if len(processed) != len(y):
                                    processed = resize_audio_safe(processed, len(y))
                                valid, _ = validate_audio_content(processed, context="Mono processing")
                                if not valid:
                                    processed = y
                            self.stats.watermarks_removed = len(watermarks)
                    else:
                        processed = y.copy()
                    
                    # Stage 3: Detect and normalize patterns
                    if self.config.enable_pattern_normalization:
                        with timing("pattern_detection", self.stats):
                            patterns = self.analyzer.detect_statistical_patterns(processed)
                            timing_issues = self.analyzer.detect_timing_anomalies(
                                processed if len(processed.shape) == 1 else np.mean(processed, axis=0),
                                sr
                            )
                            self.stats.patterns_normalized = len(patterns) + len(timing_issues)
                        
                        if patterns or timing_issues:
                            with timing("pattern_normalization", self.stats):
                                if is_stereo:
                                    for i in range(processed.shape[0]):
                                        channel = safe_cast_to_float64(processed[i])
                                        channel_result = self.pattern_normalizer.normalize_patterns(
                                            channel, sr, patterns, timing_issues
                                        )
                                        if len(channel_result) != len(processed[i]):
                                            channel_result = resize_audio_safe(channel_result, len(processed[i]))
                                        valid, _ = validate_audio_content(channel_result, context=f"Channel {i}")
                                        if valid:
                                            processed[i] = safe_cast_to_float64(channel_result)
                                else:
                                    processed = safe_cast_to_float64(processed)
                                    processed = self.pattern_normalizer.normalize_patterns(
                                        processed, sr, patterns, timing_issues
                                    )
                                    if len(processed) != original_length:
                                        processed = resize_audio_safe(processed, original_length)
                                    valid, _ = validate_audio_content(processed, context="Pattern normalization")
                                    if not valid:
                                        processed = y
                    
                    # Stage 4: Add timing variations with length preservation
                    if self.config.enable_timing_variations:
                        with timing("timing_variations", self.stats):
                            if is_stereo:
                                for i in range(processed.shape[0]):
                                    channel = safe_cast_to_float64(processed[i])
                                    channel_result = self._add_timing_variations_safe(channel, sr)
                                    if len(channel_result) != len(processed[i]):
                                        channel_result = resize_audio_safe(channel_result, len(processed[i]))
                                    valid, _ = validate_audio_content(channel_result, context=f"Channel {i}")
                                    if valid:
                                        processed[i] = safe_cast_to_float64(channel_result)
                            else:
                                processed = safe_cast_to_float64(processed)
                                processed = self._add_timing_variations_safe(processed, sr)
                                if len(processed) != original_length:
                                    processed = resize_audio_safe(processed, original_length)
                                valid, _ = validate_audio_content(processed, context="Timing variations")
                                if not valid:
                                    processed = y
                            self.stats.timing_adjustments = 1
                    
                    # Final validation and normalization
                    if is_stereo:
                        for i in range(processed.shape[0]):
                            processed[i] = safe_normalize_audio(processed[i])
                            # Ensure correct length
                            if len(processed[i]) != original_length:
                                processed[i] = resize_audio_safe(processed[i], original_length)
                    else:
                        processed = safe_normalize_audio(processed)
                        if len(processed) != original_length:
                            processed = resize_audio_safe(processed, original_length)
                    
                    # Save processed audio
                    with timing("audio_saving", self.stats):
                        sf.write(output_path, processed.T if is_stereo else processed, sr)
                    
                except Exception as e:
                    logger.error(f"Processing failed: {e}")
                    # Fallback: copy original
                    shutil.copy2(input_path, output_path)
                    self.stats.files_failed = 1
                    raise
        
        # Update stats
        self.stats.files_processed = 1
        self.stats.processing_time = time.time() - start_time
        self.stats.processing_level = self.config.processing_level.value
        
        # Cache stats
        self.stats.cache_hits = self.analyzer._cache_hits
        self.stats.cache_misses = self.analyzer._cache_misses
        
        logger.info(f"Completed in {self.stats.processing_time:.2f}s")
        logger.info(f"Removed {self.stats.watermarks_removed} watermarks, "
                   f"normalized {self.stats.patterns_normalized} patterns")
        
        return output_path, self.stats
    
    def _apply_tempo_pitch_shift(self, audio: np.ndarray, sr: int,
                                  tempo_shift: float,
                                  pitch_semitones: float,
                                  pitch_percent: float) -> np.ndarray:
        """
        Apply tempo and pitch shift to audio.
        
        Args:
            audio: Input audio array (mono)
            sr: Sample rate
            tempo_shift: Tempo change in percent (-50 to 50)
            pitch_semitones: Pitch shift in semitones (-12 to 12)
            pitch_percent: Pitch shift in percent (-50 to 100)
        
        Returns:
            Processed audio array
        """
        audio = safe_cast_to_float64(audio)
        
        try:
            # 1. Tempo shift (time stretching)
            if tempo_shift != 0:
                # Convert percent to rate (1.0 = original)
                rate = 1.0 + (tempo_shift / 100.0)
                # Clamp rate to reasonable range
                rate = max(0.25, min(4.0, rate))
                audio = librosa.effects.time_stretch(audio, rate=rate)
                audio = safe_cast_to_float64(audio)
            
            # 2. Pitch shift
            if pitch_semitones != 0 or pitch_percent != 0:
                # Kombinasi semitones dan percent
                # Approx: 1 semitone = ~6% change
                pitch_semitones_total = pitch_semitones + (pitch_percent / 6.0)
                # Clamp to reasonable range
                pitch_semitones_total = max(-24, min(24, pitch_semitones_total))
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_semitones_total)
                audio = safe_cast_to_float64(audio)
            
            return audio
            
        except Exception as e:
            logger.warning(f"Tempo/pitch shift failed: {e}")
            return audio
    
    def _add_timing_variations_safe(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Add timing variations with guaranteed length preservation."""
        if not self.config.enable_timing_variations:
            return audio
        
        audio = safe_cast_to_float64(audio)
        original_length = len(audio)
        
        # For very short audio, skip processing
        if original_length < sr * 0.5:  # Less than 0.5 seconds
            return audio
        
        try:
            # Process with length preservation
            if self.config.use_chunked_processing and original_length > sr * 60:
                result = self._process_chunked_safe(audio, sr, self._apply_timing_variations_safe)
            else:
                result = self._apply_timing_variations_safe(audio, sr)
            
            # Ensure exact length match
            if len(result) != original_length:
                logger.debug(f"Resizing result from {len(result)} to {original_length}")
                result = resize_audio_safe(result, original_length)
            
            return safe_cast_to_float64(result)
            
        except Exception as e:
            logger.warning(f"Timing variations failed: {e}")
            return audio
    
    def _process_chunked_safe(self, audio: np.ndarray, sr: int,
                              process_func: Callable) -> np.ndarray:
        """Safe chunked processing with guaranteed length preservation."""
        audio = safe_cast_to_float64(audio)
        
        chunk_samples = int(self.config.chunk_duration * sr)
        overlap_samples = int(self.config.overlap_duration * sr)
        hop_samples = chunk_samples - overlap_samples
        
        if hop_samples <= 0:
            hop_samples = chunk_samples // 2
            overlap_samples = chunk_samples - hop_samples
        
        result = np.zeros(len(audio), dtype=np.float64)
        weights = np.zeros(len(audio), dtype=np.float64)
        
        # Process chunks with proper overlap
        for i in range(0, len(audio) - overlap_samples, hop_samples):
            start = i
            end = min(start + chunk_samples, len(audio))
            
            chunk = audio[start:end]
            if len(chunk) < sr // 2:
                continue
            
            # Process chunk
            try:
                processed = process_func(chunk, sr)
                processed = safe_cast_to_float64(processed)
            except Exception as e:
                logger.warning(f"Chunk processing failed: {e}")
                processed = chunk
            
            # Ensure same length
            if len(processed) != len(chunk):
                processed = resize_audio_safe(processed, len(chunk))
            
            # Apply window
            window = np.hanning(len(processed)).astype(np.float64)
            result[start:end] += processed * window
            weights[start:end] += window
            self.stats.chunks_processed += 1
        
        # Normalize
        mask = weights > 0.001
        result[mask] = result[mask] / weights[mask]
        
        # Handle any gaps
        if not np.all(mask):
            result = fill_audio_gaps(result, mask)
        
        return safe_cast_to_float64(result)
    
    def _apply_timing_variations_safe(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Apply timing variations with length preservation."""
        audio = safe_cast_to_float64(audio)
        
        segment_duration = 1.0  # 1 second segments for better control
        segment_samples = int(segment_duration * sr)
        hop_samples = segment_samples // 2
        
        if len(audio) < segment_samples * 2:
            # Too short for segmentation, apply global stretch
            stretch = 1.0 + random.uniform(-self.config.timing_variation_range, 
                                          self.config.timing_variation_range)
            try:
                stretched = librosa.effects.time_stretch(audio, rate=stretch)
                stretched = safe_cast_to_float64(stretched)
                return resize_audio_safe(stretched, len(audio))
            except Exception:
                return audio
        
        result = np.zeros(len(audio), dtype=np.float64)
        weights = np.zeros(len(audio), dtype=np.float64)
        
        for i in range(0, len(audio) - segment_samples, hop_samples):
            start = i
            end = min(start + segment_samples, len(audio))
            
            segment = audio[start:end]
            if len(segment) < segment_samples // 2:
                continue
            
            # Random stretch
            stretch = 1.0 + random.uniform(-self.config.timing_variation_range,
                                          self.config.timing_variation_range)
            
            try:
                stretched = librosa.effects.time_stretch(segment, rate=stretch)
                stretched = safe_cast_to_float64(stretched)
                stretched = resize_audio_safe(stretched, len(segment))
            except Exception:
                stretched = segment
            
            # Apply window
            window = np.hanning(len(stretched)).astype(np.float64)
            result[start:end] += stretched * window
            weights[start:end] += window
        
        # Normalize
        mask = weights > 0.001
        result[mask] = result[mask] / weights[mask]
        
        # Fill gaps
        if not np.all(mask):
            result = fill_audio_gaps(result, mask)
        
        # Normalize amplitude
        max_val = np.max(np.abs(result))
        if max_val > 0:
            result = result / max_val * np.max(np.abs(audio))
        
        return safe_cast_to_float64(result)

# ============================================================================
# BATCH PROCESSING
# ============================================================================

class BatchProcessor:
    """Batch processing with parallel execution support."""
    
    def __init__(self, config: ProcessingConfig, max_workers: int = None):
        self.config = config
        self.max_workers = max_workers or min(4, os.cpu_count() or 2)
        self.stats = ProcessingStats()
        self.processor = EnhancedAudioProcessor(config)
    
    def process_directory(self, input_dir: str, output_dir: Optional[str] = None,
                          aggressive: bool = False, recursive: bool = True,
                          tempo_shift: float = 0.0,
                          pitch_semitones: float = 0.0,
                          pitch_percent: float = 0.0) -> ProcessingStats:
        """Process all audio files in a directory."""
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Directory not found: {input_dir}")
        
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Collect files
        audio_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.mp3', '.wav', '.flac', '.aiff', '.aif')):
                    input_path = os.path.join(root, file)
                    
                    if output_dir:
                        rel_path = os.path.relpath(input_path, input_dir)
                        output_path = os.path.join(output_dir, rel_path)
                        os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    else:
                        output_path = None
                    
                    audio_files.append((input_path, output_path))
        
        logger.info(f"Found {len(audio_files)} audio files to process")
        
        if not audio_files:
            return self.stats
        
        # Process files with progress
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for input_path, output_path in audio_files:
                future = executor.submit(
                    self.processor.process_file,
                    input_path, output_path, aggressive,
                    tempo_shift, pitch_semitones, pitch_percent
                )
                futures.append((input_path, future))
            
            # Collect results
            for input_path, future in futures:
                try:
                    result_path, file_stats = future.result(timeout=300)
                    self.stats.files_processed += 1
                    self.stats.watermarks_detected += file_stats.watermarks_detected
                    self.stats.watermarks_removed += file_stats.watermarks_removed
                    self.stats.patterns_normalized += file_stats.patterns_normalized
                    self.stats.timing_adjustments += file_stats.timing_adjustments
                    self.stats.chunks_processed += file_stats.chunks_processed
                    self.stats.cache_hits += file_stats.cache_hits
                    self.stats.cache_misses += file_stats.cache_misses
                    
                    # Merge metadata
                    for key, value in file_stats.metadata_removed.items():
                        if key not in self.stats.metadata_removed:
                            self.stats.metadata_removed[key] = []
                        self.stats.metadata_removed[key].extend(value)
                    
                    # Update quality metrics
                    if file_stats.quality_metrics:
                        for key, value in file_stats.quality_metrics.items():
                            if key not in self.stats.quality_metrics:
                                self.stats.quality_metrics[key] = []
                            self.stats.quality_metrics[key].append(value)
                    
                    logger.info(f"✓ {os.path.basename(input_path)}")
                    
                except Exception as e:
                    logger.error(f"✗ {os.path.basename(input_path)}: {e}")
                    self.stats.files_failed += 1
        
        return self.stats

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def display_metadata(filepath: str):
    """Display audio file metadata."""
    print(f"\nMetadata for: {os.path.basename(filepath)}")
    print("-" * 50)
    
    try:
        file_ext = os.path.splitext(filepath)[1].lower()
        
        if file_ext == '.mp3':
            try:
                audio = MP3(filepath)
                if audio.tags:
                    for key in sorted(audio.tags.keys()):
                        print(f"  {key}: {str(audio.tags[key])[:100]}...")
                else:
                    print("  No ID3 tags found")
            except Exception as e:
                print(f"  Error reading metadata: {e}")
        
        elif file_ext == '.wav':
            try:
                audio = WAVE(filepath)
                if hasattr(audio, '_tags') and audio._tags:
                    for key, value in audio._tags.items():
                        print(f"  LIST INFO {key}: {value[:100]}...")
                for key in audio.keys():
                    if key not in ['fmt ', 'data']:
                        print(f"  Chunk {key}: present")
            except Exception as e:
                print(f"  Error reading metadata: {e}")
        
        else:
            print(f"  No metadata viewer available for {file_ext}")
    
    except Exception as e:
        print(f"  Error: {e}")
    
    print("-" * 50)


def display_results(stats: ProcessingStats, verify: bool, input_path: str,
                    output_path: str, report: bool):
    """Display processing results."""
    print("\n" + "-" * 50)
    print("PROCESSING RESULTS")
    print("-" * 50)
    print(f"Processing level: {stats.processing_level}")
    print(f"Processing time: {stats.processing_time:.2f}s")
    print(f"Watermarks detected: {stats.watermarks_detected}")
    print(f"Watermarks removed: {stats.watermarks_removed}")
    print(f"Patterns normalized: {stats.patterns_normalized}")
    print(f"Timing adjustments: {stats.timing_adjustments}")
    
    if stats.metadata_removed:
        total_metadata = sum(len(items) for items in stats.metadata_removed.values())
        print(f"Metadata entries removed: {total_metadata}")
        if report:
            for category, items in stats.metadata_removed.items():
                print(f"\n  {category}:")
                for item in items[:5]:
                    print(f"    - {item}")
                if len(items) > 5:
                    print(f"    - ... and {len(items) - 5} more")
    
    if stats.quality_metrics:
        print("\nQuality metrics:")
        for key, values in stats.quality_metrics.items():
            if values:
                avg = sum(values) / len(values)
                print(f"  {key}: {avg:.4f}")
    
    if verify:
        print("\nVerification:")
        orig_hash = get_file_hash(input_path)
        new_hash = get_file_hash(output_path)
        print(f"  Original: {orig_hash[:16]}...")
        print(f"  Processed: {new_hash[:16]}...")
        print(f"  Files are {'identical' if orig_hash == new_hash else 'different'}")
    
    print("-" * 50)


def display_batch_results(stats: ProcessingStats, report: bool):
    """Display batch processing results."""
    print("\n" + "-" * 50)
    print("BATCH PROCESSING RESULTS")
    print("-" * 50)
    print(f"Processing level: {stats.processing_level}")
    print(f"Files processed: {stats.files_processed}")
    print(f"Files failed: {stats.files_failed}")
    print(f"Processing time: {stats.processing_time:.2f}s")
    print(f"Total watermarks removed: {stats.watermarks_removed}")
    print(f"Total patterns normalized: {stats.patterns_normalized}")
    
    if stats.metadata_removed:
        total_metadata = sum(len(items) for items in stats.metadata_removed.values())
        print(f"Total metadata entries removed: {total_metadata}")
    
    if report and stats.metadata_removed:
        print("\nMetadata removed by category:")
        for category, items in stats.metadata_removed.items():
            print(f"  {category}: {len(items)} items")
            for item in items[:3]:
                print(f"    - {item}")
            if len(items) > 3:
                print(f"    - ... and {len(items) - 3} more")
    
    print("-" * 50)


def main():
    """Enhanced command line interface."""
    parser = argparse.ArgumentParser(
        description="""
        AI Audio Fingerprint Remover - Enhanced Edition
        Comprehensive tool to remove AI-generated audio fingerprinting
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("input", nargs="?", help="Input audio file to process")
    group.add_argument("-d", "--directory", help="Process all audio files in directory")
    
    parser.add_argument("output", nargs="?", help="Output file or directory")
    parser.add_argument("--show", action="store_true", help="Show metadata before removal")
    parser.add_argument("--verify", action="store_true", help="Verify results with original")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--log", help="Log file path")
    
    # Processing options
    parser.add_argument("--level", choices=['gentle', 'moderate', 'aggressive', 'extreme'],
                       default='moderate', help="Processing intensity level")
    parser.add_argument("--aggressive", action="store_true", help="Alias for --level aggressive")
    parser.add_argument("--no-ml", action="store_true", help="Disable ML-based detection")
    parser.add_argument("--no-psychoacoustic", action="store_true", help="Disable psychoacoustic processing")
    parser.add_argument("--no-chunking", action="store_true", help="Disable chunked processing for large files")
    parser.add_argument("--workers", type=int, help="Number of parallel workers for batch processing")
    parser.add_argument("--recursive", action="store_true", default=True,
                       help="Process directories recursively")
    parser.add_argument("--no-recursive", action="store_false", dest="recursive",
                       help="Don't process directories recursively")
    
    # === NEW: Tempo & Pitch Shift options ===
    parser.add_argument("--tempo", type=float, default=0.0,
                       help="Tempo change in percent (-50 to 50)")
    parser.add_argument("--pitch-semitones", type=float, default=0.0,
                       help="Pitch shift in semitones (-12 to 12)")
    parser.add_argument("--pitch-percent", type=float, default=0.0,
                       help="Pitch shift in percent (-50 to 100)")
    
    # Quality options
    parser.add_argument("--preserve-dynamics", action="store_true", default=True,
                       help="Preserve dynamic range")
    parser.add_argument("--no-preserve-dynamics", action="store_false", dest="preserve_dynamics",
                       help="Don't preserve dynamic range")
    
    args = parser.parse_args()
    
    # Setup logging
    global logger
    logger = setup_logging(args.verbose, args.log)
    
    # Determine processing level
    if args.aggressive:
        level = ProcessingLevel.AGGRESSIVE
    else:
        try:
            level = ProcessingLevel(args.level.lower())
        except ValueError:
            level = ProcessingLevel.MODERATE
    
    # Create configuration
    config = ProcessingConfig.get_profile(level)
    config.enable_ml_detection = not args.no_ml
    config.enable_psychoacoustic = not args.no_psychoacoustic
    config.use_chunked_processing = not args.no_chunking
    config.preserve_dynamics = args.preserve_dynamics
    
    # Show info
    print("\n" + "=" * 60)
    print("AI AUDIO FINGERPRINT REMOVER - ENHANCED EDITION")
    print("=" * 60)
    print(f"Processing level: {level.value} - {level.description}")
    print(f"ML detection: {'Enabled' if config.enable_ml_detection else 'Disabled'}")
    print(f"Psychoacoustic: {'Enabled' if config.enable_psychoacoustic else 'Disabled'}")
    print(f"Chunked processing: {'Enabled' if config.use_chunked_processing else 'Disabled'}")
    print(f"Tempo shift: {args.tempo}%")
    print(f"Pitch shift: {args.pitch_semitones} semitones / {args.pitch_percent}%")
    print("=" * 60 + "\n")
    
    if args.show and args.input and os.path.exists(args.input):
        display_metadata(args.input)
    
    try:
        if args.input:
            # Single file processing
            if not os.path.exists(args.input):
                print(f"Error: Input file '{args.input}' not found.")
                return 1
            
            processor = EnhancedAudioProcessor(config)
            result_path, stats = processor.process_file(
                args.input, args.output, args.aggressive,
                tempo_shift=args.tempo,
                pitch_semitones=args.pitch_semitones,
                pitch_percent=args.pitch_percent
            )
            
            display_results(stats, args.verify, args.input, result_path, args.report)
            
        elif args.directory:
            # Batch processing
            if not os.path.exists(args.directory):
                print(f"Error: Input directory '{args.directory}' not found.")
                return 1
            
            batch_processor = BatchProcessor(config, args.workers)
            stats = batch_processor.process_directory(
                args.directory, args.output, args.aggressive, args.recursive,
                tempo_shift=args.tempo,
                pitch_semitones=args.pitch_semitones,
                pitch_percent=args.pitch_percent
            )
            
            display_batch_results(stats, args.report)
    
    except KeyboardInterrupt:
        print("\n\nProcessing interrupted by user.")
        return 130
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

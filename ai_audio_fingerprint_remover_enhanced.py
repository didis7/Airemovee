#!/usr/bin/env python3
"""
ai_audio_fingerprint_remover_enhanced.py - Advanced AI Audio Fingerprint Remover
Version for Streamlit deployment
"""

import os
import sys
import argparse
import shutil
import tempfile
import random
import re
import hashlib
import wave
import logging
import time
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from contextlib import contextmanager

# ============================================================================
# SUPPRESS WARNINGS
# ============================================================================
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# DEPENDENCY CHECK
# ============================================================================

def check_dependencies():
    """Check for required dependencies."""
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
        print(f"Missing required libraries: {', '.join(missing)}")
        print("Please install: pip install " + " ".join(missing))
        return False
    
    return True

# ============================================================================
# IMPORTS
# ============================================================================

try:
    import numpy as np
    from scipy import signal
    from scipy.signal import butter, filtfilt, welch, spectrogram
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
    from mutagen.wave import WAVE
    from mutagen.mp3 import MP3
    from mutagen.flac import FLAC
    from mutagen.aiff import AIFF
except ImportError:
    print("Error: Required 'mutagen' library not found.")
    sys.exit(1)

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(verbose: bool = False):
    """Setup logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger('librosa').setLevel(logging.WARNING)
    logging.getLogger('numba').setLevel(logging.ERROR)
    return logging.getLogger(__name__)

# Global logger - akan diinisialisasi ulang di main
logger = setup_logging()

# ============================================================================
# CONSTANTS
# ============================================================================

class ProcessingLevel(Enum):
    GENTLE = "gentle"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"

KNOWN_AI_TAG_PATTERNS = [
    r'(?i)suno', r'(?i)openai', r'(?i)anthropic', r'(?i)stability',
    r'(?i)midjourney', r'(?i)synthesia', r'(?i)ai[_.-]?gen', r'(?i)ml[_.-]?gen',
    r'(?i)model', r'(?i)dalle', r'(?i)chatgpt', r'(?i)gpt', r'(?i)elevenlabs',
    r'(?i)neural', r'(?i)deepfake', r'(?i)synthetic', r'(?i)generated',
    r'(?i)claude', r'(?i)voice\.ai', r'(?i)murf', r'(?i)descript',
]

KNOWN_CUSTOM_CHUNKS = [
    'sunf', 'aicm', 'ainf', 'genm', 'gens', 'modl', 'crid', 'meta', 'json',
    'suna', 'elev', 'mlmd', 'gena', 'orig', 'prom', 'seed', 'sigf', 'uuid',
]

POTENTIAL_WATERMARK_FREQS = [
    [19500, 20000],
    [15000, 17000],
    [50, 200],
    [8000, 8500],
    [12000, 12500],
    [17500, 18000],
]

# ============================================================================
# SAFE ARRAY OPERATIONS
# ============================================================================

def safe_cast_to_float64(arr):
    """Safely cast array to float64."""
    if arr is None:
        return np.array([], dtype=np.float64)
    if arr.dtype == np.float64:
        return arr
    try:
        return arr.astype(np.float64)
    except Exception:
        return np.asarray(arr, dtype=np.float64)

def safe_divide(a, b, default=0.0):
    """Safe division."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b, where=(b != 0))
        result[~np.isfinite(result)] = default
        return result

def safe_normalize_audio(audio, target_peak=0.95):
    """Safely normalize audio."""
    if len(audio) == 0:
        return audio
    audio = safe_cast_to_float64(audio)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        gain = target_peak / max_val
        if gain < 1.0:
            return audio * gain
    return audio

def resize_audio_safe(audio, target_length):
    """Safely resize audio."""
    if len(audio) == target_length:
        return audio
    if len(audio) == 0:
        return np.zeros(target_length, dtype=np.float64)
    audio = safe_cast_to_float64(audio)
    try:
        x_old = np.linspace(0, 1, len(audio), dtype=np.float64)
        x_new = np.linspace(0, 1, target_length, dtype=np.float64)
        return np.interp(x_new, x_old, audio).astype(np.float64)
    except Exception:
        if len(audio) > target_length:
            return audio[:target_length].astype(np.float64)
        else:
            return np.pad(audio, (0, target_length - len(audio)), mode='constant').astype(np.float64)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ProcessingConfig:
    processing_level: ProcessingLevel = ProcessingLevel.MODERATE
    filter_order: int = 2
    filter_width_multiplier: float = 1.0
    skip_low_freq_threshold: int = 200
    watermark_detection_threshold: float = 0.6
    timing_stretch_range: float = 0.001
    distribution_noise_level: float = 0.00001
    micro_dynamics_amount: float = 0.0001
    timing_variation_range: float = 0.002
    segment_overlap_ratio: float = 0.5
    stft_size: int = 2048
    enable_watermark_removal: bool = True
    enable_pattern_normalization: bool = True
    enable_metadata_cleaning: bool = True

    @classmethod
    def get_profile(cls, level):
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
                skip_low_freq_threshold=350,
                timing_stretch_range=0.0005,
                distribution_noise_level=0.000005,
                micro_dynamics_amount=0.0002,
                timing_variation_range=0.0005,
                stft_size=1024,
                watermark_detection_threshold=0.8,
            ),
            ProcessingLevel.MODERATE: cls(
                processing_level=ProcessingLevel.MODERATE,
                filter_order=2,
                filter_width_multiplier=0.7,
                skip_low_freq_threshold=250,
                timing_stretch_range=0.002,
                distribution_noise_level=0.00005,
                micro_dynamics_amount=0.0005,
                timing_variation_range=0.002,
                stft_size=2048,
                watermark_detection_threshold=0.6,
            ),
            ProcessingLevel.AGGRESSIVE: cls(
                processing_level=ProcessingLevel.AGGRESSIVE,
                filter_order=3,
                filter_width_multiplier=1.0,
                skip_low_freq_threshold=200,
                timing_stretch_range=0.005,
                distribution_noise_level=0.0001,
                micro_dynamics_amount=0.001,
                timing_variation_range=0.004,
                stft_size=2048,
                watermark_detection_threshold=0.5,
            ),
            ProcessingLevel.EXTREME: cls(
                processing_level=ProcessingLevel.EXTREME,
                filter_order=4,
                filter_width_multiplier=1.5,
                skip_low_freq_threshold=150,
                timing_stretch_range=0.01,
                distribution_noise_level=0.0005,
                micro_dynamics_amount=0.002,
                timing_variation_range=0.008,
                stft_size=4096,
                watermark_detection_threshold=0.3,
            )
        }
        return profiles.get(level, profiles[ProcessingLevel.MODERATE])

@dataclass
class ProcessingStats:
    files_processed: int = 0
    files_failed: int = 0
    metadata_removed: Dict[str, List[str]] = field(default_factory=dict)
    watermarks_detected: int = 0
    watermarks_removed: int = 0
    patterns_normalized: int = 0
    processing_level: str = "moderate"
    processing_time: float = 0.0

# ============================================================================
# METADATA CLEANER
# ============================================================================

class MetadataCleaner:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.ai_signatures = self._build_ai_signatures()
        self.custom_chunks = set(KNOWN_CUSTOM_CHUNKS)

    def _build_ai_signatures(self) -> Set[str]:
        signatures = set()
        for pattern in KNOWN_AI_TAG_PATTERNS:
            signatures.add(pattern)
        field_names = [
            'generator', 'created_by', 'software', 'source', 'origin',
            'model', 'ai_model', 'voice_model', 'synthesizer', 'encoder',
            'generation', 'synthesized', 'voice_id', 'prompt', 'text_prompt',
            'parameters', 'settings', 'config', 'version', 'api_version',
            'timestamp', 'uuid', 'watermark', 'fingerprint'
        ]
        for field in field_names:
            signatures.add(f'(?i){field}')
        return signatures

    def clean_metadata(self, filepath: str, output_path: Optional[str] = None,
                       aggressive: bool = False) -> Tuple[str, Dict[str, List[str]]]:
        """Clean metadata from audio file."""
        removed_metadata = {}
        
        if output_path is None:
            temp_fd, output_path = tempfile.mkstemp(
                suffix=os.path.splitext(filepath)[1]
            )
            os.close(temp_fd)
        
        shutil.copy2(filepath, output_path)
        file_ext = os.path.splitext(filepath)[1].lower()
        
        try:
            if file_ext == '.mp3':
                removed_metadata.update(self._clean_mp3(output_path))
            elif file_ext == '.wav':
                removed_metadata.update(self._clean_wav(output_path, aggressive))
            elif file_ext == '.flac':
                removed_metadata.update(self._clean_flac(output_path))
            elif file_ext in ['.aiff', '.aif']:
                removed_metadata.update(self._clean_aiff(output_path))
        except Exception as e:
            logger.error(f"Metadata cleaning failed: {e}")
        
        return output_path, removed_metadata

    def _clean_mp3(self, filepath: str) -> Dict[str, List[str]]:
        removed = {}
        try:
            audio = MP3(filepath)
            if audio.tags:
                removed_tags = []
                ai_tags = []
                for key in list(audio.tags.keys()):
                    tag_str = str(audio.tags[key])
                    if any(re.search(pattern, tag_str) for pattern in self.ai_signatures):
                        ai_tags.append(key)
                        removed_tags.append(f"{key}: {tag_str[:100]}...")
                for key in ai_tags:
                    del audio.tags[key]
                audio.save()
                if removed_tags:
                    removed['mp3_id3'] = removed_tags
        except Exception as e:
            logger.warning(f"MP3 metadata cleaning error: {e}")
        return removed

    def _clean_wav(self, filepath: str, aggressive: bool) -> Dict[str, List[str]]:
        removed = {}
        try:
            audio = WAVE(filepath)
            removed_chunks = []
            
            if hasattr(audio, '_tags') and audio._tags:
                for key, value in list(audio._tags.items()):
                    if any(re.search(pattern, str(value)) for pattern in self.ai_signatures):
                        removed_chunks.append(f"LIST INFO {key}: {value[:100]}...")
                        del audio._tags[key]
            
            for key in list(audio.keys()):
                if any(chunk.lower() in key.lower() for chunk in self.custom_chunks):
                    removed_chunks.append(f"Custom chunk: {key}")
                    del audio[key]
            
            audio.save()
            if removed_chunks:
                removed['wav_chunks'] = removed_chunks
            
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

    def _clean_flac(self, filepath: str) -> Dict[str, List[str]]:
        removed = {}
        try:
            audio = FLAC(filepath)
            if audio.tags:
                removed_tags = []
                for key in list(audio.tags.keys()):
                    if any(re.search(pattern, str(audio[key])) for pattern in self.ai_signatures):
                        removed_tags.append(f"{key}: {str(audio[key])[:100]}...")
                        del audio.tags[key]
                if removed_tags:
                    removed['flac_tags'] = removed_tags
            if audio.pictures:
                removed['flac_pictures'] = [f"Removed {len(audio.pictures)} embedded pictures"]
                audio.clear_pictures()
            audio.save()
        except Exception as e:
            logger.warning(f"FLAC metadata cleaning error: {e}")
        return removed

    def _clean_aiff(self, filepath: str) -> Dict[str, List[str]]:
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
# WATERMARK REMOVER
# ============================================================================

class WatermarkRemover:
    def __init__(self, config: ProcessingConfig):
        self.config = config

    def remove_watermarks_spectral(self, audio: np.ndarray, sr: int,
                                   watermarks: List[Dict[str, Any]]) -> np.ndarray:
        """Remove watermarks using spectral processing."""
        if not self.config.enable_watermark_removal or not watermarks:
            return audio
        
        audio = safe_cast_to_float64(audio)
        result = audio.copy()
        
        freq_ranges = []
        for w in watermarks:
            freq_range = w.get('freq_range')
            if freq_range and freq_range not in freq_ranges:
                freq_ranges.append(freq_range)
        
        if not freq_ranges:
            return result
        
        for freq_range in freq_ranges:
            if freq_range[1] > sr / 2:
                continue
            if freq_range[0] < self.config.skip_low_freq_threshold:
                continue
            
            b, a = self._design_bandstop_filter(sr, freq_range, self.config.filter_order)
            if b is not None and a is not None:
                try:
                    b = safe_cast_to_float64(b)
                    a = safe_cast_to_float64(a)
                    filtered = filtfilt(b, a, result)
                    filtered = safe_cast_to_float64(filtered)
                    result = filtered
                except Exception as e:
                    logger.warning(f"Filter application failed: {e}")
        
        return safe_cast_to_float64(result)

    def _design_bandstop_filter(self, sr: int, freq_range: List[float],
                                order: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            nyquist = sr / 2
            low = max(0.01, freq_range[0] / nyquist)
            high = min(0.99, freq_range[1] / nyquist)
            if low >= high:
                return None, None
            
            center = (low + high) / 2
            width = (high - low) * self.config.filter_width_multiplier
            low = max(0.01, center - width / 2)
            high = min(0.99, center + width / 2)
            
            b, a = butter(order, [low, high], btype='bandstop')
            return safe_cast_to_float64(b), safe_cast_to_float64(a)
        except Exception as e:
            logger.warning(f"Filter design failed: {e}")
            return None, None

# ============================================================================
# PATTERN NORMALIZER
# ============================================================================

class PatternNormalizer:
    def __init__(self, config: ProcessingConfig):
        self.config = config

    def normalize_patterns(self, audio: np.ndarray, sr: int,
                           patterns: List[Dict[str, Any]],
                           timing_issues: List[Dict[str, Any]]) -> np.ndarray:
        if not self.config.enable_pattern_normalization:
            return audio
        
        audio = safe_cast_to_float64(audio)
        result = audio.copy()
        
        if patterns:
            result = self._normalize_distribution(result)
        
        if timing_issues:
            result = self._normalize_timing(result, sr)
        
        result = self._add_micro_dynamics(result, sr)
        
        return safe_cast_to_float64(result)

    def _normalize_timing(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = safe_cast_to_float64(audio)
        segment_len = sr // 10
        hop_len = int(segment_len * self.config.segment_overlap_ratio)
        if hop_len <= 0:
            hop_len = segment_len // 2
        
        segments = []
        for i in range(0, len(audio) - segment_len, hop_len):
            segments.append(audio[i:i+segment_len])
        
        if not segments:
            return audio
        
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
            
            if len(stretched) > segment_len:
                stretched = stretched[:segment_len]
            elif len(stretched) < segment_len:
                stretched = np.pad(stretched, (0, segment_len - len(stretched)), mode='constant')
            processed_segments.append(safe_cast_to_float64(stretched))
        
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
        
        mask = weights > 0.001
        reconstructed[mask] = reconstructed[mask] / weights[mask]
        
        max_val = np.max(np.abs(reconstructed))
        if max_val > 0:
            reconstructed = reconstructed / max_val * np.max(np.abs(audio))
        
        return safe_cast_to_float64(reconstructed)

    def _normalize_distribution(self, audio: np.ndarray) -> np.ndarray:
        audio = safe_cast_to_float64(audio)
        noise = np.random.randn(len(audio)).astype(np.float64) * self.config.distribution_noise_level
        try:
            envelope = np.abs(signal.hilbert(audio))
            envelope = safe_cast_to_float64(envelope)
            smoothed = gaussian_filter1d(envelope, sigma=min(50, len(audio) // 1000 + 1))
            shaped_noise = noise * (smoothed / (np.max(smoothed) + 1e-10))
            return audio + shaped_noise
        except Exception:
            return audio

    def _add_micro_dynamics(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if self.config.micro_dynamics_amount <= 0:
            return audio
        audio = safe_cast_to_float64(audio)
        try:
            envelope = np.abs(signal.hilbert(audio))
            envelope = safe_cast_to_float64(envelope)
            smoothed = gaussian_filter1d(envelope, sigma=min(50, len(audio) // 500 + 1))
            variations = np.sin(np.linspace(0, 20 * np.pi, len(audio)) + random.random() * 10).astype(np.float64)
            dynamics_adjustment = smoothed * variations * self.config.micro_dynamics_amount
            return audio + dynamics_adjustment
        except Exception:
            return audio

# ============================================================================
# AUDIO PROCESSOR
# ============================================================================

class EnhancedAudioProcessor:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.watermark_remover = WatermarkRemover(config)
        self.pattern_normalizer = PatternNormalizer(config)
        self.metadata_cleaner = MetadataCleaner(config)
        self.stats = ProcessingStats()

    def process_file(self, input_path: str, output_path: Optional[str] = None,
                     aggressive: bool = False,
                     tempo_shift: float = 0.0,
                     pitch_semitones: float = 0.0,
                     pitch_percent: float = 0.0) -> Tuple[str, ProcessingStats]:
        """Process a single audio file."""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if output_path is None:
            output_path = input_path
        
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Processing: {os.path.basename(input_path)}")
        start_time = time.time()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Clean metadata
                result_path, removed_metadata = self.metadata_cleaner.clean_metadata(
                    input_path, None, aggressive
                )
                self.stats.metadata_removed = removed_metadata
                
                # Load audio
                y, sr = librosa.load(result_path, sr=None, mono=False)
                is_stereo = len(y.shape) > 1
                y_mono = np.mean(y, axis=0) if is_stereo else y
                y_mono = safe_cast_to_float64(y_mono)
                original_length = y.shape[1] if is_stereo else len(y)
                
                # Apply tempo and pitch shift
                if tempo_shift != 0 or pitch_semitones != 0 or pitch_percent != 0:
                    logger.info(f"Tempo: {tempo_shift}%, Pitch: {pitch_semitones} semitones")
                    if is_stereo:
                        processed_channels = []
                        for channel in y:
                            channel = safe_cast_to_float64(channel)
                            channel_processed = self._apply_tempo_pitch_shift(
                                channel, sr, tempo_shift, pitch_semitones, pitch_percent
                            )
                            processed_channels.append(safe_cast_to_float64(channel_processed))
                        y = np.array(processed_channels, dtype=np.float64)
                    else:
                        y = safe_cast_to_float64(y)
                        y = self._apply_tempo_pitch_shift(
                            y, sr, tempo_shift, pitch_semitones, pitch_percent
                        )
                        y = safe_cast_to_float64(y)
                
                # Detect watermarks
                watermarks = self._detect_watermarks(y_mono, sr)
                self.stats.watermarks_detected = len(watermarks)
                
                # Remove watermarks
                if watermarks:
                    if is_stereo:
                        processed = y.copy()
                        for i in range(y.shape[0]):
                            channel = safe_cast_to_float64(y[i])
                            channel_result = self.watermark_remover.remove_watermarks_spectral(
                                channel, sr, watermarks
                            )
                            if len(channel_result) != len(y[i]):
                                channel_result = resize_audio_safe(channel_result, len(y[i]))
                            processed[i] = safe_cast_to_float64(channel_result)
                    else:
                        y = safe_cast_to_float64(y)
                        processed = self.watermark_remover.remove_watermarks_spectral(
                            y, sr, watermarks
                        )
                        processed = safe_cast_to_float64(processed)
                        if len(processed) != len(y):
                            processed = resize_audio_safe(processed, len(y))
                    self.stats.watermarks_removed = len(watermarks)
                else:
                    processed = y.copy()
                
                # Normalize patterns
                if self.config.enable_pattern_normalization:
                    patterns = self._detect_patterns(processed)
                    timing_issues = self._detect_timing(processed, sr)
                    self.stats.patterns_normalized = len(patterns) + len(timing_issues)
                    
                    if patterns or timing_issues:
                        if is_stereo:
                            for i in range(processed.shape[0]):
                                channel = safe_cast_to_float64(processed[i])
                                channel_result = self.pattern_normalizer.normalize_patterns(
                                    channel, sr, patterns, timing_issues
                                )
                                if len(channel_result) != len(processed[i]):
                                    channel_result = resize_audio_safe(channel_result, len(processed[i]))
                                processed[i] = safe_cast_to_float64(channel_result)
                        else:
                            processed = safe_cast_to_float64(processed)
                            processed = self.pattern_normalizer.normalize_patterns(
                                processed, sr, patterns, timing_issues
                            )
                            if len(processed) != original_length:
                                processed = resize_audio_safe(processed, original_length)
                
                # Final normalization
                if is_stereo:
                    for i in range(processed.shape[0]):
                        processed[i] = safe_normalize_audio(processed[i])
                        if len(processed[i]) != original_length:
                            processed[i] = resize_audio_safe(processed[i], original_length)
                else:
                    processed = safe_normalize_audio(processed)
                    if len(processed) != original_length:
                        processed = resize_audio_safe(processed, original_length)
                
                # Save
                sf.write(output_path, processed.T if is_stereo else processed, sr)
                
                # Cleanup temp file
                try:
                    if result_path != input_path and os.path.exists(result_path):
                        os.unlink(result_path)
                except:
                    pass
                
            except Exception as e:
                logger.error(f"Processing failed: {e}")
                shutil.copy2(input_path, output_path)
                self.stats.files_failed = 1
                raise
        
        self.stats.files_processed = 1
        self.stats.processing_time = time.time() - start_time
        self.stats.processing_level = self.config.processing_level.value
        
        logger.info(f"Completed in {self.stats.processing_time:.2f}s")
        logger.info(f"Removed {self.stats.watermarks_removed} watermarks")
        
        return output_path, self.stats

    def _apply_tempo_pitch_shift(self, audio: np.ndarray, sr: int,
                                  tempo_shift: float,
                                  pitch_semitones: float,
                                  pitch_percent: float) -> np.ndarray:
        audio = safe_cast_to_float64(audio)
        try:
            if tempo_shift != 0:
                rate = 1.0 + (tempo_shift / 100.0)
                rate = max(0.25, min(4.0, rate))
                audio = librosa.effects.time_stretch(audio, rate=rate)
                audio = safe_cast_to_float64(audio)
            
            if pitch_semitones != 0 or pitch_percent != 0:
                pitch_total = pitch_semitones + (pitch_percent / 6.0)
                pitch_total = max(-24, min(24, pitch_total))
                audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=pitch_total)
                audio = safe_cast_to_float64(audio)
            
            return audio
        except Exception as e:
            logger.warning(f"Tempo/pitch shift failed: {e}")
            return audio

    def _detect_watermarks(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        detected = []
        if len(audio) == 0:
            return detected
        
        audio = safe_cast_to_float64(audio)
        
        try:
            if len(audio) > sr * 300:
                decimation = max(1, len(audio) // (sr * 120))
                audio_analysis = audio[::decimation]
                analysis_sr = sr // decimation
            else:
                audio_analysis = audio
                analysis_sr = sr
            
            audio_analysis = safe_cast_to_float64(audio_analysis)
            nperseg = min(2048, len(audio_analysis) // 8)
            if nperseg < 64:
                nperseg = 64
            
            freqs, times, spec = spectrogram(
                audio_analysis, fs=analysis_sr,
                window='hann', nperseg=nperseg,
                noverlap=nperseg // 2, scaling='spectrum'
            )
            
            if spec.size == 0:
                return detected
            
            spec = safe_cast_to_float64(spec)
            freqs = safe_cast_to_float64(freqs)
            
            for freq_range in POTENTIAL_WATERMARK_FREQS:
                scaled_range = [f * analysis_sr / sr for f in freq_range]
                if scaled_range[1] > analysis_sr / 2:
                    continue
                
                freq_mask = (freqs >= scaled_range[0]) & (freqs <= scaled_range[1])
                if not np.any(freq_mask):
                    continue
                
                band_energy = np.mean(spec[freq_mask], axis=0)
                mean_energy = np.mean(band_energy)
                std_energy = np.std(band_energy)
                
                if std_energy > 0:
                    normalized = (band_energy - mean_energy) / (std_energy + 1e-10)
                    autocorr = np.correlate(normalized, normalized, mode='full')
                    autocorr = autocorr[len(autocorr)//2:]
                    
                    if len(autocorr) > 1:
                        peaks, _ = signal.find_peaks(autocorr, height=0.3, distance=5)
                        if len(peaks) >= 3:
                            detected.append({
                                'type': 'periodic_watermark',
                                'freq_range': freq_range,
                                'peak_count': len(peaks),
                                'confidence': min(1.0, len(peaks) / 10)
                            })
                
                if scaled_range[0] > 15000 * analysis_sr / sr:
                    variation_ratio = std_energy / (mean_energy + 1e-10)
                    if variation_ratio < 0.1:
                        detected.append({
                            'type': 'constant_energy',
                            'freq_range': freq_range,
                            'variation_ratio': float(variation_ratio),
                            'confidence': min(1.0, (0.1 - variation_ratio) * 10)
                        })
            
            detected = [w for w in detected if w.get('confidence', 0) > self.config.watermark_detection_threshold]
            logger.info(f"Detected {len(detected)} spectral watermarks")
            
        except Exception as e:
            logger.error(f"Watermark detection failed: {e}")
        
        return detected

    def _detect_patterns(self, audio: np.ndarray) -> List[Dict[str, Any]]:
        detected = []
        if len(audio) == 0:
            return detected
        
        audio = safe_cast_to_float64(audio)
        if len(audio.shape) > 1:
            audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio
        audio_mono = safe_cast_to_float64(audio_mono)
        
        try:
            stft = np.abs(librosa.stft(audio_mono, n_fft=min(1024, len(audio_mono)//2)))
            if stft.size > 0:
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
        except Exception as e:
            logger.error(f"Pattern detection failed: {e}")
        
        return detected

    def _detect_timing(self, audio: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        detected = []
        if len(audio) == 0:
            return detected
        
        audio = safe_cast_to_float64(audio)
        if len(audio.shape) > 1:
            audio_mono = np.mean(audio, axis=1)
        else:
            audio_mono = audio
        audio_mono = safe_cast_to_float64(audio_mono)
        
        try:
            hop_length = min(512, len(audio_mono) // 100)
            if hop_length < 1:
                hop_length = 1
            
            onset_env = librosa.onset.onset_strength(
                y=audio_mono, sr=sr, hop_length=hop_length
            )
            
            if len(onset_env) > 20:
                onsets = librosa.onset.onset_detect(
                    onset_envelope=onset_env,
                    sr=sr, hop_length=hop_length,
                    units='time'
                )
                
                if len(onsets) > 5:
                    intervals = np.diff(onsets)
                    if len(intervals) > 0:
                        cv = np.std(intervals) / (np.mean(intervals) + 1e-10)
                        if cv < 0.1:
                            detected.append({
                                'type': 'mechanical_timing',
                                'cv': float(cv),
                                'confidence': 1.0 - cv
                            })
        except Exception as e:
            logger.error(f"Timing detection failed: {e}")
        
        return detected

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="AI Audio Fingerprint Remover - Enhanced Edition"
    )
    
    parser.add_argument("input", help="Input audio file to process")
    parser.add_argument("output", nargs="?", help="Output file")
    parser.add_argument("--level", choices=['gentle', 'moderate', 'aggressive', 'extreme'],
                       default='moderate', help="Processing intensity level")
    parser.add_argument("--aggressive", action="store_true", help="Aggressive processing")
    parser.add_argument("--tempo", type=float, default=0.0,
                       help="Tempo change in percent (-50 to 50)")
    parser.add_argument("--pitch-semitones", type=float, default=0.0,
                       help="Pitch shift in semitones (-12 to 12)")
    parser.add_argument("--pitch-percent", type=float, default=0.0,
                       help="Pitch shift in percent (-50 to 100)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    global logger
    logger = setup_logging(args.verbose)
    
    if not check_dependencies():
        return 1
    
    if args.aggressive:
        level = ProcessingLevel.AGGRESSIVE
    else:
        try:
            level = ProcessingLevel(args.level.lower())
        except ValueError:
            level = ProcessingLevel.MODERATE
    
    config = ProcessingConfig.get_profile(level)
    
    print("=" * 60)
    print("AI AUDIO FINGERPRINT REMOVER - ENHANCED EDITION")
    print("=" * 60)
    print(f"Processing level: {level.value}")
    print(f"Tempo: {args.tempo}%")
    print(f"Pitch: {args.pitch_semitones} semitones / {args.pitch_percent}%")
    print("=" * 60)
    
    try:
        processor = EnhancedAudioProcessor(config)
        output_path = args.output or args.input
        result_path, stats = processor.process_file(
            args.input, output_path, args.aggressive,
            tempo_shift=args.tempo,
            pitch_semitones=args.pitch_semitones,
            pitch_percent=args.pitch_percent
        )
        
        print("\n" + "-" * 50)
        print("PROCESSING RESULTS")
        print("-" * 50)
        print(f"Processing level: {stats.processing_level}")
        print(f"Processing time: {stats.processing_time:.2f}s")
        print(f"Watermarks detected: {stats.watermarks_detected}")
        print(f"Watermarks removed: {stats.watermarks_removed}")
        print(f"Patterns normalized: {stats.patterns_normalized}")
        print(f"Output: {result_path}")
        print("-" * 50)
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

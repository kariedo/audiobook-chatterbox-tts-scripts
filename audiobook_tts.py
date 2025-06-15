#!/usr/bin/env python3
"""
Audiobook TTS Generator
Converts text files and ebooks to audiobooks using Chatterbox TTS
Supports chunking, parallel processing, resume capability, voice cloning, MP3 conversion, and time-limited file splitting
"""

import argparse
import logging
import os
import sys
import time
import json
import re
import subprocess
import shutil
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple
import hashlib
import math

# Suppress known warnings from dependencies
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*LoRACompatibleLinear.*")
warnings.filterwarnings("ignore", message=".*torch.backends.cuda.sdp_kernel.*")
warnings.filterwarnings("ignore", message=".*LlamaSdpaAttention.*")
warnings.filterwarnings("ignore", message=".*past_key_values.*")
warnings.filterwarnings("ignore", message=".*attn_implementation.*")
warnings.filterwarnings("ignore", message=".*scaled_dot_product_attention.*")

# Set environment variables to suppress additional warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d

# Import extracted modules
from audio_converter import AudioConverter, AudioSplitter
from text_processor import TextProcessor, TextChunker
from audio_validator import AudioValidator, EndingDetector
from progress_tracker import ProgressTracker

# Setup logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def setup_mac_compatibility():
    """Setup Mac M4 compatibility for Chatterbox"""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    map_location = torch.device(device)
    
    torch_load_original = torch.load
    def patched_torch_load(*args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = map_location
        return torch_load_original(*args, **kwargs)
    
    torch.load = patched_torch_load
    return device

class AudiobookTTS:
    """Main TTS audiobook generator"""
    
    def __init__(self, voice_file: Optional[str] = None, max_workers: int = 2, 
                 exaggeration: float = 0.8, cfg_weight: float = 0.8, pitch_shift: float = 0.0,
                 mp3_bitrate: str = "128k", mp3_enabled: bool = False, remove_wav: bool = False,
                 split_minutes: int = 5, memory_cleanup_interval: int = 5, debug_memory: bool = False,
                 metadata: dict = None, smart_split: bool = True):
        self.device = setup_mac_compatibility()
        logging.info(f"Initializing Chatterbox TTS on {self.device}...")
        
        # Set MPS memory allocation strategy for better memory management
        if self.device == "mps":
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
            # Force immediate cleanup
            torch.mps.empty_cache()
        
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.voice_file = voice_file
        # Reduce default workers for MPS to prevent OOM
        self.max_workers = 1 if self.device == "mps" else max_workers
        
        # Voice characteristics settings
        self.exaggeration = exaggeration
        self.cfg_weight = cfg_weight
        self.pitch_shift = pitch_shift  # Semitones to shift pitch
        
        # MP3 conversion settings
        self.mp3_enabled = mp3_enabled
        self.mp3_bitrate = mp3_bitrate
        self.remove_wav = remove_wav
        
        # File splitting settings
        self.split_minutes = split_minutes
        self.smart_split = smart_split
        
        # MP3 metadata settings
        self.metadata = metadata or {}
        
        # Check FFmpeg availability if MP3 is enabled or splitting is requested
        if self.mp3_enabled or self.split_minutes > 0:
            if not AudioConverter.check_ffmpeg_available():
                if self.mp3_enabled:
                    logging.warning("⚠️ FFmpeg not found - MP3 conversion will be disabled")
                    self.mp3_enabled = False
                if self.split_minutes > 0:
                    logging.warning("⚠️ FFmpeg not found - File splitting will be disabled")
                    self.split_minutes = 0
            else:
                if self.mp3_enabled:
                    logging.info(f"🎵 MP3 conversion enabled (bitrate: {mp3_bitrate})")
                if self.split_minutes > 0:
                    logging.info(f"✂️ File splitting enabled (max {split_minutes} minutes per file)")
        
        # Initialize memory tracking and performance monitoring
        self.chunks_since_cleanup = 0
        self.cleanup_interval = memory_cleanup_interval  # Clear cache every N chunks
        self.performance_history = []  # Track processing times to detect slowdown
        self.adaptive_cleanup = True  # Enable adaptive cleanup based on performance
        self.severe_degradation_count = 0  # Count severe performance issues
        self.model_reinit_threshold = 2  # Reinitialize model after N severe degradations (lowered for faster recovery)
        self.debug_memory = debug_memory  # Enable detailed memory debugging
        
        # Warmup
        with torch.no_grad():
            _ = self.model.generate("Warmup", exaggeration=self.exaggeration, cfg_weight=self.cfg_weight)
        
        # Initial cleanup after warmup
        self._force_memory_cleanup()
        
        # Initialize ending detector for quality control
        self.ending_detector = EndingDetector()
        self.regenerated_chunks_count = 0
        
        logging.info(f"✅ TTS model ready for audiobook generation")
        logging.info(f"   Voice settings: exag={self.exaggeration}, cfg={self.cfg_weight}")
        if self.pitch_shift != 0:
            logging.info(f"   Pitch shift: {self.pitch_shift:+.1f} semitones")
        logging.info(f"   Memory cleanup interval: every {self.cleanup_interval} chunks")
        logging.info(f"   Ending detection enabled for quality control")
    
    def apply_pitch_shift(self, wav: torch.Tensor) -> torch.Tensor:
        """Apply pitch shifting to generated audio"""
        if self.pitch_shift == 0:
            return wav
        
        try:
            # Calculate pitch ratio
            pitch_ratio = 2 ** (self.pitch_shift / 12.0)
            original_sr = self.model.sr
            new_sr = int(original_sr * pitch_ratio)
            
            # Apply pitch shift via resampling
            resampler = ta.transforms.Resample(original_sr, new_sr)
            shifted_wav = resampler(wav)
            
            # Resample back to maintain playback speed
            resampler_back = ta.transforms.Resample(new_sr, original_sr)
            final_wav = resampler_back(shifted_wav)
            
            return final_wav
            
        except Exception as e:
            logging.warning(f"Pitch shift failed: {e}, using original audio")
            return wav
    
    def _get_memory_usage(self) -> dict:
        """Get current memory usage statistics"""
        stats = {}
        
        try:
            # Get basic system memory info
            import psutil
            system_memory = psutil.virtual_memory()
            stats['system_used_mb'] = system_memory.used / 1024 / 1024
            stats['system_available_mb'] = system_memory.available / 1024 / 1024
            stats['system_percent'] = system_memory.percent
            
            # Get current process memory
            process = psutil.Process()
            memory_info = process.memory_info()
            stats['rss_mb'] = memory_info.rss / 1024 / 1024
            stats['vms_mb'] = memory_info.vms / 1024 / 1024
            
        except Exception as e:
            # Fallback to basic resource tracking
            import resource
            try:
                usage = resource.getrusage(resource.RUSAGE_SELF)
                stats['rss_mb'] = usage.ru_maxrss / 1024  # macOS reports in bytes
                stats['system_percent'] = 0.0  # Can't determine without psutil
            except:
                stats['rss_mb'] = 0
                stats['system_percent'] = 0.0
        
        try:
            # Add GPU memory if available (this is most important)
            if self.device == "mps":
                # MPS memory tracking
                if hasattr(torch.mps, 'current_allocated_memory'):
                    stats['gpu_allocated_mb'] = torch.mps.current_allocated_memory() / 1024 / 1024
                else:
                    stats['gpu_allocated_mb'] = 0
                    
                if hasattr(torch.mps, 'max_memory_allocated'):
                    stats['gpu_max_allocated_mb'] = torch.mps.max_memory_allocated() / 1024 / 1024
                else:
                    stats['gpu_max_allocated_mb'] = 0
                    
            elif self.device == "cuda":
                stats['gpu_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
                stats['gpu_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
                stats['gpu_max_allocated_mb'] = torch.cuda.max_memory_allocated() / 1024 / 1024
                stats['gpu_max_reserved_mb'] = torch.cuda.max_memory_reserved() / 1024 / 1024
            else:
                stats['gpu_allocated_mb'] = 0
                
        except Exception as e:
            stats['gpu_allocated_mb'] = 0
            stats['gpu_max_allocated_mb'] = 0
        
        try:
            # Add torch object counts for debugging
            import gc
            torch_objects = [obj for obj in gc.get_objects() if torch.is_tensor(obj)]
            stats['torch_tensor_count'] = len(torch_objects)
            
            # Calculate total tensor memory (simplified)
            tensor_memory_mb = 0
            for obj in torch_objects[:100]:  # Only check first 100 to avoid performance issues
                try:
                    if hasattr(obj, 'numel') and hasattr(obj, 'element_size'):
                        tensor_memory_mb += obj.numel() * obj.element_size() / 1024 / 1024
                except:
                    pass
            stats['torch_tensor_memory_mb'] = tensor_memory_mb
            
        except Exception as e:
            stats['torch_tensor_count'] = 0
            stats['torch_tensor_memory_mb'] = 0
            
        return stats
    
    def _calculate_performance_metrics(self, generation_time: float, text_length: int, word_count: int, text_content: str = "") -> dict:
        """Calculate performance metrics considering text complexity"""
        if generation_time <= 0 or text_length <= 0:
            return {"chars_per_sec": 0, "words_per_min": 0, "expected_time": 0, "performance_ratio": 1.0}
        
        # Calculate actual performance metrics
        chars_per_sec = text_length / generation_time
        words_per_min = (word_count * 60) / generation_time
        
        # Calculate expected generation time based on text complexity
        # Base rate: ~10-15 chars/sec for typical TTS (varies by complexity)
        base_chars_per_sec = 12.0  # Conservative baseline
        
        # Complexity adjustments
        complexity_factor = 1.0
        
        # Longer texts may have slightly lower throughput due to memory usage
        if text_length > 150:
            complexity_factor *= 1.1
        
        # Texts with many punctuation marks are more complex (use text_content if provided, otherwise estimate)
        if text_content:
            punct_count = sum(1 for c in text_content if c in '.,!?;:()[]"\'')
        else:
            # Estimate based on typical punctuation density
            punct_count = int(text_length * 0.08)  # ~8% punctuation in typical text
        
        punct_density = punct_count / max(text_length, 1)
        if punct_density > 0.15:  # High punctuation density
            complexity_factor *= 1.2
        
        expected_chars_per_sec = base_chars_per_sec / complexity_factor
        expected_time = text_length / expected_chars_per_sec
        
        # Performance ratio: how we're doing vs expected (1.0 = exactly as expected, <1.0 = better, >1.0 = worse)
        performance_ratio = generation_time / expected_time if expected_time > 0 else 1.0
        
        return {
            "chars_per_sec": chars_per_sec,
            "words_per_min": words_per_min,
            "expected_time": expected_time,
            "performance_ratio": performance_ratio
        }
    
    def _detect_performance_degradation(self, current_time: float, text_length: int = 100, word_count: int = 20, text_content: str = "") -> bool:
        """Detect if performance is degrading based on text-relative metrics"""
        
        # Calculate performance metrics for this chunk
        perf_metrics = self._calculate_performance_metrics(current_time, text_length, word_count, text_content)
        performance_ratio = perf_metrics["performance_ratio"]
        
        # Store performance ratios instead of absolute times
        if not hasattr(self, 'performance_ratios'):
            self.performance_ratios = []
        
        self.performance_ratios.append(performance_ratio)
        
        # Keep only last 15 measurements for faster detection
        if len(self.performance_ratios) > 15:
            self.performance_ratios = self.performance_ratios[-15:]
        
        # Still track absolute times for compatibility
        self.performance_history.append(current_time)
        if len(self.performance_history) > 15:
            self.performance_history = self.performance_history[-15:]
        
        # Need at least 6 measurements to detect trend
        if len(self.performance_ratios) < 6:
            return False
        
        # Check for immediate severe slowdown (>3x expected time)
        if performance_ratio > 3.0:
            return True
        
        # Compare recent average to earlier average (using performance ratios)
        recent_avg_ratio = sum(self.performance_ratios[-3:]) / 3  # Last 3 chunks
        earlier_avg_ratio = sum(self.performance_ratios[:3]) / 3   # First 3 chunks
        
        # If recent chunks are taking 50% longer relative to text complexity
        degradation_threshold = 1.5
        is_degraded = recent_avg_ratio > earlier_avg_ratio * degradation_threshold
        
        # Also check if we have consistent poor performance (>2x expected)
        recent_slow_count = sum(1 for ratio in self.performance_ratios[-5:] if ratio > 2.0)
        if recent_slow_count >= 3:  # 3 out of last 5 chunks are slow relative to text
            return True
            
        return is_degraded
    
    def _clear_model_caches(self):
        """Clear internal model caches that cause performance degradation"""
        try:
            # Clear T3 model KV-cache if it exists
            if hasattr(self.model, 't3') and hasattr(self.model.t3, 'patched_model'):
                # Clear any cached past_key_values or internal state
                if hasattr(self.model.t3.patched_model, 'clear_cache'):
                    self.model.t3.patched_model.clear_cache()
                
                # Force clear any transformer internal caches
                if hasattr(self.model.t3.patched_model, 'tfmr'):
                    tfmr = self.model.t3.patched_model.tfmr
                    # Clear position embeddings cache if it exists
                    if hasattr(tfmr, '_position_embeddings_cache'):
                        tfmr._position_embeddings_cache.clear()
                    # Clear any attention caches
                    for layer in getattr(tfmr, 'layers', []):
                        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'past_key_value'):
                            layer.self_attn.past_key_value = None
            
            # Clear S3Gen flow cache if it exists
            if hasattr(self.model, 's3gen'):
                s3gen = self.model.s3gen
                # Clear conditional flow matching caches
                if hasattr(s3gen, 'cond_cfm') and hasattr(s3gen.cond_cfm, 'flow_cache'):
                    # Reset flow cache to empty state
                    if hasattr(s3gen.cond_cfm, 'reset_cache'):
                        s3gen.cond_cfm.reset_cache()
                    elif hasattr(s3gen.cond_cfm, 'flow_cache'):
                        # Manually clear the flow cache
                        s3gen.cond_cfm.flow_cache = torch.zeros_like(s3gen.cond_cfm.flow_cache[:, :, :0])
                
                # Clear LRU resampler cache occasionally
                if hasattr(s3gen, 'get_resampler') and hasattr(s3gen.get_resampler, 'cache_clear'):
                    s3gen.get_resampler.cache_clear()
            
            # Clear any other potential caches
            if hasattr(self.model, 'clear_caches'):
                self.model.clear_caches()
                
        except Exception as e:
            logging.warning(f"Could not clear model caches: {e}")
    
    def _force_memory_cleanup(self, aggressive: bool = False):
        """Force aggressive memory cleanup"""
        # Log memory before cleanup
        memory_before = self._get_memory_usage()
        
        # Clear model-specific caches first (most important)
        if aggressive:
            self._clear_model_caches()
        
        if self.device == "mps":
            torch.mps.empty_cache()
            torch.mps.synchronize()
            if aggressive:
                # Multiple rounds of cleanup for severe cases
                for _ in range(3):
                    torch.mps.empty_cache()
                    torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if aggressive:
                for _ in range(3):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        if aggressive:
            # Multiple rounds of garbage collection
            for _ in range(3):
                gc.collect()
        
        # Model warmup after aggressive cleanup to reset internal state
        if aggressive:
            try:
                logging.info("🔄 Performing model warmup after cleanup...")
                with torch.no_grad():
                    # Brief generation to reset model internal state
                    _ = self.model.generate("Reset", exaggeration=self.exaggeration, cfg_weight=self.cfg_weight)
                
                # Clean up warmup artifacts
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                logging.warning(f"Model warmup after cleanup failed: {e}")
        
        # Log memory after cleanup
        memory_after = self._get_memory_usage()
        rss_freed = memory_before.get('rss_mb', 0) - memory_after.get('rss_mb', 0)
        gpu_freed = memory_before.get('gpu_allocated_mb', 0) - memory_after.get('gpu_allocated_mb', 0)
        
        cleanup_type = "aggressive" if aggressive else "standard"
        if rss_freed > 10 or gpu_freed > 10:  # Only log if significant cleanup occurred
            logging.info(f"💾 {cleanup_type.title()} memory cleanup freed: {rss_freed:.1f}MB RAM, {gpu_freed:.1f}MB GPU")
    
    def _reinitialize_model(self):
        """Reinitialize the TTS model as a last resort for severe performance issues"""
        logging.warning("🔄 Reinitializing TTS model due to severe performance degradation...")
        
        # Clear all memory first
        self._force_memory_cleanup(aggressive=True)
        
        try:
            # Reinitialize the model
            del self.model
            if self.device == "mps":
                torch.mps.empty_cache()
                torch.mps.synchronize()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Recreate model
            self.model = ChatterboxTTS.from_pretrained(device=self.device)
            
            # Quick warmup
            with torch.no_grad():
                _ = self.model.generate("Reinit test", exaggeration=self.exaggeration, cfg_weight=self.cfg_weight)
            
            # Reset all counters
            self.chunks_since_cleanup = 0
            self.performance_history = []
            self.severe_degradation_count = 0
            
            logging.info("✅ Model reinitialization completed successfully")
            return True
            
        except Exception as e:
            logging.error(f"❌ Model reinitialization failed: {e}")
            return False
    
    def generate_chunk(self, chunk_text: str, output_file: Path, max_retries: int = 2) -> bool:
        """Generate audio for a single chunk with validation and retry logic"""
        import time
        chunk_start_time = time.time()
        
        for attempt in range(max_retries + 1):
            is_retry = attempt > 0
            if is_retry:
                logging.warning(f"🔄 Retry attempt {attempt}/{max_retries} for chunk generation")
                # Aggressive cleanup before retry
                self._force_memory_cleanup(aggressive=True)
                self.chunks_since_cleanup = 0
                time.sleep(1.0)  # Brief pause before retry
                
            try:
                # More frequent light cleanup every 3 chunks
                if not is_retry:  # Skip on retries since we do aggressive cleanup above
                    self.chunks_since_cleanup += 1
                    
                    # Always do light cleanup before generation
                    if self.device == "mps":
                        torch.mps.empty_cache()
                    elif self.device == "cuda":
                        torch.cuda.empty_cache()
                    
                    # Periodic deep cleanup
                    if self.chunks_since_cleanup >= self.cleanup_interval:
                        logging.info(f"🧹 Performing periodic memory cleanup (every {self.cleanup_interval} chunks)")
                        self._force_memory_cleanup()
                        self.chunks_since_cleanup = 0
                    elif self.chunks_since_cleanup % 2 == 0:
                        # Light cleanup every 2 chunks including model cache clearing (more frequent)
                        self._clear_model_caches()
                        import gc
                        gc.collect()
                
                # Track memory before TTS generation
                memory_before_tts = self._get_memory_usage()
                tts_start_time = time.time()
                
                with torch.no_grad():
                    if self.voice_file and os.path.exists(self.voice_file):
                        wav = self.model.generate(
                            chunk_text,
                            audio_prompt_path=self.voice_file,
                            exaggeration=self.exaggeration,
                            cfg_weight=self.cfg_weight
                        )
                    else:
                        wav = self.model.generate(
                            chunk_text,
                            exaggeration=self.exaggeration,
                            cfg_weight=self.cfg_weight
                        )
                
                tts_duration = time.time() - tts_start_time
                memory_after_tts = self._get_memory_usage()
                
                # Calculate performance metrics and log appropriately
                text_len = len(chunk_text)
                word_count = len(chunk_text.split())
                punct_count = sum(1 for c in chunk_text if c in '.,!?;:')
                complexity_score = (punct_count / max(word_count, 1)) * 100  # Punctuation density as complexity proxy
                
                # Get performance metrics based on text complexity
                perf_metrics = self._calculate_performance_metrics(tts_duration, text_len, word_count, chunk_text)
                performance_ratio = perf_metrics["performance_ratio"]
                chars_per_sec = perf_metrics["chars_per_sec"]
                expected_time = perf_metrics["expected_time"]
                
                # Log based on performance ratio rather than absolute time
                should_log_detailed = (performance_ratio > 1.8 or self.debug_memory)  # 80% slower than expected
                
                if should_log_detailed:
                    gpu_growth = memory_after_tts.get('gpu_allocated_mb', 0) - memory_before_tts.get('gpu_allocated_mb', 0)
                    tensor_growth = memory_after_tts.get('torch_tensor_count', 0) - memory_before_tts.get('torch_tensor_count', 0)
                    
                    if performance_ratio > 2.5:  # >2.5x slower than expected
                        logging.warning(f"🔍 TTS generation slow: {tts_duration:.1f}s (expected {expected_time:.1f}s, {performance_ratio:.1f}x), {chars_per_sec:.1f} chars/s, GPU growth: {gpu_growth:.1f}MB, Text: {text_len}chars/{word_count}words/complexity:{complexity_score:.1f}%")
                    elif performance_ratio > 1.8:  # >1.8x slower than expected
                        logging.info(f"🔍 TTS timing: {tts_duration:.1f}s (expected {expected_time:.1f}s, {performance_ratio:.1f}x), {chars_per_sec:.1f} chars/s, GPU growth: {gpu_growth:.1f}MB, Text: {text_len}chars/{word_count}words")
                    elif self.debug_memory:
                        logging.info(f"🔍 TTS timing: {tts_duration:.1f}s (expected {expected_time:.1f}s, {performance_ratio:.1f}x), {chars_per_sec:.1f} chars/s, GPU growth: {gpu_growth:.1f}MB, Text: {text_len}chars/{word_count}words")
                
                # Apply pitch shift if specified
                wav = self.apply_pitch_shift(wav)
                
                # Save audio
                ta.save(str(output_file), wav, self.model.sr)
                
                # Validate generated audio
                validation_result = AudioValidator.validate_audio_chunk(output_file, chunk_text)
                
                if not validation_result["is_valid"]:
                    logging.warning(f"🔍 Audio validation failed (attempt {attempt + 1}):")
                    for issue in validation_result["issues"]:
                        logging.warning(f"   - {issue}")
                    
                    # Log metrics for debugging
                    metrics = validation_result["metrics"]
                    if metrics:
                        duration = metrics.get("duration", 0)
                        rms = metrics.get("rms_level", 0)
                        words_per_min = metrics.get("words_per_minute", 0)
                        max_amp = metrics.get("max_amplitude", 0)
                        logging.warning(f"   Metrics: {duration:.2f}s, RMS:{rms:.4f}, {words_per_min:.0f}WPM, MaxAmp:{max_amp:.3f}")
                    
                    # Clean up failed audio and try again (unless last attempt)
                    if output_file.exists():
                        output_file.unlink()
                    del wav
                    
                    if attempt < max_retries:
                        logging.info(f"   Retrying with aggressive cleanup...")
                        continue
                    else:
                        logging.error(f"❌ Audio validation failed after {max_retries + 1} attempts")
                        return False
                else:
                    # Validation passed
                    if is_retry:
                        logging.info(f"✅ Audio validation passed on retry attempt {attempt}")
                    elif validation_result["issues"]:
                        # Has minor issues but still valid
                        logging.info(f"⚠️ Audio generated with minor issues:")
                        for issue in validation_result["issues"]:
                            logging.info(f"   - {issue}")
                    
                    # Log detailed metrics if debug enabled or if there were issues
                    if self.debug_memory or validation_result["issues"]:
                        metrics = validation_result["metrics"]
                        duration = metrics.get("duration", 0)
                        rms = metrics.get("rms_level", 0)
                        words_per_min = metrics.get("words_per_minute", 0)
                        max_amp = metrics.get("max_amplitude", 0)
                        volume_var = metrics.get("volume_variance", 0)
                        logging.info(f"🔍 Audio metrics: {duration:.2f}s, RMS:{rms:.4f}, {words_per_min:.0f}WPM, MaxAmp:{max_amp:.3f}, VolVar:{volume_var:.2f}")
                
                # Immediate cleanup
                del wav
                
                # Light cleanup after generation (deep cleanup handled periodically)
                if self.device == "mps":
                    torch.mps.empty_cache()
                elif self.device == "cuda":
                    torch.cuda.empty_cache()
                
                # Performance monitoring and adaptive cleanup (only on successful generation)
                chunk_duration = time.time() - chunk_start_time
                
                # Use text-aware performance detection for final chunk timing
                chunk_perf_metrics = self._calculate_performance_metrics(chunk_duration, text_len, word_count, chunk_text)
                chunk_performance_ratio = chunk_perf_metrics["performance_ratio"]
                
                # Log detailed timing breakdown for debugging (only if significantly slow relative to text)
                if chunk_performance_ratio > 2.0:  # 2x slower than expected for this text length
                    memory_after = self._get_memory_usage()
                    expected_chunk_time = chunk_perf_metrics["expected_time"]
                    logging.warning(f"🐌 Slow chunk generation: {chunk_duration:.1f}s (expected {expected_chunk_time:.1f}s, {chunk_performance_ratio:.1f}x slower) (GPU: {memory_after.get('gpu_allocated_mb', 0):.0f}MB, Tensors: {memory_after.get('torch_tensor_count', 0)})")
                
                if self.adaptive_cleanup and self._detect_performance_degradation(chunk_duration, text_len, word_count, chunk_text):
                    self.severe_degradation_count += 1
                    logging.warning(f"⚠️ Performance degradation detected ({self.severe_degradation_count}/{self.model_reinit_threshold})")
                    
                    if self.severe_degradation_count >= self.model_reinit_threshold:
                        # Try model reinitialization as last resort (lowered threshold for faster recovery)
                        logging.info(f"🔄 Triggering model reinitialization after {self.severe_degradation_count} severe degradations")
                        if self._reinitialize_model():
                            logging.info("✅ Model reinitialized successfully - performance should improve")
                        else:
                            logging.error("❌ Model reinitialization failed, continuing with aggressive cleanup")
                            self._force_memory_cleanup(aggressive=True)
                            self.chunks_since_cleanup = 0
                            self.performance_history = []
                            self.severe_degradation_count = 0
                    else:
                        # Try aggressive cleanup first
                        logging.info("🧹 Triggering aggressive cleanup")
                        self._force_memory_cleanup(aggressive=True)
                        self.chunks_since_cleanup = 0
                        self.performance_history = []
                
                return True
                
            except torch.mps.OutOfMemoryError as e:
                logging.error(f"MPS out of memory generating chunk (attempt {attempt + 1}): {e}")
                # Force aggressive cleanup on OOM and reset counter
                self._force_memory_cleanup(aggressive=True)
                self.chunks_since_cleanup = 0
                self.performance_history = []  # Reset history
                
                if attempt < max_retries:
                    logging.info("Retrying after OOM cleanup...")
                    continue
                else:
                    return False
                    
            except Exception as e:
                logging.error(f"Error generating chunk (attempt {attempt + 1}): {e}")
                if attempt < max_retries:
                    logging.info("Retrying after error...")
                    continue
                else:
                    return False
        
        # Should not reach here
        return False
    
    def process_audiobook(self, input_file: Path, time_limit: Optional[int] = None) -> Path:
        """Process entire audiobook with chunking and parallel processing"""
        
        # Create output directory
        output_dir = Path(input_file.stem)
        output_dir.mkdir(exist_ok=True)
        
        logging.info(f"📚 Processing audiobook: {input_file}")
        logging.info(f"📁 Output directory: {output_dir}")
        
        # Read and chunk text
        logging.info("📖 Reading and chunking text...")
        text_processor = TextProcessor()
        text = text_processor.read_file(input_file)
        
        chunker = TextChunker(max_chars=200)  # Optimal chunk size for speed
        chunks = chunker.chunk_text(text)
        
        logging.info(f"📄 Created {len(chunks)} chunks from {len(text)} characters")
        
        # Setup progress tracking
        progress = ProgressTracker(output_dir)
        progress.progress["total_chunks"] = len(chunks)
        if not progress.progress["start_time"]:
            progress.progress["start_time"] = datetime.now().isoformat()
        progress.save_progress()
        
        # Check for existing chunks
        chunks_to_process = []
        for i, chunk in enumerate(chunks):
            chunk_file = output_dir / f"chunk_{i:04d}.wav"
            
            if progress.is_chunk_completed(i) and chunk_file.exists():
                logging.info(f"⏭️ Skipping chunk {i:04d} (already completed)")
                continue
            
            chunks_to_process.append((i, chunk, chunk_file))
        
        if not chunks_to_process:
            logging.info("🎉 All chunks already completed!")
            return self._combine_chunks(output_dir, len(chunks), input_file.stem)
        
        logging.info(f"🚀 Processing {len(chunks_to_process)} remaining chunks...")
        
        # Set session start time for ETA calculation
        progress.set_session_start()
        
        # Time limit setup
        start_time = datetime.now()
        time_limit_reached = False
        
        # Process chunks with parallel execution
        completed_count = 0
        failed_count = 0
        
        def process_single_chunk(chunk_data):
            """Process a single chunk (for parallel execution)"""
            chunk_index, chunk_text, chunk_file = chunk_data
            
            logging.info(f"🎵 Generating chunk {chunk_index:04d}: {chunk_text[:50]}...")
            
            # Add delay between chunks for memory recovery
            import time
            time.sleep(0.8)  # Increased delay for better memory recovery
            
            # Record processing start time
            chunk_start_time = time.time()
            
            success = self.generate_chunk(chunk_text, chunk_file)
            chunk_duration = time.time() - chunk_start_time
            
            # Get memory stats after chunk generation
            memory_stats = self._get_memory_usage()
            
            if success:
                # Check for problematic endings and re-generate if needed
                detection_result = self.ending_detector.analyze_ending_pattern(chunk_file)
                if detection_result['is_problematic']:
                    logging.warning(f"🔍 Chunk {chunk_index:04d} has problematic ending: {detection_result['reason']}")
                    
                    # Try regeneration up to 4 times
                    max_regeneration_attempts = 4
                    regeneration_successful = False
                    
                    for regen_attempt in range(max_regeneration_attempts):
                        logging.info(f"🔄 Re-generating chunk {chunk_index:04d} (attempt {regen_attempt + 1}/{max_regeneration_attempts}) to fix ending issue...")
                        
                        # Remove the problematic chunk
                        if chunk_file.exists():
                            chunk_file.unlink()
                        
                        # Re-generate the chunk
                        regeneration_success = self.generate_chunk(chunk_text, chunk_file, max_retries=1)
                        if regeneration_success:
                            # Check the re-generated chunk
                            recheck_result = self.ending_detector.analyze_ending_pattern(chunk_file)
                            if not recheck_result['is_problematic']:
                                logging.info(f"✅ Chunk {chunk_index:04d} regeneration successful - ending issue resolved (attempt {regen_attempt + 1})")
                                regeneration_successful = True
                                break
                            else:
                                logging.warning(f"🔄 Chunk {chunk_index:04d} still problematic after attempt {regen_attempt + 1}: {recheck_result['reason']}")
                        else:
                            logging.error(f"❌ Failed to regenerate chunk {chunk_index:04d} on attempt {regen_attempt + 1}")
                    
                    if not regeneration_successful:
                        logging.warning(f"⚠️ Chunk {chunk_index:04d} still problematic after {max_regeneration_attempts} regeneration attempts, keeping anyway")
                    
                    self.regenerated_chunks_count += 1
                    logging.info(f"📊 Total chunks regenerated: {self.regenerated_chunks_count}")
                
                progress.mark_chunk_completed(chunk_index)
                try:
                    audio_length = ta.info(str(chunk_file)).num_frames / ta.info(str(chunk_file)).sample_rate
                    logging.info(f"✅ Completed chunk {chunk_index:04d} ({chunk_duration:.1f}s)")
                except Exception:
                    logging.info(f"✅ Completed chunk {chunk_index:04d} ({chunk_duration:.1f}s)")
                
                # Log detailed memory usage for every chunk if debug mode enabled
                if self.debug_memory:
                    ram_mb = memory_stats.get('rss_mb', 0)
                    gpu_mb = memory_stats.get('gpu_allocated_mb', 0)
                    gpu_max_mb = memory_stats.get('gpu_max_allocated_mb', 0)
                    sys_percent = memory_stats.get('system_percent', 0)
                    tensor_count = memory_stats.get('torch_tensor_count', 0)
                    tensor_memory_mb = memory_stats.get('torch_tensor_memory_mb', 0)
                    
                    logging.info(f"💾 Chunk {chunk_index:04d} Memory: RAM={ram_mb:.0f}MB, GPU={gpu_mb:.0f}MB (max={gpu_max_mb:.0f}MB), SysRAM={sys_percent:.1f}%, Tensors={tensor_count} ({tensor_memory_mb:.0f}MB)")
                else:
                    # Just log basic memory stats
                    ram_mb = memory_stats.get('rss_mb', 0)
                    gpu_mb = memory_stats.get('gpu_allocated_mb', 0)
                    tensor_count = memory_stats.get('torch_tensor_count', 0)
                    logging.info(f"💾 Chunk {chunk_index:04d}: {ram_mb:.0f}MB RAM, {gpu_mb:.0f}MB GPU, {tensor_count} tensors")
            else:
                progress.mark_chunk_failed(chunk_index)
                logging.error(f"❌ Failed chunk {chunk_index:04d}")
            
            return success
        
        # Reduce workers to prevent memory exhaustion
        safe_workers = min(self.max_workers, 1 if self.device == "mps" else 2)
        logging.info(f"🔧 Using {safe_workers} workers for memory safety")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=safe_workers) as executor:
            # Submit all chunks
            future_to_chunk = {
                executor.submit(process_single_chunk, chunk_data): chunk_data[0] 
                for chunk_data in chunks_to_process
            }
            
            # Process completed chunks
            for future in as_completed(future_to_chunk):
                # Check time limit
                if time_limit:
                    elapsed = (datetime.now() - start_time).total_seconds() / 60
                    if elapsed >= time_limit:
                        logging.warning(f"⏰ Time limit reached ({time_limit} minutes)")
                        time_limit_reached = True
                        
                        # Cancel remaining futures
                        for f in future_to_chunk:
                            if not f.done():
                                f.cancel()
                        break
                
                try:
                    success = future.result()
                    if success:
                        completed_count += 1
                    else:
                        failed_count += 1
                    
                    # Progress update with ETA
                    total_completed, total_failed, total_chunks = progress.get_completion_stats()
                    eta_str, chunks_per_sec = progress.calculate_eta()
                    
                    # Calculate percentage
                    percentage = (total_completed / total_chunks * 100) if total_chunks > 0 else 0
                    
                    logging.info(f"📊 Progress: {total_completed}/{total_chunks} ({percentage:.1f}%) completed, {total_failed} failed")
                    logging.info(f"⏱️ ETA: {eta_str} (Rate: {chunks_per_sec:.1f} chunks/sec)")
                    
                    # Log memory usage every 50 chunks to monitor trends
                    if total_completed % 50 == 0 and total_completed > 0:
                        memory_stats = self._get_memory_usage()
                        ram_usage = memory_stats.get('rss_mb', 0)
                        gpu_usage = memory_stats.get('gpu_allocated_mb', 0)
                        logging.info(f"💾 Memory usage: {ram_usage:.0f}MB RAM, {gpu_usage:.0f}MB GPU")
                    
                except Exception as e:
                    logging.error(f"Chunk processing error: {e}")
                    failed_count += 1
        
        # Final statistics
        total_completed, total_failed, total_chunks = progress.get_completion_stats()
        
        logging.info(f"\n🎯 Generation Complete!")
        logging.info(f"   Completed: {total_completed}/{total_chunks}")
        logging.info(f"   Failed: {total_failed}")
        
        if time_limit_reached:
            logging.info(f"   Stopped due to time limit: {time_limit} minutes")
        
        # Combine completed chunks
        if total_completed > 0:
            return self._combine_chunks(output_dir, len(chunks), input_file.stem)
        else:
            logging.error("No chunks were successfully generated!")
            return output_dir
    
    def _combine_chunks(self, output_dir: Path, total_chunks: int, base_filename: str) -> Path:
        """Combine individual chunk files into final audiobook"""
        
        logging.info("🔗 Combining chunks into final audiobook...")
        
        audio_segments = []
        sample_rate = None
        
        for i in range(total_chunks):
            chunk_file = output_dir / f"chunk_{i:04d}.wav"
            
            if chunk_file.exists():
                try:
                    wav, sr = ta.load(str(chunk_file))
                    audio_segments.append(wav)
                    
                    if sample_rate is None:
                        sample_rate = sr
                    
                except Exception as e:
                    logging.warning(f"Could not load chunk {i:04d}: {e}")
        
        if not audio_segments:
            logging.error("No audio segments to combine!")
            return output_dir
        
        # Add small pause between chunks (0.5 seconds)
        pause_samples = int(0.5 * sample_rate)
        pause = torch.zeros(1, pause_samples)
        
        # Combine all segments
        combined_audio = audio_segments[0]
        for segment in audio_segments[1:]:
            combined_audio = torch.cat([combined_audio, pause, segment], dim=1)
        
        # Save final audiobook as WAV
        final_wav_file = output_dir / "audiobook.wav"
        ta.save(str(final_wav_file), combined_audio, sample_rate)
        
        # Calculate statistics
        total_duration = combined_audio.shape[1] / sample_rate
        hours = int(total_duration // 3600)
        minutes = int((total_duration % 3600) // 60)
        seconds = int(total_duration % 60)
        
        logging.info(f"🎉 WAV audiobook complete!")
        logging.info(f"   File: {final_wav_file}")
        logging.info(f"   Duration: {hours:02d}:{minutes:02d}:{seconds:02d}")
        logging.info(f"   Chunks used: {len(audio_segments)}/{total_chunks}")
        
        # Handle file splitting and MP3 conversion
        final_output_files = []
        
        if self.split_minutes > 0:
            # Split the combined audiobook into time-limited files
            logging.info(f"✂️ Splitting audiobook into {self.split_minutes}-minute segments...")
            
            split_files = AudioSplitter.split_audio_by_time(
                input_file=final_wav_file,
                output_base_name=base_filename,
                max_minutes=self.split_minutes,
                mp3_enabled=self.mp3_enabled,
                mp3_bitrate=self.mp3_bitrate,
                remove_wav=self.remove_wav,
                metadata=self.metadata,
                smart_split=self.smart_split
            )
            
            if split_files:
                final_output_files = split_files
                logging.info(f"📱 Split audiobook into {len(split_files)} files")
                
                # Remove the combined file if split was successful and we're keeping MP3 only
                if self.remove_wav and self.mp3_enabled:
                    try:
                        final_wav_file.unlink()
                        logging.info(f"🗑️ Removed combined WAV file: {final_wav_file.name}")
                    except Exception as e:
                        logging.warning(f"Could not remove combined WAV file: {e}")
            else:
                logging.warning("⚠️ File splitting failed, keeping combined file")
                final_output_files = [final_wav_file]
        
        elif self.mp3_enabled:
            # Convert to MP3 without splitting
            final_mp3_file = output_dir / f"{base_filename}.mp3"
            logging.info("🔄 Converting final audiobook to MP3...")
            
            conversion_success = AudioConverter.convert_to_mp3(
                wav_file=final_wav_file,
                mp3_file=final_mp3_file,
                bitrate=self.mp3_bitrate,
                remove_wav=self.remove_wav,
                metadata=self.metadata
            )
            
            if conversion_success:
                final_output_files = [final_mp3_file]
                
                # Get audio info for the MP3
                mp3_info = AudioConverter.get_audio_info(final_mp3_file)
                if mp3_info:
                    logging.info(f"📱 MP3 audiobook details:")
                    logging.info(f"   File: {final_mp3_file}")
                    logging.info(f"   Size: {mp3_info.get('size', 0) // 1024 // 1024}MB")
                    logging.info(f"   Bitrate: {mp3_info.get('bitrate', 0) // 1000}kbps")
                    logging.info(f"   Format: {mp3_info.get('codec', 'mp3')}")
            else:
                final_output_files = [final_wav_file]
        else:
            # Keep WAV file as is
            final_output_files = [final_wav_file]
        
        # Clean up individual chunk files if requested
        if self.remove_wav and (self.mp3_enabled or self.split_minutes > 0):
            self._cleanup_chunk_files(output_dir, total_chunks)
        
        # Log final output files
        if final_output_files:
            logging.info("📋 Final output files:")
            for output_file in final_output_files:
                file_info = AudioConverter.get_audio_info(output_file)
                duration = file_info.get('duration', 0)
                size_mb = output_file.stat().st_size // 1024 // 1024
                logging.info(f"   📄 {output_file.name} - {duration//60:.0f}:{duration%60:02.0f} ({size_mb}MB)")
        
        return output_dir
    
    def _cleanup_chunk_files(self, output_dir: Path, total_chunks: int):
        """Remove individual chunk WAV files after successful conversion"""
        removed_count = 0
        
        logging.info("🧹 Cleaning up individual chunk files...")
        
        for i in range(total_chunks):
            chunk_file = output_dir / f"chunk_{i:04d}.wav"
            
            if chunk_file.exists():
                try:
                    chunk_file.unlink()
                    removed_count += 1
                except Exception as e:
                    logging.warning(f"Could not remove chunk file {chunk_file.name}: {e}")
        
        if removed_count > 0:
            logging.info(f"🗑️ Removed {removed_count} chunk files")

def main():
    parser = argparse.ArgumentParser(
        description="Convert text files and ebooks to audiobooks using Chatterbox TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python audiobook_tts.py book.txt
  python audiobook_tts.py novel.fb2 --voice voices/narrator.wav
  python audiobook_tts.py story.epub --limit-minutes 30 --mp3
  python audiobook_tts.py document.txt --voice voices/reader.wav --mp3 --mp3-bitrate 192k
  python audiobook_tts.py book.txt --mp3 --remove-wav --mp3-bitrate 256k --split-minutes 10
  python audiobook_tts.py novel.txt --split-minutes 3 --mp3 --tag "Author Name - Book Title"
  python audiobook_tts.py book.txt --mp3 --tag "Arthur Conan Doyle - The Adventures of Sherlock Holmes"
        """
    )
    
    parser.add_argument(
        "input_file",
        help="Input text file (supports .txt, .fb2, .epub, .md)"
    )
    
    parser.add_argument(
        "--voice",
        help="Voice reference file for cloning (e.g., voices/myvoice.wav)"
    )
    
    parser.add_argument(
        "--limit-minutes",
        type=int,
        help="Maximum processing time in minutes"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of parallel workers (default: 2)"
    )
    
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.8,
        help="Voice exaggeration level (0.2-1.2, lower=higher pitch, default: 0.8)"
    )
    
    parser.add_argument(
        "--cfg-weight", 
        type=float,
        default=0.8,
        help="CFG weight (0.2-1.0, lower=lighter voice, default: 0.8)"
    )
    
    parser.add_argument(
        "--pitch-shift",
        type=float,
        default=0.0,
        help="Pitch shift in semitones (+/-4, positive=higher, negative=lower)"
    )
    
    # MP3 conversion arguments
    parser.add_argument(
        "--mp3",
        action="store_true",
        help="Convert final audiobook to MP3 format"
    )
    
    parser.add_argument(
        "--mp3-bitrate",
        type=str,
        default="128k",
        choices=["64k", "96k", "128k", "160k", "192k", "256k", "320k"],
        help="MP3 bitrate for conversion (default: 128k)"
    )
    
    parser.add_argument(
        "--remove-wav",
        action="store_true",
        help="Remove WAV files after MP3 conversion (saves disk space)"
    )
    
    # File splitting arguments
    parser.add_argument(
        "--split-minutes",
        type=int,
        default=5,
        help="Split output into files of maximum X minutes each with sentence boundary awareness (default: 5, set to 0 to disable)"
    )
    
    parser.add_argument(
        "--disable-smart-split",
        action="store_true",
        help="Disable sentence-aware splitting and use fixed time intervals (may cut mid-sentence)"
    )
    
    parser.add_argument(
        "--memory-cleanup-interval",
        type=int,
        default=5,
        help="Perform deep memory cleanup every N chunks to prevent slowdown (default: 5, lower=more frequent)"
    )
    
    parser.add_argument(
        "--debug-memory",
        action="store_true",
        help="Enable detailed memory usage logging for every chunk to debug performance issues"
    )
    
    # MP3 metadata arguments
    parser.add_argument(
        "--tag",
        type=str,
        help="MP3 metadata in format 'Author - Book Title' (e.g., 'Arthur Conan Doyle - The Adventures of Sherlock Holmes')"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_file = Path(args.input_file)
    if not input_file.exists():
        logging.error(f"Input file not found: {input_file}")
        sys.exit(1)
    
    # Validate voice file if provided
    if args.voice and not os.path.exists(args.voice):
        logging.error(f"Voice file not found: {args.voice}")
        sys.exit(1)
    
    # Validate MP3 settings
    if args.remove_wav and not args.mp3 and args.split_minutes == 0:
        logging.warning("--remove-wav specified without --mp3 or --split-minutes, ignoring...")
        args.remove_wav = False
    
    # Validate split settings
    if args.split_minutes < 0:
        logging.error("--split-minutes must be 0 or positive")
        sys.exit(1)
    
    # Parse metadata from tag if provided
    metadata = {}
    if args.tag:
        # Parse format: "Author - Book Title"
        if ' - ' in args.tag:
            author, title = args.tag.split(' - ', 1)
            metadata['artist'] = author.strip()
            metadata['album'] = title.strip()
            metadata['title'] = title.strip()
            metadata['genre'] = 'Audiobook'
        else:
            # If no separator, treat entire string as title
            metadata['album'] = args.tag.strip()
            metadata['title'] = args.tag.strip()
            metadata['genre'] = 'Audiobook'
    
    # Log configuration
    logging.info("🎙️ Audiobook TTS Generator Starting...")
    logging.info(f"   Input: {input_file}")
    logging.info(f"   Voice: {args.voice or 'Default model voice'}")
    logging.info(f"   Voice settings: exag={args.exaggeration}, cfg={args.cfg_weight}")
    logging.info(f"   Pitch shift: {args.pitch_shift:+.1f} semitones")
    logging.info(f"   Time limit: {args.limit_minutes or 'None'} minutes")
    logging.info(f"   Workers: {args.workers}")
    
    if args.split_minutes > 0:
        split_mode = "fixed intervals" if args.disable_smart_split else "sentence-aware"
        logging.info(f"   File splitting: enabled (max {args.split_minutes} minutes per file, {split_mode})")
        output_format = "MP3" if args.mp3 else "WAV"
        logging.info(f"   Output format: {output_format}")
        if args.mp3:
            logging.info(f"   MP3 bitrate: {args.mp3_bitrate}")
    elif args.mp3:
        logging.info(f"   MP3 conversion: enabled (bitrate: {args.mp3_bitrate})")
        logging.info(f"   File splitting: disabled")
    else:
        logging.info("   MP3 conversion: disabled")
        logging.info("   File splitting: disabled")
    
    if args.remove_wav:
        logging.info(f"   Remove WAV files: {args.remove_wav}")
    
    if metadata:
        logging.info("🏷️ MP3 Metadata:")
        if metadata.get('artist'):
            logging.info(f"   Author: {metadata['artist']}")
        if metadata.get('album'):
            logging.info(f"   Book: {metadata['album']}")
    
    try:
        # Initialize TTS generator
        tts_generator = AudiobookTTS(
            voice_file=args.voice,
            max_workers=args.workers,
            exaggeration=args.exaggeration,
            cfg_weight=args.cfg_weight,
            pitch_shift=args.pitch_shift,
            mp3_enabled=args.mp3,
            mp3_bitrate=args.mp3_bitrate,
            remove_wav=args.remove_wav,
            split_minutes=args.split_minutes,
            memory_cleanup_interval=args.memory_cleanup_interval,
            debug_memory=args.debug_memory,
            metadata=metadata,
            smart_split=not args.disable_smart_split
        )
        
        # Process audiobook
        output_dir = tts_generator.process_audiobook(
            input_file=input_file,
            time_limit=args.limit_minutes
        )
        
        logging.info(f"🎉 Audiobook generation complete! Check: {output_dir}")
        
        # Show regenerated chunks statistics
        if tts_generator.regenerated_chunks_count > 0:
            logging.info(f"🔄 Quality control: {tts_generator.regenerated_chunks_count} chunks were regenerated due to ending issues")
        else:
            logging.info(f"✅ Quality control: All chunks passed ending detection")
        
        # List final output files
        final_files = []
        
        # Look for split files or single files
        if args.split_minutes > 0:
            # Look for split files with the new naming convention
            base_name = input_file.stem
            extension = ".mp3" if args.mp3 else ".wav"
            
            split_files = sorted(output_dir.glob(f"{base_name}_*.{extension.lstrip('.')}"))
            for split_file in split_files:
                final_files.append(f"🎵 {split_file.name}")
        else:
            # Look for single combined files
            wav_file = output_dir / "audiobook.wav"
            mp3_file = output_dir / f"{input_file.stem}.mp3"
            
            if mp3_file.exists():
                final_files.append(f"🎵 {mp3_file.name}")
            elif wav_file.exists():
                final_files.append(f"📄 {wav_file.name}")
        
        if final_files:
            logging.info("📋 Final output files:")
            for file_info in final_files:
                logging.info(f"   {file_info}")
        
    except KeyboardInterrupt:
        logging.info("⏹️ Generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"❌ Error during generation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
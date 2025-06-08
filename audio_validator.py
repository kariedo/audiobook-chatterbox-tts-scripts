#!/usr/bin/env python3
"""
Audio Validator Module
Contains AudioValidator and EndingDetector classes for validating TTS-generated audio chunks
"""

import logging
from pathlib import Path
from typing import Tuple
import torch
import torchaudio as ta
import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d


class AudioValidator:
    """Validates generated audio chunks for quality issues"""
    
    @staticmethod
    def validate_audio_chunk(audio_file: Path, chunk_text: str, expected_duration_range: Tuple[float, float] = (0.5, 30.0)) -> dict:
        """
        Validate an audio chunk for common issues
        
        Returns dict with validation results:
        - is_valid: bool
        - issues: list of detected problems
        - metrics: dict with audio metrics
        """
        validation_result = {
            "is_valid": True,
            "issues": [],
            "metrics": {}
        }
        
        try:
            if not audio_file.exists():
                validation_result["is_valid"] = False
                validation_result["issues"].append("File does not exist")
                return validation_result
            
            # Check file size
            file_size = audio_file.stat().st_size
            if file_size < 1024:  # Less than 1KB
                validation_result["is_valid"] = False
                validation_result["issues"].append(f"File too small: {file_size} bytes")
                return validation_result
            
            # Load and analyze audio
            try:
                waveform, sample_rate = ta.load(str(audio_file))
                validation_result["metrics"]["sample_rate"] = sample_rate
                validation_result["metrics"]["channels"] = waveform.shape[0]
                validation_result["metrics"]["samples"] = waveform.shape[1]
                
                # Calculate duration
                duration = waveform.shape[1] / sample_rate
                validation_result["metrics"]["duration"] = duration
                
                # Check duration is reasonable
                if duration < expected_duration_range[0]:
                    validation_result["issues"].append(f"Audio too short: {duration:.2f}s (expected min {expected_duration_range[0]}s)")
                elif duration > expected_duration_range[1]:
                    validation_result["issues"].append(f"Audio too long: {duration:.2f}s (expected max {expected_duration_range[1]}s)")
                
                # Check for silence (all samples near zero)
                audio_rms = torch.sqrt(torch.mean(waveform ** 2))
                validation_result["metrics"]["rms_level"] = float(audio_rms)
                
                if audio_rms < 0.001:  # Very quiet audio
                    validation_result["issues"].append(f"Audio appears silent: RMS {audio_rms:.6f}")
                
                # Check for clipping (samples at max/min values)
                max_amplitude = torch.max(torch.abs(waveform))
                validation_result["metrics"]["max_amplitude"] = float(max_amplitude)
                
                if max_amplitude > 0.95:  # Near clipping
                    validation_result["issues"].append(f"Audio may be clipped: max amplitude {max_amplitude:.3f}")
                
                # Check for DC offset
                dc_offset = torch.mean(waveform)
                validation_result["metrics"]["dc_offset"] = float(dc_offset)
                
                if abs(dc_offset) > 0.1:
                    validation_result["issues"].append(f"Significant DC offset: {dc_offset:.3f}")
                
                # Check for excessive silence at start/end
                silence_threshold = 0.01
                start_silence, end_silence = AudioValidator._detect_silence_boundaries(waveform, silence_threshold)
                validation_result["metrics"]["start_silence"] = start_silence / sample_rate
                validation_result["metrics"]["end_silence"] = end_silence / sample_rate
                
                # Warn if more than 1 second of silence at start/end
                if start_silence / sample_rate > 1.0:
                    validation_result["issues"].append(f"Excessive silence at start: {start_silence/sample_rate:.2f}s")
                if end_silence / sample_rate > 1.0:
                    validation_result["issues"].append(f"Excessive silence at end: {end_silence/sample_rate:.2f}s")
                
                # Check for reasonable correlation with text length
                text_length = len(chunk_text.strip())
                words = len(chunk_text.split())
                chars_per_second = text_length / duration if duration > 0 else 0
                words_per_minute = (words * 60) / duration if duration > 0 else 0
                
                validation_result["metrics"]["text_length"] = text_length
                validation_result["metrics"]["word_count"] = words
                validation_result["metrics"]["chars_per_second"] = chars_per_second
                validation_result["metrics"]["words_per_minute"] = words_per_minute
                
                # Typical reading speeds: 150-250 WPM, 8-15 chars/second
                if words_per_minute > 300:
                    validation_result["issues"].append(f"Speech too fast: {words_per_minute:.0f} WPM (expected < 300)")
                elif words_per_minute < 50 and words > 5:  # Only flag if substantial text
                    validation_result["issues"].append(f"Speech too slow: {words_per_minute:.0f} WPM (expected > 50)")
                
                # Check for audio artifacts (sudden volume spikes)
                if len(waveform[0]) > sample_rate:  # Only for audio longer than 1 second
                    volume_variance = AudioValidator._detect_volume_spikes(waveform)
                    validation_result["metrics"]["volume_variance"] = volume_variance
                    
                    if volume_variance > 10.0:  # High variance indicates possible artifacts
                        validation_result["issues"].append(f"High volume variance detected: {volume_variance:.2f} (possible artifacts)")
                
            except Exception as e:
                validation_result["is_valid"] = False
                validation_result["issues"].append(f"Audio loading error: {str(e)}")
                return validation_result
            
            # Set overall validity
            if validation_result["issues"]:
                validation_result["is_valid"] = False
                
        except Exception as e:
            validation_result["is_valid"] = False
            validation_result["issues"].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    @staticmethod
    def _detect_silence_boundaries(waveform: torch.Tensor, threshold: float = 0.01) -> Tuple[int, int]:
        """Detect silence at the start and end of audio"""
        audio = waveform[0] if waveform.dim() > 1 else waveform
        abs_audio = torch.abs(audio)
        
        # Find start of audio (first non-silent sample)
        start_silence = 0
        for i, sample in enumerate(abs_audio):
            if sample > threshold:
                start_silence = i
                break
        else:
            start_silence = len(abs_audio)  # All silence
        
        # Find end of audio (last non-silent sample)
        end_silence = 0
        for i, sample in enumerate(reversed(abs_audio)):
            if sample > threshold:
                end_silence = i
                break
        else:
            end_silence = len(abs_audio)  # All silence
        
        return start_silence, end_silence
    
    @staticmethod
    def _detect_volume_spikes(waveform: torch.Tensor, window_size: int = 2048) -> float:
        """Detect sudden volume changes that might indicate artifacts"""
        audio = waveform[0] if waveform.dim() > 1 else waveform
        
        # Calculate RMS for overlapping windows
        rms_values = []
        for i in range(0, len(audio) - window_size, window_size // 2):
            window = audio[i:i + window_size]
            rms = torch.sqrt(torch.mean(window ** 2))
            rms_values.append(float(rms))
        
        if len(rms_values) < 2:
            return 0.0
        
        # Calculate variance in RMS levels (higher = more spiky)
        rms_tensor = torch.tensor(rms_values)
        variance = torch.var(rms_tensor)
        
        return float(variance)


class EndingDetector:
    """Detects problematic TTS endings where RMS energy rises and stays elevated"""
    
    def __init__(self):
        self.sample_rate = 24000
        
    def analyze_ending_pattern(self, audio_file: Path, analysis_duration: float = 5.0) -> dict:
        """Analyze the ending pattern of an audio file"""
        try:
            audio, sr = librosa.load(str(audio_file), sr=self.sample_rate)
            duration = len(audio) / sr
            
            if duration < analysis_duration:
                return {'is_problematic': False, 'reason': f'Audio too short ({duration:.2f}s)'}
                
            # Extract the last N seconds
            start_sample = len(audio) - int(analysis_duration * sr)
            ending_segment = audio[start_sample:]
            
            # Calculate RMS energy in small windows
            window_size = int(sr * 0.1)  # 100ms windows
            hop_size = int(sr * 0.05)    # 50ms hop
            
            rms_values = []
            for i in range(0, len(ending_segment) - window_size, hop_size):
                window = ending_segment[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                rms_values.append(rms)
            
            rms_values = np.array(rms_values)
            if len(rms_values) < 10:
                return {'is_problematic': False, 'reason': 'Insufficient data for analysis'}
                
            # Smooth the RMS values
            rms_smooth = gaussian_filter1d(rms_values, sigma=2)
            
            # Find the minimum RMS value and its position
            min_idx = np.argmin(rms_smooth)
            min_rms = rms_smooth[min_idx]
            
            # Analyze what happens after the minimum
            remaining_frames = len(rms_smooth) - min_idx
            if remaining_frames < 8:
                return {'is_problematic': False, 'reason': 'Not enough data after minimum'}
                
            post_min_values = rms_smooth[min_idx:]
            
            # Focus on final 25% for true ending trend
            final_25_pct = int(len(post_min_values) * 0.75)
            if final_25_pct < len(post_min_values) and len(post_min_values) - final_25_pct >= 3:
                final_values = post_min_values[final_25_pct:]
                final_indices = np.arange(len(final_values))
                final_slope = np.polyfit(final_indices, final_values, 1)[0]
            else:
                final_slope = 0
                
            # Check if the very end (last 10%) is higher than mid-point
            last_10_pct_idx = int(len(post_min_values) * 0.9)
            if last_10_pct_idx < len(post_min_values):
                mid_point_rms = np.mean(post_min_values[len(post_min_values)//2:int(len(post_min_values)*0.8)])
                end_10_pct_rms = np.mean(post_min_values[last_10_pct_idx:])
                end_vs_mid_ratio = end_10_pct_rms / (mid_point_rms + 1e-10)
            else:
                end_vs_mid_ratio = 1.0
                
            # Compare first quarter vs last quarter after minimum
            quarter_len = len(post_min_values) // 4
            if quarter_len > 1:
                first_quarter = np.mean(post_min_values[:quarter_len])
                last_quarter = np.mean(post_min_values[-quarter_len:])
                quarter_ratio = last_quarter / (first_quarter + 1e-10)
            else:
                quarter_ratio = 1.0
                
            # Look at the final value compared to minimum
            end_rms = rms_smooth[-1]
            end_recovery_ratio = end_rms / (min_rms + 1e-10)
            
            # Check if energy stays above 2x minimum for substantial time
            threshold_2x = min_rms * 2
            above_2x_frames = np.sum(post_min_values > threshold_2x)
            above_2x_ratio = above_2x_frames / len(post_min_values)
            
            # DETECTION LOGIC: Final portion has upward trend AND end stays elevated
            is_problematic = False
            reason = 'normal_fade'
            
            # Primary indicator: Final portion has upward trend AND end stays elevated
            if (final_slope > min_rms * 0.1 and end_vs_mid_ratio > 2.0):
                if (end_recovery_ratio > 5.0 and quarter_ratio > 3.0):
                    is_problematic = True
                    reason = f'Rising trend detected: final_slope={final_slope:.6f}, end/mid_ratio={end_vs_mid_ratio:.2f}, recovery={end_recovery_ratio:.2f}x'
            
            # Secondary indicator: Strong end recovery with sustained elevation
            elif (end_recovery_ratio > 5.0 and quarter_ratio > 2.5 and 
                  end_vs_mid_ratio > 1.5 and final_slope > 0 and above_2x_ratio > 0.6):
                is_problematic = True
                reason = f'Sustained elevation detected: recovery={end_recovery_ratio:.2f}x, quarter_ratio={quarter_ratio:.2f}, above_2x={above_2x_ratio:.1%}'
            
            return {
                'is_problematic': is_problematic,
                'reason': reason,
                'final_slope': final_slope,
                'end_vs_mid_ratio': end_vs_mid_ratio,
                'end_recovery_ratio': end_recovery_ratio,
                'quarter_ratio': quarter_ratio,
                'above_2x_ratio': above_2x_ratio
            }
            
        except Exception as e:
            return {'is_problematic': False, 'reason': f'Analysis error: {e}'}
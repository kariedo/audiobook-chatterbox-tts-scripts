#!/usr/bin/env python3
"""
Final Ending Detector

Precisely detects problematic TTS endings where the RMS energy
RISES and STAYS ELEVATED after dropping, not just brief spikes.
"""

import librosa
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

class FinalEndingDetector:
    def __init__(self):
        self.sample_rate = 24000
        
    def load_audio(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
    
    def analyze_ending_pattern(self, audio, sr, analysis_duration=5.0):
        duration = len(audio) / sr
        
        if duration < analysis_duration:
            print(f"Audio too short (need at least {analysis_duration}s, got {duration:.2f}s)")
            return None
        
        # Extract the last N seconds
        start_sample = len(audio) - int(analysis_duration * sr)
        ending_segment = audio[start_sample:]
        
        # Calculate RMS energy in small windows
        window_size = int(sr * 0.1)  # 100ms windows
        hop_size = int(sr * 0.05)    # 50ms hop
        
        rms_values = []
        rms_times = []
        
        for i in range(0, len(ending_segment) - window_size, hop_size):
            window = ending_segment[i:i + window_size]
            rms = np.sqrt(np.mean(window ** 2))
            rms_values.append(rms)
            
            time_pos = duration - analysis_duration + (i + window_size // 2) / sr
            rms_times.append(time_pos)
        
        rms_values = np.array(rms_values)
        rms_times = np.array(rms_times)
        
        # Smooth the RMS values
        rms_smooth = gaussian_filter1d(rms_values, sigma=2)
        
        # Find the minimum RMS value and its position
        min_idx = np.argmin(rms_smooth)
        min_rms = rms_smooth[min_idx]
        min_time = rms_times[min_idx]
        
        # Analyze what happens after the minimum
        remaining_frames = len(rms_smooth) - min_idx
        
        if remaining_frames < 8:  # Need substantial data after minimum
            pattern_analysis = {
                'pattern_type': 'insufficient_data',
                'description': 'Not enough data after minimum to analyze pattern',
                'is_problematic': False
            }
        else:
            post_min_values = rms_smooth[min_idx:]
            post_min_times = rms_times[min_idx:]
            
            # Key insight: Look at the OVERALL TREND after minimum, not just peaks
            
            # Method 1: Linear regression on post-minimum data
            time_indices = np.arange(len(post_min_values))
            overall_slope = np.polyfit(time_indices, post_min_values, 1)[0]
            
            # Method 1b: Focus on final 25% for true ending trend (more restrictive)
            final_25_pct = int(len(post_min_values) * 0.75)
            if final_25_pct < len(post_min_values) and len(post_min_values) - final_25_pct >= 3:
                final_values = post_min_values[final_25_pct:]
                final_indices = np.arange(len(final_values))
                final_slope = np.polyfit(final_indices, final_values, 1)[0]
            else:
                final_slope = overall_slope
                
            # Method 1c: Check if the very end (last 10%) is higher than mid-point
            last_10_pct_idx = int(len(post_min_values) * 0.9)
            if last_10_pct_idx < len(post_min_values):
                mid_point_rms = np.mean(post_min_values[len(post_min_values)//2:int(len(post_min_values)*0.8)])
                end_10_pct_rms = np.mean(post_min_values[last_10_pct_idx:])
                end_vs_mid_ratio = end_10_pct_rms / (mid_point_rms + 1e-10)
            else:
                end_vs_mid_ratio = 1.0
            
            # Method 2: Compare first quarter vs last quarter after minimum
            quarter_len = len(post_min_values) // 4
            if quarter_len > 1:
                first_quarter = np.mean(post_min_values[:quarter_len])
                last_quarter = np.mean(post_min_values[-quarter_len:])
                quarter_ratio = last_quarter / (first_quarter + 1e-10)
            else:
                quarter_ratio = 1.0
            
            # Method 3: Look at the final value compared to minimum
            end_rms = rms_smooth[-1]
            end_recovery_ratio = end_rms / (min_rms + 1e-10)
            
            # Method 4: Check if energy stays above 2x minimum for substantial time
            threshold_2x = min_rms * 2
            above_2x_frames = np.sum(post_min_values > threshold_2x)
            above_2x_ratio = above_2x_frames / len(post_min_values)
            
            # Method 5: Find the peak and see what happens after it
            max_after_min = np.max(post_min_values)
            max_idx_relative = np.argmax(post_min_values)
            rise_ratio = max_after_min / (min_rms + 1e-10)
            
            # Analyze trend AFTER the peak (this is crucial)
            if max_idx_relative < len(post_min_values) - 3:
                post_peak_values = post_min_values[max_idx_relative:]
                post_peak_slope = np.polyfit(range(len(post_peak_values)), post_peak_values, 1)[0]
                post_peak_trend = 'rising' if post_peak_slope > min_rms * 0.01 else 'falling'
            else:
                post_peak_slope = 0
                post_peak_trend = 'unknown'
            
            # DECISIVE DETECTION LOGIC
            # True problem: After dropping, RMS RISES and STAYS ELEVATED (upward trend)
            
            is_problematic = False
            pattern_type = 'normal_fade'
            confidence = 0.0
            
            # Primary indicator: Final portion has upward trend AND end stays elevated
            if (final_slope > min_rms * 0.1 and   # Significant final upward slope
                end_vs_mid_ratio > 2.0):          # End is notably higher than middle section
                # Additional confirmations needed
                if (end_recovery_ratio > 5.0 and  # End is significantly higher than minimum
                    quarter_ratio > 3.0):         # Last quarter notably higher than first quarter
                    
                    pattern_type = 'problematic_rising_trend'
                    confidence = 0.9
                    is_problematic = True
                    
            # Secondary indicator: Strong end recovery with sustained elevation
            elif (end_recovery_ratio > 5.0 and  # Very high end recovery
                  quarter_ratio > 2.5 and       # Strong quarter-to-quarter increase
                  end_vs_mid_ratio > 1.5 and    # End significantly higher than middle
                  final_slope > 0 and           # Final trend is positive
                  above_2x_ratio > 0.6):        # Sustained elevation
                
                pattern_type = 'problematic_sustained_elevation'
                confidence = 0.8
                is_problematic = True
                
            else:
                # Classify the normal patterns for debugging
                if rise_ratio > 3.0:
                    if post_peak_trend == 'falling' and end_recovery_ratio < 2.0:
                        pattern_type = 'normal_spike_then_fade'
                    elif overall_slope < -min_rms * 0.02:
                        pattern_type = 'normal_declining_trend'
                    else:
                        pattern_type = 'normal_mixed_pattern'
                else:
                    pattern_type = 'normal_fade'
                
                confidence = 0.0
                is_problematic = False
            
            description = f'Min at {min_time:.2f}s | Final slope: {final_slope:.6f} | End/mid: {end_vs_mid_ratio:.2f} | End recovery: {end_recovery_ratio:.2f}x'
            
            pattern_analysis = {
                'pattern_type': pattern_type,
                'description': description,
                'is_problematic': is_problematic,
                'confidence': confidence,
                'min_time': min_time,
                'min_rms': min_rms,
                'max_after_min': max_after_min,
                'rise_ratio': rise_ratio,
                'end_rms': end_rms,
                'end_recovery_ratio': end_recovery_ratio,
                'overall_slope': overall_slope,
                'final_slope': final_slope,
                'end_vs_mid_ratio': end_vs_mid_ratio,
                'quarter_ratio': quarter_ratio,
                'above_2x_ratio': above_2x_ratio,
                'post_peak_slope': post_peak_slope,
                'post_peak_trend': post_peak_trend
            }
        
        results = {
            'file': '',
            'duration': duration,
            'analysis_region': f'Last {analysis_duration:.1f}s ({duration-analysis_duration:.1f}s - {duration:.1f}s)',
            'rms_values': rms_values,
            'rms_times': rms_times,
            'rms_smooth': rms_smooth,
            'pattern_analysis': pattern_analysis,
            'overall_problematic': pattern_analysis.get('is_problematic', False)
        }
        
        return results
    
    def analyze_file(self, file_path, analysis_duration=5.0, visualize=False):
        print(f"\\nAnalyzing: {file_path}")
        
        audio, sr = self.load_audio(file_path)
        if audio is None:
            return None
        
        results = self.analyze_ending_pattern(audio, sr, analysis_duration)
        if results is None:
            return None
        
        results['file'] = str(file_path)
        
        # Print results
        pattern = results['pattern_analysis']
        print(f"Duration: {results['duration']:.2f}s | Region: {results['analysis_region']}")
        print(f"Pattern: {pattern['pattern_type']}")
        print(f"Description: {pattern['description']}")
        
        if 'overall_slope' in pattern:
            print(f"Final slope: {pattern['final_slope']:.6f} | Overall slope: {pattern['overall_slope']:.6f} | Quarter ratio: {pattern['quarter_ratio']:.2f} | Above 2x: {pattern['above_2x_ratio']:.1%}")
            if 'confidence' in pattern:
                print(f"Confidence: {pattern['confidence']:.1f}")
        
        if results['overall_problematic']:
            print("🚨 PROBLEMATIC ENDING DETECTED")
        else:
            print("✅ Normal ending pattern")
        
        if visualize:
            self.create_visualization(results, file_path)
        
        return results
    
    def create_visualization(self, results, file_path):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
        
        rms_times = results['rms_times']
        rms_values = results['rms_values']
        rms_smooth = results['rms_smooth']
        pattern = results['pattern_analysis']
        
        # Plot 1: Full RMS analysis
        ax1.plot(rms_times, rms_values, 'b-', alpha=0.4, label='Raw RMS')
        ax1.plot(rms_times, rms_smooth, 'r-', linewidth=2, label='Smoothed RMS')
        
        if 'min_time' in pattern:
            ax1.axvline(x=pattern['min_time'], color='green', linestyle='--', alpha=0.8, label='Minimum')
            ax1.plot(pattern['min_time'], pattern['min_rms'], 'go', markersize=8)
            
            # Mark 2x minimum threshold
            ax1.axhline(y=pattern['min_rms'] * 2, color='orange', linestyle=':', alpha=0.6, label='2x minimum')
        
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('RMS Energy')
        ax1.set_title(f'RMS Analysis - {Path(file_path).name} - {pattern["pattern_type"]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Post-minimum trend analysis
        if 'min_time' in pattern and 'overall_slope' in pattern:
            min_idx = np.argmin(rms_smooth)
            post_min_times = rms_times[min_idx:]
            post_min_values = rms_smooth[min_idx:]
            
            ax2.plot(post_min_times, post_min_values, 'r-', linewidth=3, label='Post-minimum RMS')
            
            # Add trend line
            if len(post_min_values) > 1:
                trend_line = pattern['min_rms'] + pattern['overall_slope'] * np.arange(len(post_min_values))
                ax2.plot(post_min_times, trend_line, 'k--', linewidth=2, 
                        label=f'Overall trend (slope: {pattern["overall_slope"]:.6f})')
            
            # Mark quarters
            quarter_len = len(post_min_values) // 4
            if quarter_len > 1:
                q1_end = min_idx + quarter_len
                q4_start = len(rms_times) - quarter_len
                
                ax2.axvspan(post_min_times[0], rms_times[q1_end], alpha=0.2, color='blue', label='First quarter')
                ax2.axvspan(rms_times[q4_start], rms_times[-1], alpha=0.2, color='red', label='Last quarter')
            
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('RMS Energy')
            ax2.set_title('Post-Minimum Trend Analysis')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Detection metrics
        if 'overall_slope' in pattern:
            metrics = ['Rise Ratio', 'End Recovery', 'Quarter Ratio', 'Above 2x %', 'Overall Slope*1000']
            values = [
                pattern['rise_ratio'],
                pattern['end_recovery_ratio'], 
                pattern['quarter_ratio'],
                pattern['above_2x_ratio'] * 100,
                pattern['overall_slope'] * 1000  # Scale for visibility
            ]
            
            colors = ['red' if results['overall_problematic'] else 'green'] * len(metrics)
            
            bars = ax3.bar(metrics, values, color=colors, alpha=0.7)
            ax3.set_ylabel('Value')
            ax3.set_title('Detection Metrics')
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                        f'{value:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        output_name = Path(file_path).stem + '_final_analysis.png'
        plt.savefig(output_name, dpi=150, bbox_inches='tight')
        print(f"Visualization saved as: {output_name}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Final Ending Pattern Detector')
    parser.add_argument('file_path', nargs='?', help='Path to WAV file to analyze')
    parser.add_argument('--duration', type=float, default=5.0, help='Duration to analyze from end')
    parser.add_argument('--visualize', action='store_true', help='Create visualization')
    parser.add_argument('--batch', help='Process all WAV files in directory')
    
    args = parser.parse_args()
    
    if not args.batch and not args.file_path:
        parser.error("Either file_path or --batch is required")
    if args.batch and args.file_path:
        parser.error("Cannot use both file_path and --batch")
    
    detector = FinalEndingDetector()
    
    if args.batch:
        directory = Path(args.batch)
        if not directory.is_dir():
            print(f"Error: {args.batch} is not a valid directory")
            sys.exit(1)
        
        wav_files = list(directory.glob('*.wav'))
        if not wav_files:
            print(f"No WAV files found in {args.batch}")
            sys.exit(1)
        
        print(f"Processing {len(wav_files)} WAV files...")
        problematic_files = []
        
        for wav_file in sorted(wav_files):
            results = detector.analyze_file(wav_file, args.duration, visualize=False)
            if results and results['overall_problematic']:
                pattern = results['pattern_analysis']
                problematic_files.append((wav_file, pattern['pattern_type'], pattern.get('confidence', 0)))
        
        print(f"\\n--- FINAL BATCH RESULTS ---")
        print(f"Analyzed {len(wav_files)} files")
        print(f"Found {len(problematic_files)} problematic files")
        
        if problematic_files:
            print("\\nProblematic files:")
            for file_path, pattern_type, confidence in problematic_files:
                print(f"  {file_path.name}: {pattern_type} (confidence: {confidence:.1f})")
    else:
        file_path = Path(args.file_path)
        if not file_path.exists():
            print(f"Error: File {args.file_path} does not exist")
            sys.exit(1)
        
        results = detector.analyze_file(file_path, args.duration, args.visualize)
        
        if results:
            sys.exit(1 if results['overall_problematic'] else 0)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Voice Pitch Tuner for Chatterbox TTS
Test different parameters and audio processing to adjust voice pitch
"""

import argparse
import logging
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from pathlib import Path
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def setup_mac_compatibility():
    """Setup Mac M4 compatibility"""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    map_location = torch.device(device)
    
    torch_load_original = torch.load
    def patched_torch_load(*args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = map_location
        return torch_load_original(*args, **kwargs)
    
    torch.load = patched_torch_load
    return device

class VoicePitchTuner:
    def __init__(self, voice_file=None):
        self.device = setup_mac_compatibility()
        logging.info(f"Loading Chatterbox TTS on {self.device}...")
        
        self.model = ChatterboxTTS.from_pretrained(device=self.device)
        self.voice_file = voice_file
        
        # Warmup
        with torch.no_grad():
            _ = self.model.generate("Test", exaggeration=0.5, cfg_weight=0.5)
        
        logging.info("✅ Voice tuner ready!")
    
    def test_parameter_combinations(self, text="Hello, this is a voice pitch test."):
        """Test different parameter combinations for pitch variation"""
        
        # Different parameter combinations that affect voice characteristics
        configs = [
            {"name": "Higher_Pitch_1", "exag": 0.2, "cfg": 0.3, "description": "Lower exaggeration + low CFG (tends higher pitch)"},
            {"name": "Higher_Pitch_2", "exag": 0.3, "cfg": 0.2, "description": "Very low CFG (lighter voice)"},
            {"name": "Standard", "exag": 0.5, "cfg": 0.5, "description": "Default settings"},
            {"name": "Deeper_1", "exag": 0.8, "cfg": 0.7, "description": "High exaggeration + high CFG (tends deeper)"},
            {"name": "Deeper_2", "exag": 1.0, "cfg": 0.8, "description": "Maximum exaggeration (deeper voice)"},
            {"name": "Dramatic", "exag": 1.2, "cfg": 0.6, "description": "Very high exaggeration (most dramatic)"},
        ]
        
        output_dir = Path("pitch_tests")
        output_dir.mkdir(exist_ok=True)
        
        logging.info(f"🎵 Testing {len(configs)} parameter combinations...")
        
        for config in configs:
            logging.info(f"Generating: {config['name']} - {config['description']}")
            
            try:
                with torch.no_grad():
                    if self.voice_file:
                        wav = self.model.generate(
                            text,
                            audio_prompt_path=self.voice_file,
                            exaggeration=config['exag'],
                            cfg_weight=config['cfg']
                        )
                    else:
                        wav = self.model.generate(
                            text,
                            exaggeration=config['exag'],
                            cfg_weight=config['cfg']
                        )
                
                filename = output_dir / f"{config['name']}.wav"
                ta.save(str(filename), wav, self.model.sr)
                
                logging.info(f"✅ Saved: {filename}")
                
            except Exception as e:
                logging.error(f"❌ Error generating {config['name']}: {e}")
        
        logging.info(f"\n🎧 Test complete! Listen to files in {output_dir} to compare pitch")
        return output_dir
    
    def apply_audio_pitch_shift(self, input_wav, pitch_shift_semitones):
        """Apply pitch shifting to audio using torchaudio"""
        try:
            # Pitch shift using resampling
            original_sr = self.model.sr
            
            # Calculate new sample rate for pitch shift
            # Positive semitones = higher pitch, negative = lower pitch
            pitch_ratio = 2 ** (pitch_shift_semitones / 12.0)
            new_sr = int(original_sr * pitch_ratio)
            
            # Resample to change pitch
            resampler = ta.transforms.Resample(original_sr, new_sr)
            shifted_wav = resampler(input_wav)
            
            # Resample back to original rate to maintain playback speed
            resampler_back = ta.transforms.Resample(new_sr, original_sr)
            final_wav = resampler_back(shifted_wav)
            
            return final_wav
            
        except Exception as e:
            logging.error(f"Pitch shift error: {e}")
            return input_wav
    
    def generate_with_pitch_variations(self, text="This is a pitch variation test."):
        """Generate audio with various pitch modifications"""
        
        output_dir = Path("pitch_variations")
        output_dir.mkdir(exist_ok=True)
        
        # First generate base audio with good settings
        logging.info("🎵 Generating base audio...")
        
        with torch.no_grad():
            if self.voice_file:
                base_wav = self.model.generate(
                    text,
                    audio_prompt_path=self.voice_file,
                    exaggeration=0.5,
                    cfg_weight=0.5
                )
            else:
                base_wav = self.model.generate(
                    text,
                    exaggeration=0.5,
                    cfg_weight=0.5
                )
        
        # Save original
        ta.save(str(output_dir / "original.wav"), base_wav, self.model.sr)
        
        # Apply different pitch shifts
        pitch_shifts = [
            (-4, "much_lower"),
            (-2, "lower"),
            (-1, "slightly_lower"),
            (0, "original"),
            (1, "slightly_higher"),
            (2, "higher"), 
            (4, "much_higher")
        ]
        
        for semitones, name in pitch_shifts:
            if semitones == 0:
                continue  # Skip original, already saved
            
            logging.info(f"Applying pitch shift: {semitones} semitones ({name})")
            
            try:
                shifted_wav = self.apply_audio_pitch_shift(base_wav, semitones)
                filename = output_dir / f"pitch_{name}.wav"
                ta.save(str(filename), shifted_wav, self.model.sr)
                
                logging.info(f"✅ Saved: {filename}")
                
            except Exception as e:
                logging.error(f"❌ Error creating {name}: {e}")
        
        logging.info(f"\n🎧 Pitch variations complete! Check {output_dir}")
        return output_dir
    
    def interactive_pitch_tuner(self):
        """Interactive mode for finding optimal pitch settings"""
        
        print("\n🎛️ Interactive Voice Pitch Tuner")
        print("Test different settings to find your preferred voice pitch")
        
        current_exag = 0.5
        current_cfg = 0.5
        
        while True:
            try:
                print(f"\n🎚️ Current settings:")
                print(f"   Exaggeration: {current_exag}")
                print(f"   CFG Weight: {current_cfg}")
                print(f"   Voice file: {self.voice_file or 'Default'}")
                
                print(f"\nOptions:")
                print(f"1. Test current settings")
                print(f"2. Adjust exaggeration (lower = higher pitch)")
                print(f"3. Adjust CFG weight (lower = lighter voice)")
                print(f"4. Try preset combinations")
                print(f"5. Apply pitch shift to last generation")
                print(f"6. Change voice file")
                print(f"7. Quit")
                
                choice = input("Select option (1-7): ").strip()
                
                if choice == "1":
                    text = input("Enter text to test (or press Enter for default): ").strip()
                    if not text:
                        text = "This is a voice pitch test with the current settings."
                    
                    logging.info("🎵 Generating with current settings...")
                    
                    with torch.no_grad():
                        if self.voice_file:
                            wav = self.model.generate(
                                text,
                                audio_prompt_path=self.voice_file,
                                exaggeration=current_exag,
                                cfg_weight=current_cfg
                            )
                        else:
                            wav = self.model.generate(
                                text,
                                exaggeration=current_exag,
                                cfg_weight=current_cfg
                            )
                    
                    filename = f"test_current.wav"
                    ta.save(filename, wav, self.model.sr)
                    print(f"✅ Saved: {filename}")
                
                elif choice == "2":
                    new_exag = input(f"New exaggeration (current: {current_exag}): ").strip()
                    try:
                        current_exag = float(new_exag)
                        print(f"✅ Exaggeration set to {current_exag}")
                    except ValueError:
                        print("❌ Invalid number")
                
                elif choice == "3":
                    new_cfg = input(f"New CFG weight (current: {current_cfg}): ").strip()
                    try:
                        current_cfg = float(new_cfg)
                        print(f"✅ CFG weight set to {current_cfg}")
                    except ValueError:
                        print("❌ Invalid number")
                
                elif choice == "4":
                    self.test_parameter_combinations()
                
                elif choice == "5":
                    shift = input("Pitch shift in semitones (+/-): ").strip()
                    try:
                        semitones = float(shift)
                        # Apply to last generated file
                        last_file = "test_current.wav"
                        if Path(last_file).exists():
                            wav, sr = ta.load(last_file)
                            shifted = self.apply_audio_pitch_shift(wav, semitones)
                            shifted_file = f"test_shifted_{semitones:+.1f}.wav"
                            ta.save(shifted_file, shifted, sr)
                            print(f"✅ Pitch shifted audio saved: {shifted_file}")
                        else:
                            print("❌ No test file found. Generate audio first.")
                    except ValueError:
                        print("❌ Invalid number")
                
                elif choice == "6":
                    new_voice = input("Voice file path (or Enter for default): ").strip()
                    if new_voice and Path(new_voice).exists():
                        self.voice_file = new_voice
                        print(f"✅ Voice file set to: {new_voice}")
                    elif not new_voice:
                        self.voice_file = None
                        print("✅ Using default voice")
                    else:
                        print("❌ Voice file not found")
                
                elif choice == "7":
                    print("👋 Goodbye!")
                    break
                
                else:
                    print("❌ Invalid option")
            
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Voice Pitch Tuner for Chatterbox TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python voice_pitch_tuner.py --interactive
  python voice_pitch_tuner.py --test-params
  python voice_pitch_tuner.py --pitch-variations --voice voices/myvoice.wav
        """
    )
    
    parser.add_argument(
        "--voice",
        help="Voice reference file for cloning"
    )
    
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive pitch tuning mode"
    )
    
    parser.add_argument(
        "--test-params",
        action="store_true",
        help="Test different parameter combinations"
    )
    
    parser.add_argument(
        "--pitch-variations",
        action="store_true",
        help="Generate pitch-shifted variations"
    )
    
    parser.add_argument(
        "--text",
        default="Hello, this is a voice pitch test.",
        help="Test text to use"
    )
    
    args = parser.parse_args()
    
    # Validate voice file
    if args.voice and not Path(args.voice).exists():
        logging.error(f"Voice file not found: {args.voice}")
        return
    
    tuner = VoicePitchTuner(voice_file=args.voice)
    
    if args.interactive:
        tuner.interactive_pitch_tuner()
    elif args.test_params:
        tuner.test_parameter_combinations(args.text)
    elif args.pitch_variations:
        tuner.generate_with_pitch_variations(args.text)
    else:
        # Default: run all tests
        logging.info("🎵 Running all pitch tests...")
        tuner.test_parameter_combinations(args.text)
        tuner.generate_with_pitch_variations(args.text)

if __name__ == "__main__":
    main()
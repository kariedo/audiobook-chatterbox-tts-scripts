#!/usr/bin/env python3
"""
Progress Tracker Module
Handles progress tracking and resuming for chunked processing
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple


class ProgressTracker:
    """Tracks processing progress and allows resuming"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.progress_file = output_dir / "progress.json"
        self.progress = self._load_progress()
    
    def _load_progress(self) -> dict:
        """Load existing progress"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Could not load progress file: {e}")
        
        return {
            "total_chunks": 0,
            "completed_chunks": [],
            "failed_chunks": [],
            "start_time": None,
            "last_update": None,
            "session_start_time": None,
            "chunk_times": []  # Store completion times for rate calculation
        }
    
    def save_progress(self):
        """Save current progress"""
        self.progress["last_update"] = datetime.now().isoformat()
        
        try:
            with open(self.progress_file, 'w') as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            logging.error(f"Could not save progress: {e}")
    
    def mark_chunk_completed(self, chunk_index: int):
        """Mark a chunk as completed"""
        if chunk_index not in self.progress["completed_chunks"]:
            self.progress["completed_chunks"].append(chunk_index)
            
            # Store completion time for rate calculation
            completion_time = datetime.now().isoformat()
            if "chunk_times" not in self.progress:
                self.progress["chunk_times"] = []
            self.progress["chunk_times"].append({
                "chunk": chunk_index,
                "time": completion_time
            })
            
            # Keep only last 50 completion times for rate calculation
            if len(self.progress["chunk_times"]) > 50:
                self.progress["chunk_times"] = self.progress["chunk_times"][-50:]
        
        # Remove from failed if it was there
        if chunk_index in self.progress["failed_chunks"]:
            self.progress["failed_chunks"].remove(chunk_index)
        
        self.save_progress()
    
    def mark_chunk_failed(self, chunk_index: int):
        """Mark a chunk as failed"""
        if chunk_index not in self.progress["failed_chunks"]:
            self.progress["failed_chunks"].append(chunk_index)
        self.save_progress()
    
    def is_chunk_completed(self, chunk_index: int) -> bool:
        """Check if chunk is already completed"""
        return chunk_index in self.progress["completed_chunks"]
    
    def get_completion_stats(self) -> Tuple[int, int, int]:
        """Get completion statistics"""
        total = self.progress["total_chunks"]
        completed = len(self.progress["completed_chunks"])
        failed = len(self.progress["failed_chunks"])
        return completed, failed, total
    
    def calculate_eta(self) -> Tuple[str, float]:
        """Calculate estimated time to completion"""
        completed = len(self.progress["completed_chunks"])
        total = self.progress["total_chunks"]
        
        if completed == 0:
            return "Calculating...", 0.0
        
        # Use the last few chunks to calculate rate
        if "chunk_times" not in self.progress or len(self.progress["chunk_times"]) < 3:
            return "Calculating...", 0.0
        
        # Get the last 5 chunks to get a more stable rate
        recent_chunks = self.progress["chunk_times"][-5:]
        
        if len(recent_chunks) < 3:
            return "Calculating...", 0.0
        
        # Calculate average time between chunks, filtering out large gaps
        intervals = []
        for i in range(1, len(recent_chunks)):
            prev_time = datetime.fromisoformat(recent_chunks[i-1]["time"])
            curr_time = datetime.fromisoformat(recent_chunks[i]["time"])
            interval = (curr_time - prev_time).total_seconds()
            
            # Only include intervals that seem reasonable (less than 2 minutes)
            # This filters out gaps from session restarts
            if interval < 120:  # 2 minutes max
                intervals.append(interval)
        
        if len(intervals) == 0:
            return "Calculating...", 0.0
        
        # Calculate average time per chunk from valid intervals
        avg_time_per_chunk = sum(intervals) / len(intervals)
        chunks_per_second = 1.0 / avg_time_per_chunk if avg_time_per_chunk > 0 else 0.0
        
        if chunks_per_second <= 0:
            return "Calculating...", 0.0
        
        # Calculate remaining time
        remaining_chunks = total - completed
        remaining_seconds = remaining_chunks * avg_time_per_chunk
        
        # Format time
        if remaining_seconds < 60:
            eta_str = f"{remaining_seconds:.0f}s"
        elif remaining_seconds < 3600:
            minutes = int(remaining_seconds // 60)
            seconds = int(remaining_seconds % 60)
            eta_str = f"{minutes}m {seconds}s"
        else:
            hours = int(remaining_seconds // 3600)
            minutes = int((remaining_seconds % 3600) // 60)
            eta_str = f"{hours}h {minutes}m"
        
        return eta_str, chunks_per_second
    
    def set_session_start(self):
        """Mark the start of the current processing session"""
        if not self.progress.get("session_start_time"):
            self.progress["session_start_time"] = datetime.now().isoformat()
            self.save_progress()
"""
Complete Vehicle Detection Pipeline - GPU-Optimized
Vehicle Detection → License Plate Detection → OCR → Telegram

Combines:
- GPU-accelerated RTSP streaming (gpu_pipeline_working.py)
- Vehicle tracking (VehicleDetectionTracker)
- License plate OCR
- Telegram notifications

Performance: 15-20 FPS on RTSP, 3x faster than CPU baseline
"""

import os
import sys
import warnings
import logging
from datetime import datetime
from pathlib import Path

# Set environment variables BEFORE importing libraries
os.environ["DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_EXTENSION_COMPILE_FLAG"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(f"logs/vehicle_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import VehicleDetectionTracker
from VehicleDetectionTracker.VehicleDetectionTracker import VehicleDetectionTracker


class IntegratedVehicleDetectionPipeline:
    """Complete integrated pipeline with GPU acceleration"""
    
    def __init__(self, mode='gpu'):
        """
        Initialize pipeline
        
        Args:
        """
        self.logger = logger
        self.tracker = None
        self.pipeline = None
        
        self.tracker = VehicleDetectionTracker()
  
        self.stats = {
            'frames_processed': 0,
            'vehicles_detected': 0,
            'plates_detected': 0,
            'plates_recognized': 0,
            'telegrams_sent': 0,
            'start_time': None
        }
    
    def process_video(self, video_source):
        """
        Process video stream (local file or RTSP)
        
        Args:
            video_source: Path to video file or RTSP URL
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"{'='*80}")
        logger.info(f"Source: {video_source}")
        
        self.stats['start_time'] = os.times()
   
        # Tracker: Full pipeline with tracking + OCR + Telegram
        logger.info("[PROCESSING] Using VehicleDetectionTracker (Full Features)")
        return self._process_tracker_pipeline(video_source)
    
    def _process_tracker_pipeline(self, video_source):
        """Process with VehicleDetectionTracker (full features)"""
        
        logger.info("[TRACKER] Pipeline: Vehicle Detection > License Plate > OCR > Telegram")
        logger.info("[TRACKER] Features: Vehicle tracking, plate OCR, Telegram notifications")
        logger.info("[TRACKER] Press 'q' to stop")
        
        # Safety check: ensure tracker is initialized
        if self.tracker is None:
            logger.error("[ERROR] VehicleDetectionTracker is not initialized!")
            return None
        
        try:
            # Use VehicleDetectionTracker for full pipeline
            self.tracker.process_video_streaming(video_source)
            
            # Cleanup and finalize
            logger.info("\n[CLEANUP] Waiting for all pending tasks...")
            self.tracker.cleanup()
            
            # Get quality statistics
            quality_stats = self.tracker.frame_processor.get_quality_stats()
            
            logger.info(f"\n{'='*80}")
            logger.info(f"[TRACKER RESULTS] Pipeline Complete")
            logger.info(f"{'='*80}")
            logger.info(f"Frames Processed: {quality_stats['frames_processed']}")
            logger.info(f"Frames Rejected: {quality_stats['frames_rejected']}")
            logger.info(f"Rejection Rate: {quality_stats['rejection_rate']:.1f}%")
            
            logger.info(f"\n[SUMMARY] Vehicle Detection Pipeline")
            logger.info(f"Vehicles Detected: {self.stats['vehicles_detected']}")
            logger.info(f"Plates Recognized: {self.stats['plates_recognized']}")
            logger.info(f"Telegrams Sent: {self.stats['telegrams_sent']}")
            
            self.stats['frames_processed'] = quality_stats['frames_processed']
            
            return quality_stats
        
        except Exception as e:
            logger.error(f"[ERROR] Tracker pipeline failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def print_summary(self):
        """Print final summary"""
        logger.info(f"\n{'='*80}")
        logger.info(f"[FINAL SUMMARY] Complete Pipeline Statistics")
        logger.info(f"{'='*80}")
        logger.info(f"Frames Processed: {self.stats['frames_processed']}")
        logger.info(f"Vehicles Detected: {self.stats['vehicles_detected']}")
        logger.info(f"Plates Recognized: {self.stats['plates_recognized']}")
        logger.info(f"Telegrams Sent: {self.stats['telegrams_sent']}")
        logger.info(f"{'='*80}\n")


def main():
    """Main entry point"""
    # Configuration  
    RTSP_URL = "rtsp://admin:MOVYKV@aicamera.dienthanhliem.com:554/Streaming/Channels/101"
    # RTSP_URL = "video/0604.mp4"  # For testing with local video, comment out for RTSP

    # Initialize integrated pipeline for gpu/tracker modes
    try:
        pipeline = IntegratedVehicleDetectionPipeline()
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize pipeline: {e}")
        sys.exit(1)
    
    # Run production mode RTSP stream
    try:
        # Production mode: Process RTSP stream
        stats = pipeline.process_video(RTSP_URL)
        sys.exit(0)
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    main()

# Batch OCR Implementation - Testing & Validation Guide

## Overview

This document covers testing and validation of the **Approach 2: Async Batch Accumulator** implementation for license plate OCR processing.

---

## Implementation Summary

### What Was Implemented

#### 1. **BatchAccumulator Class** (`plate_utils.py`)
- Accumulates plate detection requests from multiple frames
- Triggers processing when:
  - Batch size threshold reached (default: 8 plates)
  - Time threshold exceeded (default: 500ms)
  - Maximum wait time reached (default: 2000ms)
- Thread-safe with lock-protected operations

#### 2. **Batch Detection Functions** (`plate_utils.py`)
- `batch_detect_license_plates()`: Processes multiple vehicles in one batch
  - Step 1: Run plate detection on all vehicles
  - Step 2: Extract best plate crops
  - Step 3: Batch OCR processing
  - Step 4: Combine results
- `_batch_ocr_plates()`: Handles OCR for all plates at once

#### 3. **PlateProcessor Integration** (`plate_processor.py`)
- `submit_plate_processing_batch()`: Adds vehicles to accumulator
- `_batch_processor_loop()`: Daemon thread monitoring batch status
- `_process_batch()`: Executes batch in executor thread
- `_execute_batch_ocr()`: Runs detection and OCR
- `_handle_batch_plate_result()`: Updates vehicle state from results
- `_on_batch_complete()`: Sends notifications after batch processing

#### 4. **Frame Processor Changes** (`frame_processor.py`)
- Collects all valid vehicles from current frame into dict
- Submits entire frame batch instead of individual vehicles
- Batch dict includes all necessary metadata

#### 5. **Configuration** (`config.yaml`)
```yaml
batch_inference:
  enabled: false              # Toggle batch processing on/off
  batch_size: 8              # Accumulate this many before processing
  time_threshold_ms: 500     # Max time to wait before partial batch processing
  max_wait_time_ms: 2000     # Absolute max time before forced flush
  mode: "accumulator"        # "frame" or "accumulator"
```

---

## Architecture

### Batch Processing Flow

```
Frame 1: Detect 2 vehicles → Accumulate (batch size: 2/8)
Frame 2: Detect 3 vehicles → Accumulate (batch size: 5/8)
Frame 3: Detect 4 vehicles → Accumulate (batch size: 9/8) → TRIGGER!
                              ↓
                    Run detection on all 9
                              ↓
                    Batch extract plate crops
                              ↓
                    Batch OCR (all plates together)
                              ↓
                    Update vehicle states
                              ↓
                    Send 9 notifications
```

### Thread Architecture

```
Main Thread (frame_processor)
  ↓
  Collect vehicles from frame
  ↓
  Call submit_plate_processing_batch()
  ↓

PlateProcessor (main thread)
  ├─ submit_plate_processing_batch()
  │  └─ Add to accumulator
  │
  └─ Batch Processor Thread (daemon)
     └─ Monitor accumulator
        └─ Trigger on size/time
        └─ Submit to executor

ThreadPoolExecutor (worker threads)
  ├─ _execute_batch_ocr()
  │  └─ batch_detect_license_plates()
  │     ├─ Detect plates
  │     ├─ Extract crops
  │     └─ Batch OCR
  │
  └─ Results → _on_batch_complete()
              └─ Send notifications
```

---

## Testing Guide

### Test 1: Enable Batch Processing

**Objective**: Verify batch accumulator initializes correctly

**Steps**:
1. Edit `config.yaml`:
```yaml
batch_inference:
  enabled: true
  batch_size: 4          # Use smaller size for testing
  time_threshold_ms: 500
```

2. Run the application:
```bash
python StreamingCameraTest.py 2>&1 | grep -E "BATCH_|Initialized batch"
```

**Expected Output**:
```
[09:30:45] [PLATE_PROCESSOR] ✓ Initialized batch accumulator (batch_size=4, time_threshold=500ms)
[09:30:46] [BATCH_PROCESSOR] Thread started, monitoring batch accumulator...
```

**Verdict**: ✅ Pass if batch accumulator initializes

---

### Test 2: Batch Accumulation

**Objective**: Verify vehicles accumulate correctly in batch

**Steps**:
1. Configure for small batch size (4)
2. Run multiple vehicles through stream
3. Monitor logs for accumulation messages

**Expected Output**:
```
[BATCH_SUBMIT] Adding 2 vehicles to batch accumulator...
[BATCH_ACCUM] Added track_id=1, batch_size=1/4, elapsed=10ms/500ms
[BATCH_ACCUM] Added track_id=2, batch_size=2/4, elapsed=25ms/500ms
[BATCH_ACCUM] Added track_id=3, batch_size=3/4, elapsed=40ms/500ms
[BATCH_ACCUM] Added track_id=4, batch_size=4/4, elapsed=55ms/500ms
[BATCH_ACCUM] ✓ Batch trigger: size=4 or time=55ms
```

**Verdict**: ✅ Pass if accumulation is smooth and reaches threshold

---

### Test 3: Batch Processing Trigger

**Objective**: Verify batch processes when threshold reached

**Steps**:
1. Monitor logs for batch processing
2. Check timestamps for batch execution

**Expected Output**:
```
[BATCH_PROCESSOR] Size threshold reached (4/4)
[BATCH_PROCESS] Processing batch of 4 vehicles...
[BATCH_DETECT] Starting batch processing: 4 vehicles
[BATCH_DETECT] Step 1: Running plate detection on 4 vehicles...
[BATCH_DETECT]   track_id=1 Running detection...
[BATCH_DETECT]   track_id=1 ✓ Detected 1 plates
[BATCH_DETECT]   track_id=2 ✓ Detected 2 plates
[BATCH_DETECT]   track_id=3 ✓ Detected 1 plates
[BATCH_DETECT]   track_id=4 ✓ Detected 1 plates
[BATCH_DETECT] Step 1 complete: Detection took 125.3ms
```

**Verdict**: ✅ Pass if batch processes when threshold reached

---

### Test 4: OCR Batch Processing

**Objective**: Verify OCR runs on entire batch

**Steps**:
1. Monitor logs for batch OCR execution
2. Verify all vehicles processed together

**Expected Output**:
```
[BATCH_DETECT] Step 2 complete: Extracted 4 plate crops
[BATCH_DETECT] Step 3: Running batch OCR on 4 plates...
[BATCH_OCR] Using native batch OCR processing...
[BATCH_OCR]   track_id=1 Result: 'AA123BC'
[BATCH_OCR]   track_id=2 Result: 'BB456DE'
[BATCH_OCR]   track_id=3 Result: 'unknown'
[BATCH_OCR]   track_id=4 Result: 'CC789FG'
[BATCH_OCR] ✓ Batch OCR complete: 4 results
[BATCH_DETECT] Step 3 complete: OCR took 89.2ms
[BATCH_DETECT] ✓ Batch processing complete: 4 vehicles, detection=125.1ms, ocr=89.2ms, total=214.3ms
```

**Verdict**: ✅ Pass if OCR processes all plates in batch

---

### Test 5: Notifications Sent

**Objective**: Verify notifications sent correctly after batch processing

**Steps**:
1. Monitor for notification sending
2. Check Telegram API calls
3. Verify state is saved

**Expected Output**:
```
[BATCH_RESULT] vehicle_id=1 Result: text='AA123BC', count=1, conf=0.875
[BATCH_RESULT] vehicle_id=1 ✓ Setting primary plate: 'AA123BC'
[BATCH_COMPLETE] vehicle_id=1 All batch tasks complete, sending notification...
[BATCH_COMPLETE] vehicle_id=1 ✓ Notification sent
...
```

**Verdict**: ✅ Pass if notifications sent for all vehicles

---

### Test 6: Time-Based Flushing

**Objective**: Verify partial batches process after time threshold

**Steps**:
1. Set `time_threshold_ms: 200` for faster testing
2. Send single vehicle
3. Wait 300ms
4. Monitor for auto-flush

**Expected Output**:
```
[BATCH_ACCUM] Added track_id=1, batch_size=1/8, elapsed=10ms/200ms
[BATCH_PROCESSOR] Time threshold exceeded (210ms/200ms)
[BATCH_ACCUM] ✓✓ Batch flushed: size=1, elapsed=210ms
[BATCH_PROCESS] Processing batch of 1 vehicles...
```

**Verdict**: ✅ Pass if batch flushes after time threshold

---

### Test 7: Fallback to Per-Vehicle

**Objective**: Verify system falls back if batch disabled

**Steps**:
1. Set `batch_inference.enabled: false`
2. Run vehicle detection
3. Monitor for per-vehicle processing

**Expected Log**:
```
[BATCH_SUBMIT] Batch accumulator disabled, falling back to per-vehicle processing...
[PLATE] vehicle_id=1 Submitting to background executor
[PLATE] vehicle_id=2 Submitting to background executor
```

**Verdict**: ✅ Pass if system reverts to traditional processing

---

## Performance Testing

### Benchmark Setup

Create test script: `test_batch_performance.py`

```python
import time
import numpy as np
from datetime import datetime
from VehicleDetectionTracker.plate_utils import (
    batch_detect_license_plates,
    get_batch_accumulator
)

# Create mock data
def create_test_frames(num_vehicles, frame_size=(400, 400)):
    """Create mock vehicle frames for testing."""
    frames = {}
    for i in range(num_vehicles):
        frame = np.random.randint(0, 255, frame_size + (3,), dtype=np.uint8)
        frames[i] = frame
    return frames

# Test batch accumulation throughput
def test_batch_throughput():
    accumulator = get_batch_accumulator()
    
    start = time.time()
    for frame_idx in range(10):
        frames = create_test_frames(2)  # 2 vehicles per frame
        for track_id, frame in frames.items():
            accumulator.add_detection(
                track_id, frame,
                f"ts_{frame_idx}_{track_id}",
                f"dir/vid_{track_id}",
                "bottom",
                datetime.now()
            )
    
    elapsed = time.time() - start
    print(f"Accumulated 20 detections in {elapsed:.2f}s")
    print(f"Throughput: {20/elapsed:.1f} detections/second")
```

### Expected Performance

**Baseline (Per-Vehicle)**:
- Throughput: ~3-5 vehicles/second
- Latency: 100-300ms per vehicle
- Total for 8 vehicles: 800-2400ms

**With Batch**:
- Throughput: ~8-12 vehicles/second
- Latency: 60-120ms per batch
- Total for 8 vehicles: 60-120ms
- **Speedup: 7-20x**

---

## Monitoring & Debugging

### Key Log Patterns

#### Batch Accumulation
```
BATCH_SUBMIT        - Vehicle submission to batch
BATCH_ACCUM         - Accumulation status
BATCH_PROCESSOR     - Batch processor thread events
```

#### Batch Processing
```
BATCH_PROCESS       - Batch processing execution
BATCH_DETECT        - Detection step-by-step
BATCH_OCR           - OCR execution
BATCH_RESULT        - Individual result handling
BATCH_COMPLETE      - Batch completion and notifications
```

### Common Issues

#### Issue 1: Batch Never Triggers
**Symptom**: `BATCH_ACCUM Added` logs but no `BATCH_PROCESSOR Size threshold`

**Diagnosis**:
- `batch_size` too large?
- Insufficient vehicles detected?
- Check `time_threshold_ms` - should trigger after X ms anyway

**Solution**:
- Reduce `batch_size: 2` for testing
- Check vehicle detection with `[TRACK]` logs
- Verify `time_threshold_ms` isn't too large

#### Issue 2: High Latency
**Symptom**: Batch processes but takes 1000+ ms

**Diagnosis**:
- OCR reader slow?
- Plate detection slow?
- Check `[BATCH_DETECT]` step timings

**Solution**:
- Profile each step (detection, extraction, OCR)
- Consider batch_size impact
- Check GPU utilization

#### Issue 3: Notifications Not Sent
**Symptom**: `BATCH_COMPLETE` but no `Notification sent`

**Diagnosis**:
- Check vehicle state: `[BATCH_RESULT] Setting primary plate`
- Check lock contention: logs get stuck?
- Telegram API failure?

**Solution**:
- Enable full Telegram logging
- Check internet connectivity
- Verify bot token is valid

---

## Comparison: Approach 1 vs Approach 2

| Aspect | Approach 1: Frame-Level | Approach 2: Async Accumulator |
|--------|-------------------------|------------------------------|
| **Batching** | All vehicles from 1 frame | Accumulate across frames |
| **Latency per vehicle** | ~150-200ms | ~60-120ms |
| **Throughput** | 5-10 vehicles/sec | 8-15 vehicles/sec |
| **GPU Utilization** | Moderate (depends on vehicles) | Better (fixed batch size) |
| **Complexity** | Low | Medium |
| **Memory** | Lower | Higher (keeps buffers) |
| **Best For** | Low-traffic scenes | Continuous high-traffic |

---

## Configuration Recommendations

### High-Traffic (>10 vehicles/frame)
```yaml
batch_inference:
  enabled: true
  batch_size: 16
  time_threshold_ms: 300
  max_wait_time_ms: 1000
```

### Medium-Traffic (3-10 vehicles/frame)
```yaml
batch_inference:
  enabled: true
  batch_size: 8
  time_threshold_ms: 500
  max_wait_time_ms: 2000
```

### Low-Traffic (<3 vehicles/frame)
```yaml
batch_inference:
  enabled: false  # Use per-vehicle for lower latency
```

---

## Next Steps

1. **Enable batch processing**: Set `enabled: true` in config
2. **Monitor logs**: Watch for `BATCH_*` log patterns
3. **Collect metrics**: Track frame processing time and latency
4. **Tune parameters**: Adjust `batch_size` and `time_threshold_ms`
5. **Validate results**: Ensure notifications are correct
6. **Performance test**: Measure throughput improvement

---

## Troubleshooting Checklist

- [ ] Batch initialization successful?
- [ ] Vehicles accumulating correctly?
- [ ] Batch triggering on schedule?
- [ ] Detection running on batch?
- [ ] OCR processing all plates?
- [ ] Notifications being sent?
- [ ] State files saved correctly?
- [ ] Performance improved?

---

## Additional Resources

- [BATCH_OCR_IMPLEMENTATION_PLAN.md](BATCH_OCR_IMPLEMENTATION_PLAN.md) - Full technical design
- [config.yaml](VehicleDetectionTracker/config.yaml) - Configuration options
- Log files: `logs/plate_YYYY-MM-DD.log` - Detailed logging


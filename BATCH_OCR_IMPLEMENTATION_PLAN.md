# Batch OCR Inference Implementation Plan

## Current Architecture Analysis

### Bottleneck
- **Frame Detection**: Single YOLOv8 pass per frame (already efficient ✓)
- **Vehicle Detection**: Per-vehicle YOLOv8 inference in ThreadPoolExecutor (sequential OCR bottleneck ✗)
- **OCR**: Runs individually per vehicle in worker threads (2-4 workers typically)
  - Vehicle 1: OCR starts at t=50ms
  - Vehicle 2: OCR starts at t=150ms (waiting for worker)
  - Vehicle 3: OCR starts at t=250ms (waiting for worker)
  - Total: 3 sequential OCRs = 300-500ms per frame

### Improvement Opportunity
- **Batch OCR**: Process all detected plates at once
- If 3 vehicles detected: Run OCR once with 3 images = 150-200ms per frame
- **Speedup: 2-3x faster** for typical multi-vehicle frames

---

## Implementation Strategy: Two Approaches

### Approach 1: Frame-Level Batch OCR (Recommended - Low Risk)
**Concept**: Collect all vehicle detections from current frame, batch OCR them after all detections complete

**Advantages**:
- Minimal changes to existing architecture
- Keeps per-vehicle detection unchanged
- Natural batching at frame boundary
- Easy to toggle on/off

**Disadvantages**:
- Adds slight latency waiting for all vehicle detections
- OCR happens after plate detection (not real-time)

**Implementation Steps**:
1. Track detected vehicles and their plate crops in current frame
2. Wait for all per-frame vehicles to be detected
3. Submit ONE batch OCR task with all plates
4. Update vehicle states with batch results
5. Send notifications as before

---

### Approach 2: Asynchronous Batch Accumulator (More Complex - Higher Risk)
**Concept**: Accumulate plate detections from multiple frames, OCR when batch reaches target size

**Advantages**:
- Better throughput for continuous streams
- Utilizes GPU better (larger batches)

**Disadvantages**:
- Complex state management
- Harder to debug
- Requires careful synchronization
- Possible ordering issues

---

## Recommended: Approach 1 - Frame-Level Batch OCR

### Architecture Changes

```
Current Flow:
├─ Frame arrives
├─ Detect vehicles (YOLOv8 tracking)
├─ For each vehicle:
│  ├─ Submit to executor: detect_license_plate_sync
│  └─ OCR in worker thread (waits for available worker)
└─ Stream next frame

New Flow:
├─ Frame arrives
├─ Detect vehicles (YOLOv8 tracking)
├─ For each vehicle:
│  ├─ Collect: plate detection + crop image
│  └─ Add to batch_vehicles dict
├─ When frame detection done:
│  ├─ Check if batch_vehicles has items
│  ├─ Submit: batch_detect_license_plates(batch_vehicles)
│  ├─ OCR runs ONCE on all plates
│  └─ Update all vehicles with results
└─ Stream next frame
```

### Module Changes

#### 1. `plate_utils.py` - Add Batch Detection Function

```python
def batch_detect_license_plates(
    plate_model,
    vehicle_frames_dict: Dict[int, np.ndarray],  # {track_id: frame}
    ocr_reader,
    model_lock,
    timestamp_strs_dict: Dict[int, str],  # {track_id: timestamp_str}
    vehicle_dirs_dict: Dict[int, str],  # {track_id: vehicle_dir}
    config: dict
) -> Dict[int, dict]:  # {track_id: {text, count}}
    """
    Batch detect license plates and OCR for multiple vehicles in one pass.
    
    Args:
        plate_model: YOLOv8 model
        vehicle_frames_dict: Dict of {track_id: vehicle_frame}
        ocr_reader: OCR reader instance
        model_lock: Thread lock for model inference
        timestamp_strs_dict: Dict of {track_id: timestamp_string}
        vehicle_dirs_dict: Dict of {track_id: vehicle_directory}
        config: Configuration dict
    
    Returns:
        Dict of {track_id: {text, count}} with OCR results
    """
    results = {}
    
    # Step 1: Detect plates in all vehicles
    plate_detections = {}  # {track_id: (plates, boxes)}
    
    with model_lock:
        for track_id, frame in vehicle_frames_dict.items():
            detections = _batch_detect_single_vehicle(
                plate_model, frame, track_id, config
            )
            plate_detections[track_id] = detections
    
    # Step 2: Extract and save plate crops
    plate_crops_dict = {}  # {track_id: plate_image}
    
    for track_id, (detections, boxes) in plate_detections.items():
        if detections['count'] > 0:
            plate_image = _extract_best_plate(
                vehicle_frames_dict[track_id],
                boxes[0],  # best detection
                detections['count']
            )
            plate_crops_dict[track_id] = plate_image
    
    # Step 3: Batch OCR all detected plates
    if ocr_reader and plate_crops_dict:
        batch_ocr_results = _batch_ocr_plates(
            ocr_reader, plate_crops_dict, model_lock
        )
    else:
        batch_ocr_results = {}
    
    # Step 4: Combine results
    for track_id, plate_crop in plate_crops_dict.items():
        results[track_id] = {
            'text': batch_ocr_results.get(track_id, 'unknown'),
            'count': plate_detections[track_id][0]['count']
        }
    
    return results


def _batch_ocr_plates(
    ocr_reader,
    plate_crops_dict: Dict[int, np.ndarray],
    model_lock
) -> Dict[int, str]:
    """
    Batch OCR multiple plate images.
    
    Returns:
        Dict of {track_id: ocr_text}
    """
    results = {}
    
    # If OCR reader supports batch processing, use it
    if hasattr(ocr_reader, 'read_license_plate_batch'):
        with model_lock:
            batch_results = ocr_reader.read_license_plate_batch(
                list(plate_crops_dict.values())
            )
        for (track_id, _), ocr_text in zip(plate_crops_dict.items(), batch_results):
            results[track_id] = ocr_text
    else:
        # Fallback: Serial OCR (still batched for lock contention reduction)
        with model_lock:
            for track_id, plate_image in plate_crops_dict.items():
                results[track_id] = ocr_reader.read_license_plate(plate_image)
    
    return results
```

#### 2. `plate_processor.py` - Add Frame-Level Batching

```python
class PlateProcessor:
    def __init__(self, ...):
        self.current_frame_vehicles = {}  # Vehicles detected in current frame
        self.frame_batch_lock = threading.Lock()
        self.last_frame_id = -1
    
    def submit_plate_processing_batch(self, frame_vehicles: Dict[int, dict]):
        """
        Submit batch plate processing for all vehicles in current frame.
        
        Args:
            frame_vehicles: Dict of {track_id: {frame, direction, timestamp, ...}}
        """
        if not frame_vehicles:
            return
        
        self.log(f"[BATCH_OCR] Submitting batch OCR for {len(frame_vehicles)} vehicles")
        
        future = self.executor.submit(
            self._process_batch_plates,
            frame_vehicles
        )
        
        # Add callback for when batch task completes
        future.add_done_callback(
            lambda f: self._on_batch_plate_task_complete(frame_vehicles.keys())
        )
    
    def _process_batch_plates(self, frame_vehicles: Dict[int, dict]) -> Dict[int, dict]:
        """Execute batch OCR and update vehicle states."""
        try:
            # Extract inputs for batch processing
            vehicle_frames_dict = {
                tid: v['frame'] for tid, v in frame_vehicles.items()
            }
            timestamp_strs_dict = {
                tid: v['timestamp_str'] for tid, v in frame_vehicles.items()
            }
            vehicle_dirs_dict = {
                tid: v['vehicle_dir'] for tid, v in frame_vehicles.items()
            }
            
            # Run batch detection and OCR
            batch_results = batch_detect_license_plates(
                self.plate_model,
                vehicle_frames_dict,
                self.ocr_reader,
                self.model_lock,
                timestamp_strs_dict,
                vehicle_dirs_dict,
                self.detection_config
            )
            
            # Update vehicle states and check for notifications
            for track_id, result in batch_results.items():
                self._handle_plate_result(
                    track_id,
                    result,
                    frame_vehicles[track_id]['direction'],
                    frame_vehicles[track_id]['timestamp'],
                    frame_vehicles[track_id]['vehicle_dir']
                )
            
            self.log(f"[BATCH_OCR] ✓ Batch OCR complete for {len(batch_results)} vehicles")
            return batch_results
            
        except Exception as e:
            self.log(f"[BATCH_OCR] Error in batch OCR: {e}")
            import traceback
            self.log(traceback.format_exc())
            return {}
```

#### 3. `frame_processor.py` - Collect and Submit Batch

```python
def process_frame_streaming(self, frame, frame_timestamp, plate_processor):
    # ... existing vehicle detection code ...
    
    # Instead of:
    #   plate_processor.submit_plate_processing(track_id, ...)
    
    # Collect all vehicles detected in this frame
    frame_vehicles_batch = {}
    
    for box, track_id, class_id in zip(boxes, track_ids, class_id_list):
        # ... existing filtering and frame extraction ...
        
        if vehicle_frame.size > 0:
            frame_vehicles_batch[track_id] = {
                'frame': vehicle_frame.copy(),
                'direction': direction_label,
                'timestamp': frame_timestamp,
                'timestamp_str': timestamp_str,
                'vehicle_dir': vehicle_dir
            }
    
    # Submit entire batch for processing
    if frame_vehicles_batch:
        plate_processor.submit_plate_processing_batch(frame_vehicles_batch)
    
    return frame
```

---

## Performance Estimates

### Baseline (Current)
- Frame rate: 30fps
- Detections per frame: 2-4 vehicles
- OCR time per vehicle: 50-100ms
- Total OCR time: 100-400ms (sequential in thread pool)
- **Throughput impact: Significant**

### With Batch OCR (Approach 1)
- Frame rate: 30fps (same detection)
- Detections per frame: 2-4 vehicles
- Batch OCR time: 60-120ms (all at once)
- Total OCR time: 60-120ms
- **Speedup: 2-3x**

### With Async Batch Accumulator (Approach 2)
- Frame rate: 30fps
- Batch size: 8-16
- Batch OCR time: 150-200ms
- Throughput: Higher for continuous monitoring
- **Speedup: 2-5x depending on vehicle density**

---

## Implementation Phases

### Phase 1: Foundation (Low Risk)
- [ ] Add `batch_detect_license_plates()` to `plate_utils.py`
- [ ] Add batch collection logic to `frame_processor.py`
- [ ] Add `submit_plate_processing_batch()` to `plate_processor.py`
- [ ] Test with debug logging

### Phase 2: Optimization
- [ ] Profile OCR performance
- [ ] Implement batch OCR in OCR reader if available
- [ ] Optimize image preprocessing for batch

### Phase 3: Advanced (Optional)
- [ ] Implement Approach 2 (async batch accumulator)
- [ ] Add configurable batch sizes
- [ ] Implement adaptive batching

---

## Testing Strategy

1. **Unit Tests**
   - Verify batch detection handles 1, 2, 3+ vehicles
   - Verify OCR results match single-vehicle mode
   - Test edge cases (empty frames, single vehicle)

2. **Integration Tests**
   - Run on real stream with 2-4 vehicles
   - Verify notifications still sent correctly
   - Check state consistency

3. **Performance Tests**
   - Measure frame processing time
   - Measure OCR time (batch vs sequential)
   - Monitor memory usage

---

## Rollback Plan

- Keep current `submit_plate_processing()` unchanged
- New batch feature can be toggled with config flag
- If issues: revert to per-vehicle processing

---

## Configuration

```yaml
# config.yaml
batch_inference:
  enabled: true
  mode: "frame"  # "frame" or "accumulator"
  accumulator_batch_size: 8
  max_wait_time_ms: 500
```

---

## Key Considerations

1. **Thread Safety**: Batch dict needs proper locking
2. **Memory**: Store multiple frames temporarily
3. **Notifications**: Ensure ALL vehicles get notified (not just first)
4. **Backward Compatibility**: Existing single-vehicle code should still work
5. **Debugging**: Add detailed batch processing logs

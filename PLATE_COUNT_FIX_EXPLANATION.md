# Vehicle Plate Counts Fix - Summary

## Issue Fixed

`vehicle_plate_counts` was counting **frame detections**, not **vehicles**.

**Before:**
- Track ID 1, plate "77C-123": count=5 (detected in 5 frames)
- Track ID 2, plate "77C-123": count=3 (detected in 3 frames)
- **Total for plate "77C-123": 8** (WRONG - should be 2 vehicles)

**After:**
- Track ID 1, plate "77C-123": count=1 (1 vehicle)
- Track ID 2, plate "77C-123": count=1 (1 vehicle)  
- **Total for plate "77C-123": 2** (CORRECT - 2 vehicles)

---

## Changes Made

### 1. **Tracking Logic** (line ~451)
Changed from incrementing count on each frame detection:
```python
# OLD (WRONG)
self.vehicle_plate_counts[track_id][plate] = (
    self.vehicle_plate_counts[track_id].get(plate, 0) + 1
)

# NEW (CORRECT)
if plate not in self.vehicle_plate_counts[track_id]:
    self.vehicle_plate_counts[track_id][plate] = 1
```

### 2. **Data Structure Documentation** (line ~48)
Updated comment to clarify meaning:
```python
# OLD
self.vehicle_plate_counts = {}  # {track_id: {plate_text: count}}

# NEW
self.vehicle_plate_counts = {}  # {track_id: {plate_text: 1}} - count of vehicles with each plate
```

### 3. **Load/Restore Logic** (line ~1005)
Added normalization for old data on load:
```python
# Converts old frame-detection counts to new vehicle counts
# Example: count=5 (5 frames) → count=1 (1 vehicle)
if old_count != 1:
    log_plate(track_id, f"Normalized plate count from {old_count} (frame detections) to 1 (vehicle)")
```

---

## Behavior

### Daily Summary Function
The `get_today_vehicles_summary()` now correctly reports:
```
Plate "77C-123.45": 2 vehicles  (instead of summing frame detections)
Plate "77C-456.78": 1 vehicle
Plate "77C-789.01": 3 vehicles
```

### State File Format
```json
{
  "vehicle_plate_counts": {
    "1": {"77C-253.29": 1},   // Vehicle 1 has 1 count (1 vehicle)
    "3": {"77C-253.29": 1}    // Vehicle 3 has 1 count (1 vehicle)
  }
}
```

Sum = 2 vehicles with plate "77C-253.29" ✅

---

## Data Migration

When the system loads old state files with frame-based counts, it automatically normalizes them:
```
Old: {plate: 5}  (5 frame detections)
         ↓ (normalize on load)
New: {plate: 1}  (1 vehicle)
```

This ensures backward compatibility without losing data.

---

## Verification

To verify the fix is working:

```python
python verify_plate_counts.py
```

Expected output:
```
✅ Vehicle State Summary
Total vehicles: 2

Vehicle Plate Mapping:
  Track ID 1: plate=77C-253.29, counts={'77C-253.29': 1}
  Track ID 3: plate=77C-253.29, counts={'77C-253.29': 1}

📊 Plate Summary (how many vehicles have each plate):
  Plate "77C-253.29": 2 vehicle(s)
```

---

## Impact

- ✅ Daily vehicle reports now show correct vehicle counts, not frame detection counts
- ✅ Backward compatible with existing state files (automatic normalization on load)
- ✅ Clearer semantics: vehicle_plate_counts = vehicles per plate
- ✅ Accurate aggregation in daily summaries


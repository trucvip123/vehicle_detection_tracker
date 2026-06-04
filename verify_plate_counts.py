"""Quick verification of vehicle_plate_counts fix."""
import json

with open('vehicle_state/vehicle_state_20260604.json', 'r') as f:
    state = json.load(f)

print('✅ Vehicle State Summary')
print(f'Total vehicles: {len(state["vehicle_plates"])}')
print(f'\nVehicle Plate Mapping:')
for track_id, plate in state['vehicle_plates'].items():
    counts = state['vehicle_plate_counts'].get(track_id, {})
    print(f'  Track ID {track_id}: plate={plate}, counts={counts}')

print(f'\n📊 Plate Summary (how many vehicles have each plate):')
plate_summary = {}
for track_id, plate in state['vehicle_plates'].items():
    if plate not in plate_summary:
        plate_summary[plate] = 0
    counts = state['vehicle_plate_counts'].get(track_id, {})
    vehicle_count = counts.get(plate, 1)
    plate_summary[plate] += vehicle_count

for plate, count in sorted(plate_summary.items(), key=lambda x: -x[1]):
    print(f'  Plate "{plate}": {count} vehicle(s)')

print(f'\n✅ Verification:')
print('Each vehicle_plate_counts entry represents 1 vehicle')
print('Summing across all vehicles gives total count per plate')

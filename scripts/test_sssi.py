from geosight.tools.geocoder import geocode_postcode
from geosight.tools.protected_areas import fetch_protected_areas

loc = geocode_postcode('TQ13 8HH')  # Dartmoor
print(f'Testing: {loc.display_name}')
result = fetch_protected_areas(loc.lat, loc.lon, radius_km=5.0)
print(result.summary)
for d in result.designations:
    print(f'  - {d.type}: {d.name}')
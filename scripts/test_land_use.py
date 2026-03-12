from geosight.tools.geocoder import geocode_postcode
from geosight.tools.land_use import fetch_land_use

loc = geocode_postcode('TQ13 8HH')  # Dartmoor
print(f'Testing: {loc.display_name}')
result = fetch_land_use(loc.lat, loc.lon, radius_m=500)
print(result.summary)
print('Land uses:', result.land_uses)
print('Natural features:', result.natural_features)
print('Waterways:', result.waterways)
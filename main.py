import ee
ee.Initialize(project='key-scarab-332210')

# ---------- CONFIG ----------
POLYGON = [
      [56.73698287457228, 37.461360390525726],
      [37.46150356833912, 56.73837695270777],
      [56.73854425549507, 37.46047763204037],
      [56.73715017735958, 37.460334452262494],
    ]
FIELD = ee.Geometry.Polygon([POLYGON])

START = '2025-06-01'
END   = '2025-08-31'
SCALE = 10

# thresholds (tweak to taste)
CLOUD_PROB_TH = 40   # cloud probability threshold
NDVI_MIN = 0.35
NDWI_MAX = 0.10
MIN_CONNECTED_PIXELS = 5
DYING_THRESH = -0.002  # NDVI/day
HEALTHY_THRESH = 0.002

# ---------- 1) load collections ----------
s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
      .filterBounds(FIELD)
      .filterDate(START, END)
      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 70))
      .sort('system:time_start', False))

cloud_prob = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
              .filterBounds(FIELD)
              .filterDate(START, END))

# join cloud probability by system:index
saveCloud = ee.Join.saveFirst(matchKey='cloud_prob')
matchFilter = ee.Filter.equals(leftField='system:index', rightField='system:index')
joined = saveCloud.apply(primary=s2, secondary=cloud_prob, condition=matchFilter)
s2_joined = ee.ImageCollection(joined)

# ---------- 2) function to prepare each image for slope and composite ----------
def prepare(img_element):
    img = ee.Image(img_element)  # ensure Image

    # attached cloud probability image (or fallback)
    cloud_prop = img.get('cloud_prob')
    cloud_img = ee.Image(ee.Algorithms.If(
        cloud_prop,
        ee.Image(cloud_prop),                   # attached cloud image
        ee.Image.constant(0).rename('probability')  # fallback: zeros
    ))
    cloud_prob_band = ee.Image(cloud_img).select('probability')

    # build cloud mask
    cloud_mask = cloud_prob_band.lt(CLOUD_PROB_TH)

    # compute NDVI (SR bands)
    ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')

    # add time band as float for regression
    time_millis = ee.Image.constant(ee.Date(img.get('system:time_start')).millis()).float().rename('time')

    # return image with NDVI and time, masked by cloud mask
    return img.addBands([ndvi, time_millis]).updateMask(cloud_mask)

prepared = s2_joined.map(prepare)

# ensure we have images
n = prepared.size().getInfo()
if n == 0:
    raise RuntimeError('No cloud-free images after cloud masking in specified period.')

# ---------- 3) median composite for vegetation mask ----------
median = prepared.median().clip(FIELD)

# NDVI and NDWI on composite
ndvi_comp = median.select('NDVI')
ndwi_comp = median.normalizedDifference(['B3', 'B8']).rename('NDWI')

# vegetation raw test
veg_raw = ndvi_comp.gte(NDVI_MIN).And(ndwi_comp.lt(NDWI_MAX))

# clean the vegetation mask
veg_img = veg_raw.updateMask(veg_raw).rename('veg').toByte()
veg_open = veg_img.focal_min(radius=1, units='pixels').focal_max(radius=1, units='pixels')
connected = veg_open.connectedPixelCount(maxSize=1024, eightConnected=True)
veg_clean = veg_open.updateMask(connected.gte(MIN_CONNECTED_PIXELS)).rename('veg_clean')

# ---------- 4) compute NDVI slope (use the prepared collection with NDVI + time) ----------
# linear fit reducer across images: expects bands 'time' and 'NDVI'
regression = prepared.select(['time', 'NDVI']).reduce(ee.Reducer.linearFit())
# scale to NDVI per day
slope_per_day = regression.select('scale').multiply(1000 * 60 * 60 * 24).rename('slope_per_day')

# ---------- 5) apply vegetation mask to slope (keep slope only where veg_clean==1) ----------
slope_masked = slope_per_day.updateMask(veg_clean)

# ---------- 6) visualize slope (dying -> red, stable->yellow, improving->green) ----------
slope_vis = slope_masked.unitScale(DYING_THRESH, HEALTHY_THRESH).clamp(0,1)
# convert to RGB visualization: palette from red->yellow->green
slope_rgb = slope_vis.visualize(palette=['ff0000','ffff00','00ff00'])

# optionally, make non-vegetation transparent by masking slope_rgb with veg_clean
# Create an RGB image forced to be RGB (3 bands). Then mask.
slope_rgb_masked = ee.Image(ee.Algorithms.If(
    slope_rgb, 
    ee.Image(slope_rgb).updateMask(veg_clean), 
    ee.Image(slope_rgb).updateMask(veg_clean)
)).copyProperties(slope_rgb)

# ---------- 7) Export the final visualized health image as GeoTIFF to Drive ----------
task = ee.batch.Export.image.toDrive(
    image=ee.Image(slope_rgb_masked),
    description='veg_health_masked_summer2025_gtiff',
    folder='EarthEngine',
    fileNamePrefix='veg_health_masked_summer2025',
    region=FIELD,
    scale=SCALE,
    maxPixels=1e9,
    fileFormat='GEO_TIFF'
)
task.start()
print('Export task started: veg_health_masked_summer2025 (check Earth Engine Tasks and Drive/EarthEngine).')

# polygon_coords = [
#     [56.73639949411154, 37.46023172459497],
#     [56.738184839487076, 37.46023172459497],
#     [56.738184839487076, 37.45891727703601],
#     [56.73639949411154, 37.45891727703601],
# ]

# # Step 1: Find center
# center_lon = sum([p[0] for p in polygon_coords]) / len(polygon_coords)
# center_lat = sum([p[1] for p in polygon_coords]) / len(polygon_coords)

# # Step 2: Scale factor
# scale = 4

# # Step 3: Create new scaled polygon
# scaled_polygon = []
# for lon, lat in polygon_coords:
#     new_lon = center_lon + (lon - center_lon) * scale
#     new_lat = center_lat + (lat - center_lat) * scale
#     scaled_polygon.append([new_lon, new_lat])

# print(scaled_polygon)

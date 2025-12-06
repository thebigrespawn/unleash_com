from flask import Flask, request, jsonify, send_file
import ee
import io
import numpy as np
from PIL import Image
import requests
from datetime import date, timedelta

# ------------------------
# 0. Initialize Earth Engine
# ------------------------
SERVICE_ACCOUNT = 'unleash-treeger@key-scarab-332210.iam.gserviceaccount.com'
KEY_FILE = 'key-scarab-332210-fe55f40b5089.json'

credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_FILE)
ee.Initialize(credentials)

# ------------------------
# 1. Flask app
# ------------------------
app = Flask(__name__)

# ---------- CONFIG ----------
SCALE = 10                # full Sentinel-2 resolution
CLOUD_PROB_TH = 30
NDVI_MIN = 0.35
NDWI_MAX = 0.05               # stricter water mask
BSI_MAX = 0.2                 # bare soil index threshold
MIN_CONNECTED_PIXELS = 10
CONNECTED_MAXSIZE = 256       # for faster connectedPixelCount
DEFAULT_MAX_IMAGES = None      # limit to best N images (None = no limit)

# ------------------------
# 2. Helpers
# ------------------------
def download_tif(image, region, scale):
    """Download EE image as temporary TIFF bytes. Uses tileScale to reduce memory pressure."""
    url = image.getDownloadURL({
        "scale": scale,
        "region": region.getInfo(),
        "format": "GEO_TIFF",
        "tileScale": 4
    })
    r = requests.get(url, timeout=180)
    r.raise_for_status()
    return io.BytesIO(r.content)


def severity_to_red_alpha_png(severity_bytes, veg_bytes):
    """Return RGBA PNG bytes for masked severity."""
    import rasterio
    with rasterio.MemoryFile(severity_bytes) as mem_sev, rasterio.MemoryFile(veg_bytes) as mem_veg:
        with mem_sev.open() as src_sev:
            severity = src_sev.read(1).astype(np.float32)
        with mem_veg.open() as src_veg:
            veg = src_veg.read(1).astype(np.uint8)

    severity = np.clip(severity, 0, 1)
    red = (severity * 255).astype(np.uint8)
    alpha = np.where(veg > 0, red, 0).astype(np.uint8)

    rgba = np.zeros((severity.shape[0], severity.shape[1], 4), dtype=np.uint8)
    rgba[:, :, 0] = red
    rgba[:, :, 3] = alpha

    buf = io.BytesIO()
    Image.fromarray(rgba, mode='RGBA').save(buf, format='PNG')
    buf.seek(0)
    return buf

# ------------------------
# 3. Main route
# ------------------------
@app.route('/compute_masks', methods=['POST'])
def compute_masks():
    data = request.json or {}
    coords = data.get("polygon")
    if not coords or len(coords) < 3:
        return jsonify({"error": "Polygon must have at least 3 points"}), 400

    max_images = data.get("max_images", DEFAULT_MAX_IMAGES)
    FIELD = ee.Geometry.Polygon([coords])
    END = data.get("end", date.today().isoformat())
    START = data.get("start", (date.today() - timedelta(days=7)).isoformat())

    # Load Sentinel-2 collections
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(FIELD)
          .filterDate(START, END)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 70))
          .sort("system:time_start"))

    if max_images is not None:
        s2 = s2.sort('CLOUDY_PIXEL_PERCENTAGE').limit(int(max_images))

    cloudp = (ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
              .filterBounds(FIELD)
              .filterDate(START, END))

    joined = ee.Join.saveFirst("cloud").apply(s2, cloudp, ee.Filter.equals("system:index", None, "system:index"))
    coll = ee.ImageCollection(joined)

    try:
        coll_count = coll.size().getInfo()
    except Exception as e:
        return jsonify({"error": "Failed to query collection size", "detail": str(e)}), 500
    if coll_count == 0:
        return jsonify({"error": "No Sentinel-2 images for the polygon/date range. Expand dates."}), 400

    # Prepare images: NDVI, NDWI, Bare Soil Index (BSI), cloud mask
    def prepare(img):
        img = ee.Image(img)
        cloud_img = ee.Image(img.get("cloud"))
        cmask = cloud_img.select("probability").lt(CLOUD_PROB_TH)

        ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI").float()
        ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI").float()
        bsi = ((img.select("B11").add(img.select("B4")))
               .subtract(img.select("B8").add(img.select("B2"))))
        bsi = bsi.divide(img.select("B11").add(img.select("B4").add(img.select("B8")).add(img.select("B2")))).rename("BSI")

        veg_mask = ndvi.gte(NDVI_MIN).And(ndwi.lt(NDWI_MAX)).And(bsi.lt(BSI_MAX))
        time = ee.Image.constant(ee.Date(img.get("system:time_start")).millis()).rename("time").toFloat()
        return ee.Image().addBands([ndvi, ndwi, bsi, time]).updateMask(cmask.And(veg_mask))

    prep = ee.ImageCollection(coll.map(prepare))

    med = prep.reduce(ee.Reducer.percentile([50])).clip(FIELD)
    veg_raw = med.select("NDVI_p50").gte(NDVI_MIN).And(med.select("NDWI_p50").lt(NDWI_MAX))


    veg = veg_raw.updateMask(veg_raw).rename("veg").toByte()

    connected = veg_raw.connectedPixelCount(CONNECTED_MAXSIZE, True)
    veg_clean = veg_raw.updateMask(connected.gte(50)).rename("veg_clean")


    # NDVI slope -> masked severity
    fit = prep.select(["time", "NDVI"]).reduce(ee.Reducer.linearFit())
    slope_day = fit.select("scale").multiply(1000 * 60 * 60 * 24)
    severity_masked = slope_day.clamp(-0.03, 0).multiply(-1).divide(0.03).updateMask(veg_clean).rename("severity_masked")

    try:
        veg_bytes = download_tif(veg_clean, FIELD, SCALE)
        severity_bytes = download_tif(severity_masked, FIELD, SCALE)
    except Exception as e:
        return jsonify({"error": "Failed to download image from GEE", "detail": str(e)}), 500

    severity_png = severity_to_red_alpha_png(severity_bytes, veg_bytes)

    return send_file(
        severity_png,
        mimetype='image/png',
        as_attachment=True,
        download_name='severity_red_alpha.png'
    )


# ------------------------
# 4. Run server
# ------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050)

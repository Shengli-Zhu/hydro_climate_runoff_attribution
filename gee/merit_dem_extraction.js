// =============================================================================
// MERIT DEM Export for Three Study Countries
// Dataset: MERIT/DEM/v1_0_3 (Yamazaki et al. 2017)
// Resolution: 3 arc-seconds (~90 m)
// Countries: Saudi Arabia (arid), Italy (transition), Bangladesh (humid)
// Output: Single-band GeoTIFF per country (elevation in metres)
// =============================================================================

// ---------------------- 1. Load MERIT DEM ----------------------
var meritDem = ee.Image('MERIT/DEM/v1_0_3').select('dem');

// ---------------------- 2. Country Boundaries (same as ERA5-Land script) ----------------------
var countries = ee.FeatureCollection('FAO/GAUL/2015/level0');
var saudi     = countries.filter(ee.Filter.eq('ADM0_NAME', 'Saudi Arabia'));
var italy     = countries.filter(ee.Filter.eq('ADM0_NAME', 'Italy'));
var bangladesh = countries.filter(ee.Filter.eq('ADM0_NAME', 'Bangladesh'));

// ---------------------- 3. Visualise (optional preview in Code Editor) ----------------------
var visParams = {min: 0, max: 3000, palette: ['#006994', '#228B22', '#8B4513', '#FFFFFF']};
Map.addLayer(meritDem.clip(saudi.geometry()),      visParams, 'Saudi DEM');
Map.addLayer(meritDem.clip(italy.geometry()),      visParams, 'Italy DEM');
Map.addLayer(meritDem.clip(bangladesh.geometry()), visParams, 'Bangladesh DEM');
Map.centerObject(italy, 5);

// ---------------------- 4. Export Function ----------------------
var exportDem = function(region, regionName) {
  var clipped = meritDem.clip(region.geometry());

  Export.image.toDrive({
    image: clipped,
    description: regionName + '_DEM',
    folder: 'GEE_DEM',
    fileNamePrefix: regionName + '_DEM',
    region: region.geometry().bounds(),
    scale: 90,           // native MERIT DEM resolution (~3 arc-seconds)
    crs: 'EPSG:4326',
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });
  print('Export task submitted: ' + regionName + '_DEM.tif');
};

// ---------------------- 5. Submit Three Export Tasks ----------------------
exportDem(saudi,      'Saudi');
exportDem(italy,      'Italy');
exportDem(bangladesh, 'Bangladesh');

print('Done. Check Tasks panel → Run all three.');
print('Download GeoTIFFs from Google Drive folder "GEE_DEM"');
print('Place them at: data/DEM/Saudi_DEM.tif, Italy_DEM.tif, Bangladesh_DEM.tif');

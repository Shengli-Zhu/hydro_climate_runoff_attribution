// =============================================================================
// ERA5-Land Monthly Data Extraction for Three Countries
// Dataset: ECMWF/ERA5_LAND/MONTHLY_AGGR
// Period: 2000-01 to 2025-03
// Countries: Saudi Arabia (arid), Italy (transition), Bangladesh (humid)
// =============================================================================

// ---------------------- 1. Load ERA5-Land Monthly Dataset ----------------------
var era5land = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
  .filterDate('2000-01-01', '2025-04-01');

// ---------------------- 2. Define Country Boundaries ----------------------
var countries = ee.FeatureCollection('FAO/GAUL/2015/level0');
var saudi = countries.filter(ee.Filter.eq('ADM0_NAME', 'Saudi Arabia'));
var italy = countries.filter(ee.Filter.eq('ADM0_NAME', 'Italy'));
var bangladesh = countries.filter(ee.Filter.eq('ADM0_NAME', 'Bangladesh'));

// ---------------------- 3. Variable Extraction & Unit Conversion ----------------------
var processMonth = function(image) {
  // Precipitation: m -> mm
  var P = image.select('total_precipitation_sum').multiply(1000).rename('P_mm');

  // Evapotranspiration: m -> mm (ERA5 stores ET as negative, take absolute)
  var ET = image.select('total_evaporation_sum').multiply(-1000).rename('ET_mm');

  // Surface runoff: m -> mm
  var R_sro = image.select('surface_runoff_sum').multiply(1000).rename('R_sro_mm');

  // Subsurface runoff: m -> mm
  var R_ssro = image.select('sub_surface_runoff_sum').multiply(1000).rename('R_ssro_mm');

  // Total runoff
  var R = R_sro.add(R_ssro).rename('R_mm');

  // Soil water storage (volumetric -> mm equivalent depth)
  var S1 = image.select('volumetric_soil_water_layer_1').multiply(70);   // 0-7 cm
  var S2 = image.select('volumetric_soil_water_layer_2').multiply(210);  // 7-28 cm
  var S3 = image.select('volumetric_soil_water_layer_3').multiply(720);  // 28-100 cm
  var S4 = image.select('volumetric_soil_water_layer_4').multiply(1890); // 100-289 cm
  var S = S1.add(S2).add(S3).add(S4).rename('S_mm');

  // Temperature: K -> Celsius
  var T = image.select('temperature_2m').subtract(273.15).rename('T_C');
  var Td = image.select('dewpoint_temperature_2m').subtract(273.15).rename('Td_C');

  // Radiation (keep original units: J/m^2)
  var Rn_sw = image.select('surface_net_solar_radiation_sum').rename('Rn_sw');
  var Rn_lw = image.select('surface_net_thermal_radiation_sum').rename('Rn_lw');

  // Wind speed: compute magnitude from u,v components (m/s)
  var u = image.select('u_component_of_wind_10m');
  var v = image.select('v_component_of_wind_10m');
  var wind = u.pow(2).add(v.pow(2)).sqrt().rename('Wind');

  // Soil temperature: K -> Celsius
  var Ts = image.select('soil_temperature_level_1').subtract(273.15).rename('Ts_C');

  // Surface pressure (Pa)
  var SP = image.select('surface_pressure').rename('SP');

  return image.addBands([P, ET, R_sro, R_ssro, R, S, T, Td, Rn_sw, Rn_lw, wind, Ts, SP]);
};

var processed = era5land.map(processMonth);

// ---------------------- 4. Extract Area-Weighted Mean Time Series ----------------------
var bandNames = ['P_mm', 'ET_mm', 'R_sro_mm', 'R_ssro_mm', 'R_mm', 'S_mm',
                 'T_C', 'Td_C', 'Rn_sw', 'Rn_lw', 'Wind', 'Ts_C', 'SP'];

var extractTimeSeries = function(region, regionName) {
  return processed.map(function(image) {
    var stats = image.select(bandNames)
      .reduceRegion({
        reducer: ee.Reducer.mean(),
        geometry: region.geometry(),
        scale: 11132,
        bestEffort: true
      });
    return ee.Feature(null, stats)
      .set('date', image.date().format('YYYY-MM'))
      .set('country', regionName);
  });
};

var saudi_ts = extractTimeSeries(saudi, 'Saudi_Arabia');
var italy_ts = extractTimeSeries(italy, 'Italy');
var bangladesh_ts = extractTimeSeries(bangladesh, 'Bangladesh');

// ---------------------- 5. Export to Google Drive as CSV ----------------------
Export.table.toDrive({
  collection: saudi_ts,
  description: 'Saudi_ERA5Land_Monthly',
  folder: 'GEE_ERA5Land_Export',
  fileNamePrefix: 'Saudi_ERA5Land_Monthly',
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: italy_ts,
  description: 'Italy_ERA5Land_Monthly',
  folder: 'GEE_ERA5Land_Export',
  fileNamePrefix: 'Italy_ERA5Land_Monthly',
  fileFormat: 'CSV'
});

Export.table.toDrive({
  collection: bangladesh_ts,
  description: 'Bangladesh_ERA5Land_Monthly',
  folder: 'GEE_ERA5Land_Export',
  fileNamePrefix: 'Bangladesh_ERA5Land_Monthly',
  fileFormat: 'CSV'
});

// ---------------------- 6. Export Annual Totals as GeoTIFF (for spatial trend maps) ----------------------
var years = ee.List.sequence(2000, 2024);

var annualImages = function(region, regionName, varName) {
  return years.map(function(y) {
    var startDate = ee.Date.fromYMD(y, 1, 1);
    var endDate = ee.Date.fromYMD(ee.Number(y).add(1), 1, 1);
    var annual = processed.filterDate(startDate, endDate)
      .select(varName).sum()
      .clip(region.geometry());
    return annual.set('year', y).set('system:time_start', startDate.millis());
  });
};

// Export annual P, ET, R for each country (for pixel-level MK trend analysis)
var exportAnnual = function(region, regionName, varName) {
  var collection = ee.ImageCollection.fromImages(annualImages(region, regionName, varName));
  var stack = collection.toBands();
  Export.image.toDrive({
    image: stack,
    description: regionName + '_Annual_' + varName,
    folder: 'GEE_ERA5Land_Export',
    fileNamePrefix: regionName + '_Annual_' + varName,
    region: region.geometry(),
    scale: 11132,
    maxPixels: 1e13
  });
};

// Export for spatial trend maps (Section 7.4)
var varsToExport = ['P_mm', 'ET_mm', 'R_mm'];
var regions = [
  {region: saudi, name: 'Saudi'},
  {region: italy, name: 'Italy'},
  {region: bangladesh, name: 'Bangladesh'}
];

regions.forEach(function(r) {
  varsToExport.forEach(function(v) {
    exportAnnual(r.region, r.name, v);
  });
});

print('All exports configured. Run tasks in the GEE Tasks tab.');

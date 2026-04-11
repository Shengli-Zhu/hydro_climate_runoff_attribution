// =============================================================================
// ERA5-Land Monthly Data Extraction for Three Countries
// Dataset: ECMWF/ERA5_LAND/MONTHLY_AGGR
// Period: 1950-02 to 2025-12
// Countries: Saudi Arabia (arid), Italy (transition), Bangladesh (humid)
// Output: Monthly GeoTIFF stacks (~911 bands) + Annual GeoTIFF stacks (76 bands)
// =============================================================================

// ---------------------- 1. Load ERA5-Land Monthly Dataset ----------------------
var era5land = ee.ImageCollection('ECMWF/ERA5_LAND/MONTHLY_AGGR')
  .filterDate('1950-02-01', '2026-01-01');  // end is exclusive, so this captures through 2025-12

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

// ---------------------- 4. Verify system:index format (run first, check console) ----------------------
// Expected output: "200001" (YYYYMM)
// If format differs, adjust slice indices in exportMonthlyStack below.
print('Total images:', processed.size());
print('system:index sample:', processed.first().getString('system:index'));

// ---------------------- 5. Monthly GeoTIFF Stack Export (312 bands per file) ----------------------
// Each file: one variable, one country, 312 bands (2000_01 ... 2025_12)
// Pixel outside country boundary = nodata

var exportMonthlyStack = function(region, regionName, varName) {
  // Build YYYY_MM band labels from system:index (e.g. "200001" -> "2000_01")
  var bandList = ee.List(processed.aggregate_array('system:index')).map(function(idx) {
    idx = ee.String(idx);
    return idx.slice(0, 4).cat('_').cat(idx.slice(4, 6));
  });

  var renamed = processed.map(function(image) {
    var idx = image.getString('system:index');
    var label = idx.slice(0, 4).cat('_').cat(idx.slice(4, 6));
    return image.select([varName]).rename([label]);
  });

  var stack = renamed.toBands().rename(bandList).clip(region.geometry());

  Export.image.toDrive({
    image: stack,
    description: regionName + '_Monthly_' + varName,
    folder: 'GEE_ERA5Land_Monthly',
    fileNamePrefix: regionName + '_Monthly_' + varName,
    region: region.geometry().bounds(),
    scale: 11132,
    crs: 'EPSG:4326',
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });
};

// ---------------------- 6. Annual GeoTIFF Stack Export (26 bands per file) ----------------------
// Flux/accumulation variables -> .sum() for annual total
// State/intensive variables   -> .mean() for annual average
// Note: monthly values are already in mm (P, ET, R) or monthly-mean (T, S, Wind)
// Annual P/ET/R = sum of 12 monthly mm values = annual total mm
// Annual T/S    = mean of 12 monthly values

var SUM_VARS  = ['P_mm', 'ET_mm', 'R_sro_mm', 'R_ssro_mm', 'R_mm', 'Rn_sw', 'Rn_lw'];

var exportAnnualStack = function(region, regionName, varName) {
  var years = ee.List.sequence(1950, 2025);

  var annualCollection = ee.ImageCollection.fromImages(
    years.map(function(y) {
      var start = ee.Date.fromYMD(y, 1, 1);
      var end   = ee.Date.fromYMD(ee.Number(y).add(1), 1, 1);
      var monthly = processed.filterDate(start, end).select(varName);
      var annual = (SUM_VARS.indexOf(varName) >= 0) ? monthly.sum() : monthly.mean();
      return annual.set('system:index', ee.Number(y).int().format('%d'));
    })
  );

  var yearLabels = ee.List.sequence(1950, 2025).map(function(y) {
    return ee.Number(y).int().format('%d');
  });
  var stack = annualCollection.toBands().rename(yearLabels).clip(region.geometry());

  Export.image.toDrive({
    image: stack,
    description: regionName + '_Annual_' + varName,
    folder: 'GEE_ERA5Land_Annual',
    fileNamePrefix: regionName + '_Annual_' + varName,
    region: region.geometry().bounds(),
    scale: 11132,
    crs: 'EPSG:4326',
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });
};

// ---------------------- 7. Trigger All Exports ----------------------
var regions = [
  {region: saudi,      name: 'Saudi'},
  {region: italy,      name: 'Italy'},
  {region: bangladesh, name: 'Bangladesh'}
];

// Monthly exports: 3 countries x 13 variables = 39 tasks
// Outputs: Saudi_Monthly_P_mm.tif, Italy_Monthly_T_C.tif, etc.
var allVars = ['P_mm', 'ET_mm', 'R_sro_mm', 'R_ssro_mm', 'R_mm', 'S_mm',
               'T_C', 'Td_C', 'Rn_sw', 'Rn_lw', 'Wind', 'Ts_C', 'SP'];

regions.forEach(function(r) {
  allVars.forEach(function(v) {
    exportMonthlyStack(r.region, r.name, v);
  });
});

// Annual exports (spatial trend maps): 3 countries x 5 variables = 15 tasks
// Outputs: Saudi_Annual_P_mm.tif, Italy_Annual_T_C.tif, etc.
var annualVars = ['P_mm', 'ET_mm', 'R_mm', 'T_C', 'S_mm'];

regions.forEach(function(r) {
  annualVars.forEach(function(v) {
    exportAnnualStack(r.region, r.name, v);
  });
});

print('Export tasks configured: 39 monthly (~911 bands each) + 15 annual (76 bands each) = 54 total.');
print('Tip: submit Bangladesh tasks first (smallest files), then Italy, then Saudi.');
print('Period: 1950-02 to 2025-12 (~911 months)');

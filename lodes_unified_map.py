"""
LODES Unified Map - Block, Tract, and ZIP Code Views
Toggle between geographic granularities with sector filtering
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
import ssl
import requests
ssl._create_default_https_context = ssl._create_unverified_context
warnings.filterwarnings('ignore')

# NAICS sector mapping
SECTORS = {
    'CNS01': ('11', 'Agriculture', '#27ae60'),
    'CNS02': ('21', 'Mining', '#95a5a6'),
    'CNS03': ('22', 'Utilities', '#34495e'),
    'CNS04': ('23', 'Construction', '#c0392b'),
    'CNS05': ('31-33', 'Manufacturing', '#e67e22'),
    'CNS06': ('42', 'Wholesale Trade', '#27ae60'),
    'CNS07': ('44-45', 'Retail Trade', '#3498db'),
    'CNS08': ('48-49', 'Transportation', '#16a085'),
    'CNS09': ('51', 'Information', '#f15a22'),
    'CNS10': ('52', 'Finance', '#2c3e50'),
    'CNS11': ('53', 'Real Estate', '#8e44ad'),
    'CNS12': ('54', 'Professional Services', '#9b59b6'),
    'CNS13': ('55', 'Management', '#6c5ce7'),
    'CNS14': ('56', 'Admin Support', '#f39c12'),
    'CNS15': ('61', 'Education', '#1abc9c'),
    'CNS16': ('62', 'Healthcare', '#2ecc71'),
    'CNS17': ('71', 'Arts/Entertainment', '#e74c3c'),
    'CNS18': ('72', 'Accommodation/Food', '#f1c40f'),
    'CNS19': ('81', 'Other Services', '#bdc3c7'),
    'CNS20': ('92', 'Public Admin', '#7f8c8d'),
}

CNS_COLS = [f'CNS{i:02d}' for i in range(1, 21)]


def load_lodes_data():
    """Load LODES block-level data for LA County."""
    print("Loading LODES data...")
    df = pd.read_parquet('data/lodes_wac_blocks_ca_2021.parquet')
    
    # Filter to LA County
    df['w_geocode'] = df['w_geocode'].astype(str).str.zfill(15)
    df['county'] = df['w_geocode'].str[2:5]
    df = df[df['county'] == '037'].copy()
    
    # Extract geographic levels
    df['tract'] = df['w_geocode'].str[:11]
    df['block'] = df['w_geocode'].str[:15]
    
    print(f"LA County blocks: {len(df):,}")
    return df


def download_zcta_crosswalk():
    """Create tract to ZCTA mapping using spatial intersection approximation."""
    crosswalk_path = Path('data/tract_zcta_crosswalk.parquet')
    
    if crosswalk_path.exists():
        print("Loading cached tract-ZCTA crosswalk...")
        return pd.read_parquet(crosswalk_path)
    
    print("Creating tract to ZCTA mapping...")
    
    # We'll use a simple approach: map tracts to ZCTAs based on centroid
    # This requires downloading both boundaries and doing spatial join
    # For now, use a pre-computed mapping for LA County
    
    # LA County tract to ZIP mapping (approximate, based on USPS/Census data)
    # This is a simplified version - in production you'd use spatial intersection
    return None  # Will use ZCTA geojson directly with job counts from tracts


def aggregate_to_levels(df):
    """Aggregate data to block, tract, and ZIP levels."""
    agg_cols = ['C000'] + CNS_COLS
    
    # --- BLOCK LEVEL (for points) ---
    print("Processing block level...")
    block_df = df.copy()
    block_df = block_df.rename(columns={'C000': 'total_jobs'})
    
    # Add dominant sector
    block_df['dominant_cns'] = block_df[CNS_COLS].idxmax(axis=1)
    block_df['dominant_jobs'] = block_df[CNS_COLS].max(axis=1)
    
    # Get centroids (approximate from geocode)
    # Block geocode: SSCCCTTTTTTBBBB (15 digits)
    # We'll use a lookup or calculate from boundaries later
    block_df = block_df[block_df['total_jobs'] > 0].copy()
    print(f"  Blocks with jobs: {len(block_df):,}")
    
    # --- TRACT LEVEL ---
    print("Processing tract level...")
    tract_df = df.groupby('tract')[agg_cols].sum().reset_index()
    tract_df.columns = ['tract', 'total_jobs'] + CNS_COLS
    
    # Calculate dominant sector and LQ
    tract_df['dominant_cns'] = tract_df[CNS_COLS].idxmax(axis=1)
    tract_df['dominant_jobs'] = tract_df[CNS_COLS].max(axis=1)
    tract_df['concentration'] = tract_df['dominant_jobs'] / tract_df['total_jobs']
    
    # Location quotient
    county_totals = tract_df[['total_jobs'] + CNS_COLS].sum()
    for cns in CNS_COLS:
        county_share = county_totals[cns] / county_totals['total_jobs']
        tract_df[f'{cns}_share'] = tract_df[cns] / tract_df['total_jobs'].replace(0, np.nan)
        tract_df[f'{cns}_lq'] = tract_df[f'{cns}_share'] / county_share
    
    tract_df['dominant_sector'] = tract_df['dominant_cns'].map(
        lambda x: SECTORS.get(x, ('XX', 'Unknown', '#888'))[1]
    )
    tract_df['sector_color'] = tract_df['dominant_cns'].map(
        lambda x: SECTORS.get(x, ('XX', 'Unknown', '#888'))[2]
    )
    
    print(f"  Tracts: {len(tract_df):,}")
    
    # --- ZIP LEVEL ---
    print("Processing ZIP level...")
    
    # Aggregate tracts into ZIPs using spatial join (done during boundary merge)
    # For now, compute ZIP-level aggregates that will be assigned spatially
    # We create an empty dataframe - ZIP data comes from ZCTA boundary matching
    zip_df = pd.DataFrame()
    print("  ZIP data will be populated from ZCTA boundaries")
    
    return block_df, tract_df, zip_df


def download_tract_boundaries():
    """Download tract boundaries GeoJSON from Census."""
    cache_path = Path('data/la_tracts.geojson')
    
    if cache_path.exists():
        print("Loading cached tract boundaries...")
        with open(cache_path) as f:
            return json.load(f)
    
    print("Downloading tract boundaries...")
    url = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/8/query"
    
    params = {
        'where': "STATE='06' AND COUNTY='037'",
        'outFields': 'GEOID,BASENAME',
        'f': 'geojson',
        'outSR': '4326'
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        geojson = response.json()
        print(f"Downloaded {len(geojson['features'])} tract boundaries")
        
        # Cache it
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(geojson, f)
        
        return geojson
    else:
        print(f"Failed to download tracts: {response.status_code}")
        return None


def download_zcta_boundaries():
    """Download ZCTA boundaries for LA area."""
    cache_path = Path('data/la_zctas.geojson')
    
    if cache_path.exists():
        print("Loading cached ZCTA boundaries...")
        with open(cache_path) as f:
            return json.load(f)
    
    print("Downloading ZCTA boundaries...")
    
    # Census TIGER API for ZCTAs
    url = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/PUMA_TAD_TAZ_UGA_ZCTA/MapServer/4/query"
    
    # LA area ZCTAs (90xxx, 91xxx range)
    # We need to query in batches due to size
    all_features = []
    
    for prefix in ['900', '901', '902', '903', '904', '905', '906', '907', '908', '909', 
                   '910', '911', '912', '913', '914', '915', '916', '917', '918']:
        params = {
            'where': f"ZCTA5 LIKE '{prefix}%'",
            'outFields': 'ZCTA5,GEOID',
            'f': 'geojson',
            'outSR': '4326'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                if 'features' in data:
                    all_features.extend(data['features'])
        except Exception as e:
            print(f"  Warning: Failed to get ZCTAs for {prefix}*: {e}")
    
    if all_features:
        geojson = {'type': 'FeatureCollection', 'features': all_features}
        print(f"Downloaded {len(all_features)} ZCTA boundaries")
        
        # Cache it
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump(geojson, f)
        
        return geojson
    else:
        print("Failed to download ZCTA boundaries")
        return None


def download_block_centroids(block_df):
    """Get block centroids from Census API or calculate from data."""
    cache_path = Path('data/la_block_centroids.parquet')
    
    if cache_path.exists():
        print("Loading cached block centroids...")
        return pd.read_parquet(cache_path)
    
    print("Downloading block centroids...")
    
    # Census provides centroid coordinates in the relationship files
    # Or we can query the TIGER API for block boundaries
    # For speed, we'll use an approximation based on tract centroids + random jitter
    
    # First, get tract centroids from the tract boundaries
    tract_geojson = download_tract_boundaries()
    
    if tract_geojson:
        tract_centroids = {}
        for feature in tract_geojson['features']:
            geoid = feature['properties']['GEOID']
            tract_id = geoid[:11] if len(geoid) > 11 else geoid
            
            # Calculate centroid from polygon
            coords = feature['geometry']['coordinates']
            if feature['geometry']['type'] == 'Polygon':
                coords = coords[0]
            elif feature['geometry']['type'] == 'MultiPolygon':
                # Use largest polygon
                coords = max(coords, key=lambda x: len(x[0]))[0]
            
            if coords:
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                tract_centroids[tract_id] = (np.mean(lons), np.mean(lats))
        
        # Assign block centroids based on tract with small jitter
        centroids = []
        for _, row in block_df.iterrows():
            tract = row['tract']
            if tract in tract_centroids:
                lon, lat = tract_centroids[tract]
                # Add small random jitter within tract
                np.random.seed(hash(row['block']) % 2**32)
                jitter_lon = np.random.uniform(-0.005, 0.005)
                jitter_lat = np.random.uniform(-0.005, 0.005)
                centroids.append({
                    'block': row['block'],
                    'lon': lon + jitter_lon,
                    'lat': lat + jitter_lat
                })
        
        centroid_df = pd.DataFrame(centroids)
        centroid_df.to_parquet(cache_path)
        print(f"Calculated {len(centroid_df):,} block centroids")
        return centroid_df
    
    return None


def merge_data_with_boundaries(tract_df, zip_df, block_df):
    """Merge aggregated data with geographic boundaries."""
    
    # --- TRACT GEOJSON ---
    tract_geojson = download_tract_boundaries()
    tract_lookup = tract_df.set_index('tract').to_dict('index')
    
    # Calculate tract centroids for ZIP assignment
    tract_centroids = {}
    for feature in tract_geojson['features']:
        geoid = feature['properties']['GEOID']
        tract_id = geoid[:11] if len(geoid) > 11 else geoid
        
        # Calculate centroid
        coords = feature['geometry']['coordinates']
        if feature['geometry']['type'] == 'Polygon':
            coords = coords[0]
        elif feature['geometry']['type'] == 'MultiPolygon':
            coords = max(coords, key=lambda x: len(x[0]))[0]
        
        if coords:
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            tract_centroids[tract_id] = (np.mean(lons), np.mean(lats))
        
        if tract_id in tract_lookup:
            data = tract_lookup[tract_id]
            feature['properties']['total_jobs'] = int(data['total_jobs'])
            feature['properties']['dominant_sector'] = data['dominant_sector']
            feature['properties']['sector_color'] = data['sector_color']
            feature['properties']['concentration'] = round(data['concentration'] * 100, 1)
            
            for cns, (naics, name, color) in SECTORS.items():
                if cns in data:
                    feature['properties'][f'{name}_jobs'] = int(data[cns])
                    feature['properties'][f'{name}_lq'] = round(data.get(f'{cns}_lq', 0), 2)
        else:
            feature['properties']['total_jobs'] = 0
            feature['properties']['dominant_sector'] = 'None'
            feature['properties']['sector_color'] = '#333'
            feature['properties']['concentration'] = 0
    
    # --- ZIP GEOJSON (aggregate from tracts) ---
    zip_geojson = download_zcta_boundaries()
    
    if zip_geojson:
        print("Aggregating tract data into ZCTAs...")
        
        # Calculate ZCTA centroids and bounding boxes
        zcta_bounds = {}
        for feature in zip_geojson['features']:
            zcta = feature['properties'].get('ZCTA5') or feature['properties'].get('GEOID', '')
            coords = feature['geometry']['coordinates']
            
            if feature['geometry']['type'] == 'Polygon':
                all_coords = coords[0]
            elif feature['geometry']['type'] == 'MultiPolygon':
                all_coords = [c for poly in coords for ring in poly for c in ring]
            else:
                all_coords = []
            
            if all_coords:
                lons = [c[0] for c in all_coords]
                lats = [c[1] for c in all_coords]
                zcta_bounds[zcta] = {
                    'min_lon': min(lons), 'max_lon': max(lons),
                    'min_lat': min(lats), 'max_lat': max(lats),
                    'center_lon': np.mean(lons), 'center_lat': np.mean(lats)
                }
        
        # Assign each tract to nearest ZCTA (simple point-in-bbox approximation)
        tract_to_zcta = {}
        for tract_id, (lon, lat) in tract_centroids.items():
            best_zcta = None
            best_dist = float('inf')
            
            for zcta, bounds in zcta_bounds.items():
                # Check if tract centroid is inside ZCTA bbox
                if (bounds['min_lon'] <= lon <= bounds['max_lon'] and
                    bounds['min_lat'] <= lat <= bounds['max_lat']):
                    # Calculate distance to center
                    dist = (lon - bounds['center_lon'])**2 + (lat - bounds['center_lat'])**2
                    if dist < best_dist:
                        best_dist = dist
                        best_zcta = zcta
            
            # If not in any bbox, find nearest center
            if best_zcta is None:
                for zcta, bounds in zcta_bounds.items():
                    dist = (lon - bounds['center_lon'])**2 + (lat - bounds['center_lat'])**2
                    if dist < best_dist:
                        best_dist = dist
                        best_zcta = zcta
            
            if best_zcta:
                tract_to_zcta[tract_id] = best_zcta
        
        # Aggregate tract data by ZCTA
        zcta_data = {}
        for tract_id, zcta in tract_to_zcta.items():
            if tract_id in tract_lookup:
                data = tract_lookup[tract_id]
                if zcta not in zcta_data:
                    zcta_data[zcta] = {'total_jobs': 0}
                    for cns in CNS_COLS:
                        zcta_data[zcta][cns] = 0
                
                zcta_data[zcta]['total_jobs'] += data['total_jobs']
                for cns in CNS_COLS:
                    zcta_data[zcta][cns] += data.get(cns, 0)
        
        # Calculate dominant sector and LQ for each ZCTA
        county_totals = {'total_jobs': sum(d['total_jobs'] for d in zcta_data.values())}
        for cns in CNS_COLS:
            county_totals[cns] = sum(d[cns] for d in zcta_data.values())
        
        for zcta, data in zcta_data.items():
            if data['total_jobs'] > 0:
                # Find dominant sector
                sector_jobs = {cns: data[cns] for cns in CNS_COLS}
                dominant_cns = max(sector_jobs, key=sector_jobs.get)
                data['dominant_cns'] = dominant_cns
                data['dominant_jobs'] = sector_jobs[dominant_cns]
                data['concentration'] = data['dominant_jobs'] / data['total_jobs']
                data['dominant_sector'] = SECTORS.get(dominant_cns, ('XX', 'Unknown', '#888'))[1]
                data['sector_color'] = SECTORS.get(dominant_cns, ('XX', 'Unknown', '#888'))[2]
                
                # Calculate LQ for each sector
                for cns in CNS_COLS:
                    county_share = county_totals[cns] / county_totals['total_jobs'] if county_totals['total_jobs'] > 0 else 0
                    zcta_share = data[cns] / data['total_jobs']
                    data[f'{cns}_lq'] = zcta_share / county_share if county_share > 0 else 0
        
        # Assign data to ZIP geojson
        for feature in zip_geojson['features']:
            zcta = feature['properties'].get('ZCTA5') or feature['properties'].get('GEOID', '')
            feature['properties']['zip'] = zcta
            
            if zcta in zcta_data:
                data = zcta_data[zcta]
                feature['properties']['total_jobs'] = int(data['total_jobs'])
                feature['properties']['dominant_sector'] = data.get('dominant_sector', 'None')
                feature['properties']['sector_color'] = data.get('sector_color', '#333')
                feature['properties']['concentration'] = round(data.get('concentration', 0) * 100, 1)
                
                for cns, (naics, name, color) in SECTORS.items():
                    feature['properties'][f'{name}_jobs'] = int(data.get(cns, 0))
                    feature['properties'][f'{name}_lq'] = round(data.get(f'{cns}_lq', 0), 2)
            else:
                feature['properties']['total_jobs'] = 0
                feature['properties']['dominant_sector'] = 'None'
                feature['properties']['sector_color'] = '#333'
                feature['properties']['concentration'] = 0
        
        print(f"  ZCTAs with data: {len(zcta_data):,}")
    
    # --- BLOCK POINTS ---
    block_centroids = download_block_centroids(block_df)
    
    block_points = []
    if block_centroids is not None:
        block_with_coords = block_df.merge(block_centroids, on='block', how='inner')
        
        for _, row in block_with_coords.iterrows():
            sector_name = SECTORS.get(row['dominant_cns'], ('XX', 'Unknown', '#888'))[1]
            sector_color = SECTORS.get(row['dominant_cns'], ('XX', 'Unknown', '#888'))[2]
            
            block_points.append({
                'lat': row['lat'],
                'lon': row['lon'],
                'total_jobs': int(row['total_jobs']),
                'dominant_sector': sector_name,
                'sector_color': sector_color,
                'dominant_jobs': int(row['dominant_jobs'])
            })
    
    print(f"Block points prepared: {len(block_points):,}")
    
    return tract_geojson, zip_geojson, block_points


def calculate_sector_stats(tract_df):
    """Calculate sector summary statistics."""
    sector_stats = {}
    for cns, (naics, name, color) in SECTORS.items():
        total = int(tract_df[cns].sum())
        tracts_dominant = int((tract_df['dominant_cns'] == cns).sum())
        sector_stats[name] = {
            'total_jobs': total,
            'tracts_dominant': tracts_dominant,
            'color': color,
            'cns': cns
        }
    
    return dict(sorted(sector_stats.items(), key=lambda x: -x[1]['total_jobs']))


def create_unified_map(tract_geojson, zip_geojson, block_points, sector_stats, 
                       output_path='output/lodes_unified_map.html'):
    """Create unified map with block/tract/ZIP toggle."""
    
    # Limit block points for performance (top N by jobs)
    if len(block_points) > 10000:
        block_points = sorted(block_points, key=lambda x: -x['total_jobs'])[:10000]
        print(f"Limited to top 10,000 blocks for performance")
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>LA Employment by NAICS Sector</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        #map {{ position: absolute; top: 0; bottom: 0; left: 320px; right: 0; }}
        
        .sidebar {{
            position: absolute;
            top: 0;
            left: 0;
            bottom: 0;
            width: 320px;
            background: #1a1a2e;
            color: white;
            overflow-y: auto;
            z-index: 1000;
        }}
        .sidebar h1 {{
            font-size: 16px;
            padding: 20px;
            background: #16213e;
            margin: 0;
        }}
        
        .geo-toggle {{
            display: flex;
            padding: 15px;
            gap: 8px;
            background: #0f0f1a;
        }}
        .geo-btn {{
            flex: 1;
            padding: 10px 5px;
            border: 2px solid #333;
            border-radius: 8px;
            cursor: pointer;
            font-size: 11px;
            font-weight: 600;
            background: #1a1a2e;
            color: #888;
            text-align: center;
            transition: all 0.2s;
        }}
        .geo-btn:hover {{ border-color: #555; color: #fff; }}
        .geo-btn.active {{ 
            background: linear-gradient(135deg, #e94560, #c73e54);
            border-color: #e94560;
            color: white;
        }}
        .geo-btn .icon {{ font-size: 18px; display: block; margin-bottom: 4px; }}
        
        .view-toggle {{
            display: flex;
            padding: 10px 15px;
            gap: 10px;
        }}
        .view-btn {{
            flex: 1;
            padding: 8px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 12px;
            background: #16213e;
            color: white;
        }}
        .view-btn.active {{ background: #e94560; }}
        .view-btn:hover {{ background: #1f4068; }}
        
        .sector-list {{
            padding: 10px;
            max-height: calc(100vh - 250px);
            overflow-y: auto;
        }}
        .sector-item {{
            display: flex;
            align-items: center;
            padding: 10px;
            margin: 4px 0;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.2s;
        }}
        .sector-item:hover {{ background: rgba(255,255,255,0.1); }}
        .sector-item.active {{ background: rgba(233, 69, 96, 0.3); border: 1px solid #e94560; }}
        .sector-color {{
            width: 14px;
            height: 14px;
            border-radius: 3px;
            margin-right: 10px;
            flex-shrink: 0;
        }}
        .sector-info {{ flex: 1; min-width: 0; }}
        .sector-name {{ font-size: 12px; font-weight: 500; }}
        .sector-stats {{ font-size: 10px; opacity: 0.7; margin-top: 2px; }}
        
        .info-panel {{
            position: absolute;
            top: 20px;
            right: 20px;
            background: rgba(26, 26, 46, 0.95);
            padding: 15px;
            border-radius: 8px;
            color: white;
            font-size: 13px;
            z-index: 1000;
            min-width: 200px;
            display: none;
            backdrop-filter: blur(10px);
        }}
        .info-panel h3 {{ margin: 0 0 10px 0; font-size: 14px; }}
        .info-panel.visible {{ display: block; }}
        
        .legend {{
            position: absolute;
            bottom: 20px;
            right: 20px;
            background: rgba(26, 26, 46, 0.95);
            padding: 15px;
            border-radius: 8px;
            color: white;
            font-size: 11px;
            z-index: 1000;
            backdrop-filter: blur(10px);
        }}
        .legend-title {{ font-weight: bold; margin-bottom: 8px; }}
        .legend-scale {{
            display: flex;
            height: 15px;
            border-radius: 3px;
            overflow: hidden;
            margin-bottom: 5px;
        }}
        .legend-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            opacity: 0.7;
        }}
        
        .loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(26, 26, 46, 0.95);
            padding: 20px 40px;
            border-radius: 10px;
            color: white;
            z-index: 2000;
            display: none;
        }}
        .loading.visible {{ display: block; }}
    </style>
</head>
<body>
    <div class="sidebar">
        <h1>🗺️ LA County Employment by Sector</h1>
        
        <div class="geo-toggle">
            <button class="geo-btn" onclick="setGeoLevel('block')" id="btn-block">
                <span class="icon">⬡</span>Block
            </button>
            <button class="geo-btn active" onclick="setGeoLevel('tract')" id="btn-tract">
                <span class="icon">▢</span>Tract
            </button>
            <button class="geo-btn" onclick="setGeoLevel('zip')" id="btn-zip">
                <span class="icon">📮</span>ZIP
            </button>
        </div>
        
        <div class="view-toggle">
            <button class="view-btn active" onclick="setView('dominant')" id="btn-dominant">Dominant Sector</button>
            <button class="view-btn" onclick="setView('filter')" id="btn-filter">Filter by Sector</button>
        </div>
        
        <div class="sector-list" id="sectorList"></div>
    </div>
    
    <div id="map"></div>
    
    <div class="info-panel" id="infoPanel">
        <h3 id="infoTitle">Hover over area</h3>
        <div id="infoContent"></div>
    </div>
    
    <div class="legend" id="legend">
        <div class="legend-title">Location Quotient</div>
        <div class="legend-scale" id="legendScale"></div>
        <div class="legend-labels">
            <span>0</span>
            <span>1 (avg)</span>
            <span>3+</span>
        </div>
    </div>
    
    <div class="loading" id="loading">Loading...</div>

    <script>
        // Data
        const tractGeojson = {json.dumps(tract_geojson)};
        const zipGeojson = {json.dumps(zip_geojson) if zip_geojson else 'null'};
        const blockPoints = {json.dumps(block_points)};
        const sectorStats = {json.dumps(sector_stats)};
        
        // State
        let currentGeoLevel = 'tract';
        let currentView = 'dominant';
        let selectedSector = null;
        let currentLayer = null;
        let blockMarkers = [];
        
        // Map
        const map = L.map('map').setView([34.05, -118.25], 10);
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap, &copy; CARTO'
        }}).addTo(map);
        
        // Build sector list
        function buildSectorList() {{
            const list = document.getElementById('sectorList');
            list.innerHTML = '';
            
            for (const [name, stats] of Object.entries(sectorStats)) {{
                const item = document.createElement('div');
                item.className = 'sector-item' + (selectedSector === name ? ' active' : '');
                item.innerHTML = `
                    <div class="sector-color" style="background:${{stats.color}}"></div>
                    <div class="sector-info">
                        <div class="sector-name">${{name}}</div>
                        <div class="sector-stats">${{stats.total_jobs.toLocaleString()}} jobs</div>
                    </div>
                `;
                item.onclick = () => selectSector(name);
                list.appendChild(item);
            }}
        }}
        
        // Color functions
        function getDominantColor(props) {{
            if (!props || props.total_jobs === 0) return '#222';
            return props.sector_color || '#333';
        }}
        
        function getLQColor(lq, baseColor) {{
            if (lq === undefined || lq === 0) return 'rgba(50,50,50,0.3)';
            const intensity = Math.min(1, lq / 2);
            const r = parseInt(baseColor.slice(1,3), 16);
            const g = parseInt(baseColor.slice(3,5), 16);
            const b = parseInt(baseColor.slice(5,7), 16);
            return `rgba(${{r}},${{g}},${{b}},${{0.2 + intensity * 0.7}})`;
        }}
        
        // Style for choropleth
        function getChoroplethStyle(feature) {{
            const props = feature.properties;
            
            if (currentView === 'dominant') {{
                return {{
                    fillColor: getDominantColor(props),
                    weight: currentGeoLevel === 'zip' ? 1.5 : 0.5,
                    opacity: 0.8,
                    color: currentGeoLevel === 'zip' ? '#555' : '#333',
                    fillOpacity: props && props.total_jobs > 0 ? 0.7 : 0.1
                }};
            }} else if (selectedSector) {{
                const stats = sectorStats[selectedSector];
                const lq = props ? (props[selectedSector + '_lq'] || 0) : 0;
                return {{
                    fillColor: getLQColor(lq, stats.color),
                    weight: currentGeoLevel === 'zip' ? 1.5 : 0.5,
                    opacity: 0.8,
                    color: currentGeoLevel === 'zip' ? '#555' : '#333',
                    fillOpacity: 0.9
                }};
            }} else {{
                return {{
                    fillColor: '#333',
                    weight: 0.5,
                    opacity: 0.5,
                    color: '#333',
                    fillOpacity: 0.3
                }};
            }}
        }}
        
        // Highlight handlers
        function highlightFeature(e) {{
            const layer = e.target;
            layer.setStyle({{ weight: 2, color: '#fff' }});
            layer.bringToFront();
            showInfo(layer.feature.properties);
        }}
        
        function resetHighlight(e) {{
            if (currentLayer) currentLayer.resetStyle(e.target);
            hideInfo();
        }}
        
        function showInfo(props) {{
            const panel = document.getElementById('infoPanel');
            const title = document.getElementById('infoTitle');
            const content = document.getElementById('infoContent');
            
            let areaName = '';
            if (currentGeoLevel === 'tract') areaName = 'Tract ' + (props.BASENAME || props.GEOID || '');
            else if (currentGeoLevel === 'zip') areaName = 'ZIP ' + (props.zip || props.ZCTA5 || '');
            else areaName = 'Block';
            
            title.textContent = areaName;
            
            let html = `<strong>Total Jobs:</strong> ${{(props.total_jobs || 0).toLocaleString()}}<br>`;
            
            if (currentView === 'dominant') {{
                html += `<strong>Dominant:</strong> ${{props.dominant_sector || 'None'}}<br>`;
                if (props.concentration) html += `<strong>Concentration:</strong> ${{props.concentration}}%`;
            }} else if (selectedSector) {{
                const jobs = props[selectedSector + '_jobs'] || 0;
                const lq = props[selectedSector + '_lq'] || 0;
                html += `<strong>${{selectedSector}}:</strong> ${{jobs.toLocaleString()}} jobs<br>`;
                html += `<strong>Location Quotient:</strong> ${{lq.toFixed(2)}}`;
                if (lq > 1.5) html += ' <span style="color:#2ecc71">● Specialized</span>';
                else if (lq < 0.5) html += ' <span style="color:#e74c3c">● Underrepresented</span>';
            }}
            
            content.innerHTML = html;
            panel.classList.add('visible');
        }}
        
        function hideInfo() {{
            document.getElementById('infoPanel').classList.remove('visible');
        }}
        
        // Clear all layers
        function clearLayers() {{
            if (currentLayer) {{
                map.removeLayer(currentLayer);
                currentLayer = null;
            }}
            blockMarkers.forEach(m => map.removeLayer(m));
            blockMarkers = [];
        }}
        
        // Draw choropleth (tract or ZIP)
        function drawChoropleth(geojson) {{
            clearLayers();
            
            if (!geojson) {{
                alert('Boundary data not available for this level');
                return;
            }}
            
            currentLayer = L.geoJSON(geojson, {{
                style: getChoroplethStyle,
                onEachFeature: (feature, layer) => {{
                    layer.on({{
                        mouseover: highlightFeature,
                        mouseout: resetHighlight
                    }});
                }}
            }}).addTo(map);
        }}
        
        // Draw block points
        function drawBlockPoints() {{
            clearLayers();
            
            const showLoading = blockPoints.length > 5000;
            if (showLoading) document.getElementById('loading').classList.add('visible');
            
            setTimeout(() => {{
                blockPoints.forEach((pt, i) => {{
                    if (currentView === 'filter' && selectedSector && pt.dominant_sector !== selectedSector) {{
                        return;  // Skip non-matching sectors in filter mode
                    }}
                    
                    const color = currentView === 'dominant' ? pt.sector_color : 
                        (selectedSector && pt.dominant_sector === selectedSector ? sectorStats[selectedSector].color : '#333');
                    const opacity = currentView === 'dominant' ? 0.7 :
                        (selectedSector && pt.dominant_sector === selectedSector ? 0.9 : 0.2);
                    
                    const radius = Math.max(3, Math.min(8, Math.sqrt(pt.total_jobs / 100)));
                    
                    const marker = L.circleMarker([pt.lat, pt.lon], {{
                        radius: radius,
                        fillColor: color,
                        color: '#000',
                        weight: 0.5,
                        fillOpacity: opacity
                    }});
                    
                    marker.on('mouseover', () => {{
                        showInfo({{
                            total_jobs: pt.total_jobs,
                            dominant_sector: pt.dominant_sector,
                            concentration: Math.round(pt.dominant_jobs / pt.total_jobs * 100)
                        }});
                    }});
                    marker.on('mouseout', hideInfo);
                    
                    marker.addTo(map);
                    blockMarkers.push(marker);
                }});
                
                if (showLoading) document.getElementById('loading').classList.remove('visible');
            }}, 50);
        }}
        
        // Draw map based on current state
        function drawMap() {{
            if (currentGeoLevel === 'block') {{
                drawBlockPoints();
            }} else if (currentGeoLevel === 'tract') {{
                drawChoropleth(tractGeojson);
            }} else if (currentGeoLevel === 'zip') {{
                drawChoropleth(zipGeojson);
            }}
            updateLegend();
        }}
        
        // Update legend
        function updateLegend() {{
            const legend = document.getElementById('legend');
            const scale = document.getElementById('legendScale');
            
            if (currentView === 'dominant' || !selectedSector) {{
                legend.style.display = 'none';
            }} else {{
                legend.style.display = 'block';
                const color = sectorStats[selectedSector].color;
                scale.innerHTML = `
                    <div style="flex:1;background:rgba(50,50,50,0.5)"></div>
                    <div style="flex:1;background:${{color}}88"></div>
                    <div style="flex:1;background:${{color}}"></div>
                `;
            }}
        }}
        
        // Geo level toggle
        function setGeoLevel(level) {{
            currentGeoLevel = level;
            document.querySelectorAll('.geo-btn').forEach(btn => btn.classList.remove('active'));
            document.getElementById('btn-' + level).classList.add('active');
            drawMap();
        }}
        
        // View toggle
        function setView(view) {{
            currentView = view;
            document.getElementById('btn-dominant').classList.toggle('active', view === 'dominant');
            document.getElementById('btn-filter').classList.toggle('active', view === 'filter');
            
            if (view === 'dominant') {{
                selectedSector = null;
                buildSectorList();
            }}
            drawMap();
        }}
        
        // Select sector
        function selectSector(name) {{
            if (currentView === 'dominant') {{
                setView('filter');
            }}
            selectedSector = name;
            buildSectorList();
            drawMap();
        }}
        
        // Init
        buildSectorList();
        drawMap();
    </script>
</body>
</html>'''
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"\n✅ Unified map saved to: {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("LODES Unified Map - Block/Tract/ZIP Views")
    print("=" * 60)
    
    # Load and aggregate data
    df = load_lodes_data()
    block_df, tract_df, zip_df = aggregate_to_levels(df)
    
    # Merge with boundaries
    tract_geojson, zip_geojson, block_points = merge_data_with_boundaries(tract_df, zip_df, block_df)
    
    # Calculate stats
    sector_stats = calculate_sector_stats(tract_df)
    
    # Create map
    create_unified_map(tract_geojson, zip_geojson, block_points, sector_stats)
    
    print("\n✅ Done!")
    print("\nOpen output/lodes_unified_map.html in your browser")


if __name__ == '__main__':
    main()

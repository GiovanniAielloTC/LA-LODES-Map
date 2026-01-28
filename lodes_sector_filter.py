"""
LODES Interactive Map with Sector Filter
Shows jobs by NAICS sector with dropdown selector
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
import ssl
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

def load_data():
    """Load and prepare LODES data."""
    print("Loading LODES data...")
    df = pd.read_parquet('data/lodes_wac_blocks_ca_2021.parquet')
    
    # Filter to LA County
    df['w_geocode'] = df['w_geocode'].astype(str).str.zfill(15)
    df['county'] = df['w_geocode'].str[2:5]
    df = df[df['county'] == '037'].copy()
    print(f"LA County blocks: {len(df):,}")
    
    # Get tract for coordinates
    df['tract'] = df['w_geocode'].str[:11]
    
    # Load tract centroids
    print("Loading tract centroids...")
    centroids = pd.read_csv("https://www2.census.gov/geo/docs/reference/cenpop2020/tract/CenPop2020_Mean_TR06.txt")
    centroids['STATEFP'] = centroids['STATEFP'].astype(str).str.zfill(2)
    centroids['COUNTYFP'] = centroids['COUNTYFP'].astype(str).str.zfill(3)
    centroids['TRACTCE'] = centroids['TRACTCE'].astype(str).str.zfill(6)
    centroids['tract'] = centroids['STATEFP'] + centroids['COUNTYFP'] + centroids['TRACTCE']
    centroids = centroids[centroids['COUNTYFP'] == '037'][['tract', 'LATITUDE', 'LONGITUDE']]
    
    df = df.merge(centroids, on='tract', how='left')
    
    # Jitter for block-level visualization
    np.random.seed(42)
    df['lat'] = df['LATITUDE'] + np.random.uniform(-0.002, 0.002, len(df))
    df['lon'] = df['LONGITUDE'] + np.random.uniform(-0.002, 0.002, len(df))
    
    # Filter to blocks with jobs and coordinates
    df = df[(df['C000'] > 0) & df['lat'].notna()].copy()
    print(f"Blocks with jobs: {len(df):,}")
    
    return df


def create_interactive_map(df, output_path='output/lodes_sector_filter.html'):
    """Create interactive HTML map with sector dropdown."""
    
    # Prepare data for each sector
    sector_data = {}
    for cns_code, (naics, name, color) in SECTORS.items():
        if cns_code in df.columns:
            sector_df = df[df[cns_code] > 0][['lat', 'lon', cns_code, 'w_geocode']].copy()
            sector_df = sector_df.rename(columns={cns_code: 'jobs'})
            # Sample if too large
            if len(sector_df) > 15000:
                sector_df = sector_df.nlargest(15000, 'jobs')
            sector_data[name] = {
                'color': color,
                'data': sector_df[['lat', 'lon', 'jobs']].values.tolist(),
                'total_jobs': int(df[cns_code].sum()),
                'block_count': int((df[cns_code] > 0).sum())
            }
    
    # Generate HTML
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>LA LODES Employment Map by NAICS Sector</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        #map {{ position: absolute; top: 60px; bottom: 0; width: 100%; }}
        .header {{
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 60px;
            background: #1a1a2e;
            color: white;
            display: flex;
            align-items: center;
            padding: 0 20px;
            z-index: 1000;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }}
        .header h1 {{
            margin: 0;
            font-size: 18px;
            font-weight: 500;
        }}
        .controls {{
            margin-left: 30px;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        select {{
            padding: 8px 15px;
            font-size: 14px;
            border: none;
            border-radius: 5px;
            background: #16213e;
            color: white;
            cursor: pointer;
        }}
        select:hover {{ background: #1f4068; }}
        .stats {{
            background: rgba(255,255,255,0.1);
            padding: 8px 15px;
            border-radius: 5px;
            font-size: 13px;
        }}
        .legend {{
            position: absolute;
            bottom: 30px;
            right: 20px;
            background: rgba(26, 26, 46, 0.95);
            padding: 15px;
            border-radius: 8px;
            color: white;
            font-size: 12px;
            z-index: 1000;
            max-height: 400px;
            overflow-y: auto;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 5px 0;
            cursor: pointer;
            padding: 3px 5px;
            border-radius: 3px;
        }}
        .legend-item:hover {{ background: rgba(255,255,255,0.1); }}
        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }}
        .info-box {{
            position: absolute;
            top: 80px;
            left: 20px;
            background: rgba(26, 26, 46, 0.95);
            padding: 15px;
            border-radius: 8px;
            color: white;
            font-size: 13px;
            z-index: 1000;
            max-width: 250px;
        }}
        .info-box h3 {{ margin: 0 0 10px 0; font-size: 14px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🗺️ LA County Employment by NAICS Sector</h1>
        <div class="controls">
            <select id="sectorSelect" onchange="changeSector()">
                <option value="ALL">All Sectors (Dominant)</option>
                {"".join(f'<option value="{name}">{name}</option>' for name in sector_data.keys())}
            </select>
            <div class="stats" id="stats">Select a sector</div>
        </div>
    </div>
    
    <div id="map"></div>
    
    <div class="info-box">
        <h3>📊 Data Source</h3>
        <p><strong>LEHD LODES 2021</strong><br>
        Workplace Area Characteristics<br>
        Block-level employment counts</p>
        <p style="font-size:11px; opacity:0.7;">Circle size = job count<br>Color = NAICS sector</p>
    </div>
    
    <div class="legend" id="legend">
        <strong>NAICS Sectors</strong>
        {"".join(f'''<div class="legend-item" onclick="selectSector('{name}')">
            <span class="legend-color" style="background:{info['color']}"></span>
            {name}
        </div>''' for name, info in sector_data.items())}
    </div>

    <script>
        // Sector data
        const sectorData = {json.dumps(sector_data)};
        
        // Initialize map
        const map = L.map('map').setView([34.05, -118.25], 10);
        
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            attribution: '&copy; OpenStreetMap, &copy; CARTO',
            maxZoom: 19
        }}).addTo(map);
        
        let currentLayer = null;
        
        function changeSector() {{
            const sector = document.getElementById('sectorSelect').value;
            
            // Remove existing layer
            if (currentLayer) {{
                map.removeLayer(currentLayer);
            }}
            
            if (sector === 'ALL') {{
                showAllSectors();
                document.getElementById('stats').innerHTML = 'Showing dominant sector per block';
                return;
            }}
            
            const data = sectorData[sector];
            if (!data) return;
            
            // Update stats
            document.getElementById('stats').innerHTML = 
                `<strong>${{sector}}</strong>: ${{data.total_jobs.toLocaleString()}} jobs in ${{data.block_count.toLocaleString()}} blocks`;
            
            // Create layer group
            currentLayer = L.layerGroup();
            
            data.data.forEach(point => {{
                const [lat, lon, jobs] = point;
                const radius = Math.max(3, Math.sqrt(jobs) * 1.5);
                
                L.circleMarker([lat, lon], {{
                    radius: radius,
                    fillColor: data.color,
                    color: data.color,
                    weight: 0,
                    fillOpacity: 0.7
                }}).bindPopup(`<strong>${{sector}}</strong><br>${{jobs}} jobs`).addTo(currentLayer);
            }});
            
            currentLayer.addTo(map);
        }}
        
        function showAllSectors() {{
            currentLayer = L.layerGroup();
            
            // For "ALL", show dominant sector per block (sample)
            const allPoints = [];
            for (const [sector, info] of Object.entries(sectorData)) {{
                info.data.slice(0, 3000).forEach(point => {{
                    allPoints.push({{
                        lat: point[0],
                        lon: point[1],
                        jobs: point[2],
                        sector: sector,
                        color: info.color
                    }});
                }});
            }}
            
            // Sample and show
            allPoints.sort(() => Math.random() - 0.5);
            allPoints.slice(0, 20000).forEach(p => {{
                const radius = Math.max(2, Math.sqrt(p.jobs) * 1.2);
                L.circleMarker([p.lat, p.lon], {{
                    radius: radius,
                    fillColor: p.color,
                    color: p.color,
                    weight: 0,
                    fillOpacity: 0.6
                }}).bindPopup(`<strong>${{p.sector}}</strong><br>${{p.jobs}} jobs`).addTo(currentLayer);
            }});
            
            currentLayer.addTo(map);
        }}
        
        function selectSector(sector) {{
            document.getElementById('sectorSelect').value = sector;
            changeSector();
        }}
        
        // Initial load
        showAllSectors();
    </script>
</body>
</html>'''
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"✅ Interactive map saved to: {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("LODES Interactive Sector Filter Map")
    print("=" * 60)
    
    df = load_data()
    create_interactive_map(df)
    
    print("\n✅ Done! Open output/lodes_sector_filter.html")


if __name__ == '__main__':
    main()

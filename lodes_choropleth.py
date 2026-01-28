"""
LODES Tract-Level Choropleth Map
Aggregate to census tract level for clearer geographic patterns
Shows where each sector is strongest
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


def load_and_aggregate():
    """Load LODES and aggregate to tract level."""
    print("Loading LODES data...")
    df = pd.read_parquet('data/lodes_wac_blocks_ca_2021.parquet')
    
    # Filter to LA County
    df['w_geocode'] = df['w_geocode'].astype(str).str.zfill(15)
    df['county'] = df['w_geocode'].str[2:5]
    df = df[df['county'] == '037'].copy()
    
    # Extract tract
    df['tract'] = df['w_geocode'].str[:11]
    
    # Aggregate to tract level
    print("Aggregating to tract level...")
    cns_cols = [f'CNS{i:02d}' for i in range(1, 21)]
    agg_cols = ['C000'] + cns_cols
    
    tract_df = df.groupby('tract')[agg_cols].sum().reset_index()
    tract_df.columns = ['tract', 'total_jobs'] + cns_cols
    
    print(f"Tracts: {len(tract_df):,}")
    
    # Calculate dominant sector per tract
    tract_df['dominant_cns'] = tract_df[cns_cols].idxmax(axis=1)
    tract_df['dominant_jobs'] = tract_df[cns_cols].max(axis=1)
    tract_df['concentration'] = tract_df['dominant_jobs'] / tract_df['total_jobs']
    
    # Map to sector names
    tract_df['dominant_sector'] = tract_df['dominant_cns'].map(
        lambda x: SECTORS.get(x, ('XX', 'Unknown', '#888'))[1]
    )
    tract_df['sector_color'] = tract_df['dominant_cns'].map(
        lambda x: SECTORS.get(x, ('XX', 'Unknown', '#888'))[2]
    )
    
    # Calculate location quotient for each sector
    # LQ = (sector_jobs_tract / total_jobs_tract) / (sector_jobs_county / total_jobs_county)
    county_totals = tract_df[['total_jobs'] + cns_cols].sum()
    for cns in cns_cols:
        county_share = county_totals[cns] / county_totals['total_jobs']
        tract_df[f'{cns}_share'] = tract_df[cns] / tract_df['total_jobs'].replace(0, np.nan)
        tract_df[f'{cns}_lq'] = tract_df[f'{cns}_share'] / county_share
    
    return tract_df


def download_tract_boundaries():
    """Download tract boundaries GeoJSON from Census."""
    print("Downloading tract boundaries...")
    
    # Census TIGER API for LA County tracts
    url = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/8/query"
    
    import requests
    
    # Query for LA County (FIPS 06037)
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
        return geojson
    else:
        print(f"Failed to download: {response.status_code}")
        return None


def create_choropleth_map(tract_df, output_path='output/lodes_choropleth.html'):
    """Create choropleth map with sector filter."""
    
    # Download tract boundaries
    geojson = download_tract_boundaries()
    
    if geojson is None:
        print("Using centroid fallback...")
        return create_centroid_map(tract_df, output_path)
    
    # Merge data with geojson
    # Census GEOID is 12 chars (state2 + county3 + tract7), LODES is 11 chars (state2 + county3 + tract6)
    # Need to match on first 11 characters
    tract_lookup = tract_df.set_index('tract').to_dict('index')
    
    for feature in geojson['features']:
        geoid = feature['properties']['GEOID']
        # Try both full GEOID and truncated version
        tract_id = geoid[:11] if len(geoid) > 11 else geoid
        if tract_id in tract_lookup:
            data = tract_lookup[tract_id]
            feature['properties']['total_jobs'] = int(data['total_jobs'])
            feature['properties']['dominant_sector'] = data['dominant_sector']
            feature['properties']['sector_color'] = data['sector_color']
            feature['properties']['concentration'] = round(data['concentration'] * 100, 1)
            # Add sector-specific data
            for cns, (naics, name, color) in SECTORS.items():
                if cns in data:
                    feature['properties'][f'{name}_jobs'] = int(data[cns])
                    feature['properties'][f'{name}_lq'] = round(data.get(f'{cns}_lq', 0), 2)
        else:
            feature['properties']['total_jobs'] = 0
            feature['properties']['dominant_sector'] = 'None'
            feature['properties']['sector_color'] = '#333'
            feature['properties']['concentration'] = 0
    
    # Calculate sector summaries
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
    
    # Sort by total jobs
    sector_stats = dict(sorted(sector_stats.items(), key=lambda x: -x[1]['total_jobs']))
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>LA Employment by NAICS Sector (Tract Level)</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        #map {{ position: absolute; top: 0; bottom: 0; left: 300px; right: 0; }}
        
        .sidebar {{
            position: absolute;
            top: 0;
            left: 0;
            bottom: 0;
            width: 300px;
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
        .view-toggle {{
            display: flex;
            padding: 10px 20px;
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
        }}
        .sector-item {{
            display: flex;
            align-items: center;
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.2s;
        }}
        .sector-item:hover {{ background: rgba(255,255,255,0.1); }}
        .sector-item.active {{ background: rgba(233, 69, 96, 0.3); border: 1px solid #e94560; }}
        .sector-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
            margin-right: 10px;
            flex-shrink: 0;
        }}
        .sector-info {{
            flex: 1;
            min-width: 0;
        }}
        .sector-name {{
            font-size: 13px;
            font-weight: 500;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .sector-stats {{
            font-size: 11px;
            opacity: 0.7;
            margin-top: 2px;
        }}
        
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
    </style>
</head>
<body>
    <div class="sidebar">
        <h1>🗺️ LA County Employment</h1>
        
        <div class="view-toggle">
            <button class="view-btn active" onclick="setView('dominant')" id="btn-dominant">Dominant Sector</button>
            <button class="view-btn" onclick="setView('filter')" id="btn-filter">Filter by Sector</button>
        </div>
        
        <div class="sector-list" id="sectorList">
            <!-- Sectors populated by JS -->
        </div>
    </div>
    
    <div id="map"></div>
    
    <div class="info-panel" id="infoPanel">
        <h3 id="infoTitle">Hover over a tract</h3>
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

    <script>
        // Data
        const geojson = {json.dumps(geojson)};
        const sectorStats = {json.dumps(sector_stats)};
        
        // State
        let currentView = 'dominant';
        let selectedSector = null;
        let tractLayer = null;
        
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
                        <div class="sector-stats">${{stats.total_jobs.toLocaleString()}} jobs · ${{stats.tracts_dominant}} tracts</div>
                    </div>
                `;
                item.onclick = () => selectSector(name);
                list.appendChild(item);
            }}
        }}
        
        // Color functions
        function getDominantColor(props) {{
            if (props.total_jobs === 0) return '#222';
            return props.sector_color;
        }}
        
        function getLQColor(lq, baseColor) {{
            // Interpolate opacity based on LQ
            // LQ < 0.5 = very faint, LQ = 1 = medium, LQ > 2 = very bright
            if (lq === undefined || lq === 0) return 'rgba(50,50,50,0.3)';
            
            const intensity = Math.min(1, lq / 2);
            // Parse hex color
            const r = parseInt(baseColor.slice(1,3), 16);
            const g = parseInt(baseColor.slice(3,5), 16);
            const b = parseInt(baseColor.slice(5,7), 16);
            
            return `rgba(${{r}},${{g}},${{b}},${{0.2 + intensity * 0.7}})`;
        }}
        
        // Style function
        function getStyle(feature) {{
            const props = feature.properties;
            
            if (currentView === 'dominant') {{
                return {{
                    fillColor: getDominantColor(props),
                    weight: 0.5,
                    opacity: 0.8,
                    color: '#333',
                    fillOpacity: props.total_jobs > 0 ? 0.7 : 0.1
                }};
            }} else if (selectedSector) {{
                const stats = sectorStats[selectedSector];
                const lq = props[selectedSector + '_lq'] || 0;
                return {{
                    fillColor: getLQColor(lq, stats.color),
                    weight: 0.5,
                    opacity: 0.8,
                    color: '#333',
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
        
        // Highlight
        function highlightFeature(e) {{
            const layer = e.target;
            layer.setStyle({{ weight: 2, color: '#fff' }});
            layer.bringToFront();
            
            const props = layer.feature.properties;
            const panel = document.getElementById('infoPanel');
            const title = document.getElementById('infoTitle');
            const content = document.getElementById('infoContent');
            
            title.textContent = 'Tract ' + props.BASENAME;
            
            let html = `<strong>Total Jobs:</strong> ${{props.total_jobs.toLocaleString()}}<br>`;
            
            if (currentView === 'dominant') {{
                html += `<strong>Dominant:</strong> ${{props.dominant_sector}}<br>`;
                html += `<strong>Concentration:</strong> ${{props.concentration}}%`;
            }} else if (selectedSector) {{
                const jobs = props[selectedSector + '_jobs'] || 0;
                const lq = props[selectedSector + '_lq'] || 0;
                html += `<strong>${{selectedSector}}:</strong> ${{jobs.toLocaleString()}} jobs<br>`;
                html += `<strong>Location Quotient:</strong> ${{lq.toFixed(2)}}`;
                if (lq > 1.5) html += ' <span style="color:#2ecc71">● Strong</span>';
                else if (lq < 0.5) html += ' <span style="color:#e74c3c">● Weak</span>';
            }}
            
            content.innerHTML = html;
            panel.classList.add('visible');
        }}
        
        function resetHighlight(e) {{
            tractLayer.resetStyle(e.target);
            document.getElementById('infoPanel').classList.remove('visible');
        }}
        
        // Draw map
        function drawMap() {{
            if (tractLayer) map.removeLayer(tractLayer);
            
            tractLayer = L.geoJSON(geojson, {{
                style: getStyle,
                onEachFeature: (feature, layer) => {{
                    layer.on({{
                        mouseover: highlightFeature,
                        mouseout: resetHighlight
                    }});
                }}
            }}).addTo(map);
            
            updateLegend();
        }}
        
        // Update legend
        function updateLegend() {{
            const legend = document.getElementById('legend');
            const scale = document.getElementById('legendScale');
            
            if (currentView === 'dominant') {{
                legend.style.display = 'none';
            }} else if (selectedSector) {{
                legend.style.display = 'block';
                const color = sectorStats[selectedSector].color;
                scale.innerHTML = `
                    <div style="flex:1;background:rgba(50,50,50,0.5)"></div>
                    <div style="flex:1;background:${{color}}88"></div>
                    <div style="flex:1;background:${{color}}"></div>
                `;
            }} else {{
                legend.style.display = 'none';
            }}
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
    
    print(f"✅ Choropleth map saved to: {output_path}")
    return output_path


def main():
    print("=" * 60)
    print("LODES Tract-Level Choropleth Map")
    print("=" * 60)
    
    tract_df = load_and_aggregate()
    create_choropleth_map(tract_df)
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()

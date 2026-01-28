"""
LODES Block-Level NAICS Map for Los Angeles
Downloads WAC data at census block level and creates interactive map

Data source: LEHD LODES
https://lehd.ces.census.gov/data/lodes/
"""
import pandas as pd
import numpy as np
import requests
from pathlib import Path
import warnings
import ssl
import urllib.request
warnings.filterwarnings('ignore')

# Fix SSL certificate issue on macOS
ssl._create_default_https_context = ssl._create_unverified_context

# Configuration
YEAR = 2021  # Latest available year with good coverage
STATE = 'ca'
LA_COUNTY_FIPS = '037'

# NAICS 2-digit sector mapping (LODES uses CNS codes)
NAICS_SECTORS = {
    'CNS01': ('11', 'Agriculture'),
    'CNS02': ('21', 'Mining'),
    'CNS03': ('22', 'Utilities'),
    'CNS04': ('23', 'Construction'),
    'CNS05': ('31-33', 'Manufacturing'),
    'CNS06': ('42', 'Wholesale Trade'),
    'CNS07': ('44-45', 'Retail Trade'),
    'CNS08': ('48-49', 'Transportation'),
    'CNS09': ('51', 'Information'),
    'CNS10': ('52', 'Finance'),
    'CNS11': ('53', 'Real Estate'),
    'CNS12': ('54', 'Professional Services'),
    'CNS13': ('55', 'Management'),
    'CNS14': ('56', 'Admin Support'),
    'CNS15': ('61', 'Education'),
    'CNS16': ('62', 'Healthcare'),
    'CNS17': ('71', 'Arts/Entertainment'),
    'CNS18': ('72', 'Accommodation/Food'),
    'CNS19': ('81', 'Other Services'),
    'CNS20': ('92', 'Public Admin'),
}

# Colors for each sector (RGB)
SECTOR_COLORS = {
    'Healthcare': [46, 204, 113],
    'Retail Trade': [52, 152, 219],
    'Accommodation/Food': [241, 196, 15],
    'Professional Services': [155, 89, 182],
    'Manufacturing': [230, 126, 34],
    'Education': [26, 188, 156],
    'Construction': [192, 57, 43],
    'Finance': [44, 62, 80],
    'Public Admin': [127, 140, 141],
    'Information': [241, 90, 34],
    'Transportation': [22, 160, 133],
    'Real Estate': [142, 68, 173],
    'Admin Support': [243, 156, 18],
    'Wholesale Trade': [39, 174, 96],
    'Other Services': [189, 195, 199],
    'Arts/Entertainment': [231, 76, 60],
    'Utilities': [52, 73, 94],
    'Agriculture': [46, 204, 113],
    'Mining': [149, 165, 166],
    'Management': [108, 92, 231],
}


def download_lodes_wac_blocks(year=YEAR, state=STATE):
    """Download LODES WAC at block level."""
    
    url = f"https://lehd.ces.census.gov/data/lodes/LODES8/{state}/wac/{state}_wac_S000_JT00_{year}.csv.gz"
    
    print(f"Downloading LODES WAC blocks for {state.upper()} {year}...")
    print(f"URL: {url}")
    
    cache_path = Path(f'data/lodes_wac_blocks_{state}_{year}.parquet')
    
    if cache_path.exists():
        print(f"Loading from cache: {cache_path}")
        return pd.read_parquet(cache_path)
    
    df = pd.read_csv(url, compression='gzip')
    print(f"Downloaded: {len(df):,} blocks")
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    print(f"Cached to: {cache_path}")
    
    return df


def filter_la_county(df):
    """Filter to LA County blocks."""
    df['w_geocode'] = df['w_geocode'].astype(str).str.zfill(15)
    df['county'] = df['w_geocode'].str[2:5]
    
    la_df = df[df['county'] == LA_COUNTY_FIPS].copy()
    print(f"LA County blocks: {len(la_df):,}")
    
    return la_df


def calculate_dominant_sector(df):
    """Find dominant NAICS sector for each block."""
    
    cns_cols = [c for c in df.columns if c.startswith('CNS')]
    
    df['dominant_sector'] = df[cns_cols].idxmax(axis=1)
    df['dominant_jobs'] = df[cns_cols].max(axis=1)
    df['total_jobs'] = df['C000']
    
    df['naics_2digit'] = df['dominant_sector'].map(lambda x: NAICS_SECTORS.get(x, ('XX', 'Unknown'))[0])
    df['sector_name'] = df['dominant_sector'].map(lambda x: NAICS_SECTORS.get(x, ('XX', 'Unknown'))[1])
    df['sector_concentration'] = df['dominant_jobs'] / df['total_jobs'].replace(0, np.nan)
    
    return df


def get_block_coordinates(df):
    """Get coordinates for blocks using tract centroids + jitter."""
    print("\nGetting block coordinates...")
    
    df['tract'] = df['w_geocode'].str[:11]
    
    # Download CA tract centroids
    url = "https://www2.census.gov/geo/docs/reference/cenpop2020/tract/CenPop2020_Mean_TR06.txt"
    print("Downloading CA tract centroids...")
    
    centroids = pd.read_csv(url)
    centroids['STATEFP'] = centroids['STATEFP'].astype(str).str.zfill(2)
    centroids['COUNTYFP'] = centroids['COUNTYFP'].astype(str).str.zfill(3)
    centroids['TRACTCE'] = centroids['TRACTCE'].astype(str).str.zfill(6)
    centroids['tract'] = centroids['STATEFP'] + centroids['COUNTYFP'] + centroids['TRACTCE']
    
    la_centroids = centroids[centroids['COUNTYFP'] == LA_COUNTY_FIPS][['tract', 'LATITUDE', 'LONGITUDE']]
    print(f"LA County tract centroids: {len(la_centroids):,}")
    
    df = df.merge(la_centroids, on='tract', how='left')
    
    # Jitter blocks within tract
    np.random.seed(42)
    df['lat'] = df['LATITUDE'] + np.random.uniform(-0.002, 0.002, len(df))
    df['lon'] = df['LONGITUDE'] + np.random.uniform(-0.002, 0.002, len(df))
    
    return df


def create_pydeck_map(df, output_path='output/lodes_naics_map.html'):
    """Create interactive map using pydeck."""
    import pydeck as pdk
    
    map_df = df[(df['total_jobs'] > 0) & df['lat'].notna()].copy()
    print(f"\nBlocks for map: {len(map_df):,}")
    
    map_df['color'] = map_df['sector_name'].map(
        lambda x: SECTOR_COLORS.get(x, [128, 128, 128])
    )
    map_df['radius'] = np.sqrt(map_df['total_jobs']) * 8
    
    # Prepare data for pydeck
    map_data = map_df[['lon', 'lat', 'radius', 'color', 'sector_name', 'total_jobs', 'sector_concentration']].copy()
    map_data['sector_concentration'] = (map_data['sector_concentration'] * 100).round(1)
    
    layer = pdk.Layer(
        'ScatterplotLayer',
        data=map_data,
        get_position=['lon', 'lat'],
        get_radius='radius',
        get_fill_color='color',
        pickable=True,
        opacity=0.7,
        stroked=False,
        filled=True,
        radius_min_pixels=1,
        radius_max_pixels=30,
    )
    
    view_state = pdk.ViewState(
        latitude=34.05,
        longitude=-118.25,
        zoom=10,
        pitch=0,
    )
    
    tooltip = {
        "html": "<b>{sector_name}</b><br/>Jobs: {total_jobs}<br/>Concentration: {sector_concentration}%",
        "style": {"backgroundColor": "#333", "color": "white", "fontSize": "12px"}
    }
    
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style='mapbox://styles/mapbox/dark-v10'
    )
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    deck.to_html(output_path)
    print(f"✅ Pydeck map saved to: {output_path}")


def create_folium_map(df, output_path='output/lodes_naics_map_folium.html'):
    """Create interactive map using folium (no mapbox token needed)."""
    import folium
    
    map_df = df[(df['total_jobs'] > 0) & df['lat'].notna()].copy()
    print(f"\nBlocks for map: {len(map_df):,}")
    
    # Sample for performance (folium is slower)
    if len(map_df) > 20000:
        map_df = map_df.sample(20000, random_state=42)
        print(f"Sampled to: {len(map_df):,} for performance")
    
    m = folium.Map(
        location=[34.05, -118.25], 
        zoom_start=10, 
        tiles='cartodbdark_matter'
    )
    
    for _, row in map_df.iterrows():
        color = SECTOR_COLORS.get(row['sector_name'], [128, 128, 128])
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        
        folium.CircleMarker(
            location=[row['lat'], row['lon']],
            radius=max(2, np.sqrt(row['total_jobs']) / 3),
            color=hex_color,
            fill=True,
            fillColor=hex_color,
            fillOpacity=0.7,
            weight=0,
            popup=f"<b>{row['sector_name']}</b><br/>Jobs: {row['total_jobs']}"
        ).add_to(m)
    
    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; 
                background: rgba(0,0,0,0.8); padding: 10px; border-radius: 5px;
                font-size: 11px; color: white;">
    <b>NAICS Sectors</b><br>
    '''
    for sector, color in list(SECTOR_COLORS.items())[:10]:
        hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
        legend_html += f'<span style="color:{hex_color}">●</span> {sector}<br>'
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    m.save(output_path)
    print(f"✅ Folium map saved to: {output_path}")


def create_sector_summary(df, output_path='output/sector_summary.csv'):
    """Create summary statistics by sector."""
    
    summary = df.groupby('sector_name').agg({
        'w_geocode': 'count',
        'total_jobs': 'sum',
        'dominant_jobs': 'sum',
        'sector_concentration': 'mean'
    }).round(2)
    
    summary.columns = ['block_count', 'total_jobs', 'sector_jobs', 'avg_concentration']
    summary = summary.sort_values('total_jobs', ascending=False)
    summary['pct_of_total'] = (summary['total_jobs'] / summary['total_jobs'].sum() * 100).round(1)
    
    print("\n📊 SECTOR SUMMARY (LA County)")
    print("=" * 70)
    print(summary.to_string())
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path)
    print(f"\n✅ Summary saved to: {output_path}")
    
    return summary


def main():
    print("=" * 60)
    print("LODES Block-Level NAICS Map for Los Angeles")
    print("=" * 60)
    
    # 1. Download LODES WAC blocks
    df = download_lodes_wac_blocks()
    
    # 2. Filter to LA County
    la_df = filter_la_county(df)
    
    # 3. Calculate dominant sector per block
    la_df = calculate_dominant_sector(la_df)
    
    # 4. Get coordinates
    la_df = get_block_coordinates(la_df)
    
    # 5. Create summary
    create_sector_summary(la_df)
    
    # 6. Create maps
    try:
        create_pydeck_map(la_df)
    except Exception as e:
        print(f"Pydeck failed ({e}), trying folium...")
    
    create_folium_map(la_df)
    
    # 7. Save processed data
    output_cols = ['w_geocode', 'tract', 'total_jobs', 'dominant_sector', 
                   'naics_2digit', 'sector_name', 'sector_concentration', 'lat', 'lon']
    Path('data').mkdir(exist_ok=True)
    la_df[output_cols].to_parquet('data/lodes_blocks_la.parquet')
    print("\n✅ Processed data saved to: data/lodes_blocks_la.parquet")
    
    print("\n" + "=" * 60)
    print("DONE! Open output/lodes_naics_map_folium.html in browser")
    print("=" * 60)


if __name__ == '__main__':
    main()

# LA LODES Block-Level NAICS Map

Interactive map of employment by NAICS 2-digit sector at census block level for Los Angeles County.

## Data Source

**LEHD LODES** (Longitudinal Employer-Household Dynamics Origin-Destination Employment Statistics)
- WAC (Workplace Area Characteristics) at block level
- Source: https://lehd.ces.census.gov/data/lodes/

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python lodes_map.py
```

## Output

- `output/lodes_naics_map.html` - Interactive map
- `output/sector_summary.csv` - Employment by sector
- `data/lodes_blocks_la.parquet` - Processed data

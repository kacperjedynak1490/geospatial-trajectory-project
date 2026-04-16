import osmnx as ox
import pandas as pd
import h3
from shapely.geometry import Polygon

def main():
    # download Porto geometry from OpenStreetMap
    porto_gdf = ox.geocode_to_gdf("Porto, Portugal")
    geom = porto_gdf.geometry.iloc[0]
    
    polygons = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
    hex_ids = set()
    
    # generate H3 hexagons at resolution 9
    for poly in polygons:
        coords = [(lat, lon) for lon, lat in poly.exterior.coords]
        
        h3_shape = h3.LatLngPoly(coords)
        hex_ids.update(h3.polygon_to_cells(h3_shape, 9))
        
    # convert H3 boundary to WKT Polygon (standard: lon lat)
    def hex_to_wkt(h_id):
        boundary = h3.cell_to_boundary(h_id)
        # Flip to (lon, lat) for standard GIS compatibility
        lon_lat = [(lon, lat) for lat, lon in boundary]
        return Polygon(lon_lat).wkt

    # create DataFrame with AREA_ID and WKT POLYGON
    records = [{"AREA_ID": h_id, "POLYGON": hex_to_wkt(h_id)} for h_id in hex_ids]
    
    #save to parquet
    pd.DataFrame(records).to_parquet("data/processed/area.parquet", index=False)

if __name__ == "__main__":
    main()
import numpy as np
import torch
from numpy.linalg import matrix_power
from shapely.geometry import Point, LineString
import pandas as pd
import geopandas as gpd


def pure_adj(k):
    acc = [-1,36,47,58,74,99,110,144,154,192]
    
    base = np.identity(193,dtype=bool)
    
    for i in range(0,193):
        if i not in acc:
            base[i][i+1]=True
            
    base[36][37]=True
    base[36][75]=True
    base[47][48]=True
    base[110][48]=True
    base[99][100]=True
    base[99][111]=True
    base[58][59]=True
    base[58][145]=True
    base[144][155]=True
    base[154][155]=True
    base[192][0]=True
    base[74][0]=True
    
    both = np.logical_or(base, base.transpose())

    output = matrix_power(both, k)
    output[output > 0] = 1.
    
    return output

def adjacency_matrix(k):
    acc = [-1,36,47,58,74,99,110,144,154,192]
    
    base = np.identity(193,dtype=bool)
    
    for i in range(0,193):
        if i not in acc:
            base[i][i+1]=True
            
    base[36][37]=True
    base[36][75]=True
    base[47][48]=True
    base[110][48]=True
    base[99][100]=True
    base[99][111]=True
    base[58][59]=True
    base[58][145]=True
    base[144][155]=True
    base[154][155]=True
    base[192][0]=True
    base[74][0]=True
    
    both = np.logical_or(base, base.transpose())

    output = matrix_power(both, k)
    output[output > 0] = 1.
    
    return torch.Tensor(output)

def adjacency_matrixq(k1, k2):
    acc = [-1,36,47,58,74,99,110,144,154,192]
    
    base = np.identity(193,dtype=bool)
    
    for i in range(0,193):
        if i not in acc:
            base[i][i+1]=True
            
    base[36][37]=True
    base[36][75]=True
    base[47][48]=True
    base[110][48]=True
    base[99][100]=True
    base[99][111]=True
    base[58][59]=True
    base[58][145]=True
    base[144][155]=True
    base[154][155]=True
    base[192][0]=True
    base[74][0]=True
    
    downstream = matrix_power(base, k1)
    upstream = matrix_power(base.T, k2)
    both = np.logical_or(upstream, downstream)

    both[both > 0] = 1.
    
    return torch.Tensor(both)

def create_linestrings(links, lengths):
    linestrings = []
    c = 0
    start = 0
    for key, value in links.items():
        linestring = pd.DataFrame(np.array(value).T, columns=['lat','lon'])
        linestring['ids'] = np.arange(start, start + lengths[c])
        start += lengths[c]
        c += 1
        linestring = gpd.GeoDataFrame(linestring, geometry=gpd.points_from_xy(linestring.lon, linestring.lat))
        lines = [LineString([linestring.geometry[i], linestring.geometry[i+1]]) for i in range(len(linestring.geometry)-1)]
        lines.append(LineString([linestring.geometry[0], linestring.geometry[1]]))
        linestring['geometry'] = lines
        linestrings.append(linestring)
    linestrings = gpd.GeoDataFrame(pd.concat(linestrings, axis=0).set_index('ids'))
    linestrings.crs = 'EPSG:4326'
    
    return linestrings


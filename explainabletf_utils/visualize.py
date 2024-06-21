import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import folium
import numpy as np
import branca
import geopandas as gpd
import pandas as pd
from explainabletf_utils.utils import *
from matplotlib import cm
from labellines import labelLine, labelLines
from skimage.transform import resize

def draw_demand(demand, links, lengths, linestrings, zoom_start=3):

    v_min = -400#np.amin(demand)*3000
    v_max = 400#np.amax(demand)*3000
    map = folium.Map(
        location=[52.355630000000005,4.873565],
        zoom_start=zoom_start,
    )

    der = np.zeros(sum(lengths))
    start = 0
    for i in range(193):
        der[start:start+lengths[i]] = demand[i]
        start = start+lengths[i]

    D = {}
    D['ids'] = np.arange(sum(lengths))
    D['demand'] = der

    D = pd.DataFrame.from_dict(D)

    data = gpd.GeoDataFrame(D, geometry=linestrings.geometry)
    data.crs = 'EPSG:4326'
    data = data.drop_duplicates('ids').sort_values('ids')


    for key, value in links.items():
        s_value = np.array(value).T
        folium.features.ColorLine(
            s_value,
            colors = np.ones(len(s_value))*demand[int(key)],
            colormap = branca.colormap.LinearColormap(colors=['blue', 'white', 'red'], vmin=v_min, vmax=v_max),
            weight=5,
            opacity=1,
        ).add_to(map)


    colorscale = branca.colormap.LinearColormap(colors=['blue', 'white', 'red'], vmin=v_min, vmax=v_max)
    style_function = lambda feature: {"fillOpacity": 0.5,"weight": 5,"color": colorscale(feature['properties']['demand'])}

    gjson = folium.features.GeoJson(data, style_function=style_function,).add_to(map)

    colorscale = colorscale.to_step(index=np.arange(v_min, v_max+20, 20))
    colorscale.caption = 'Estimated demand (veh/lane/h)'
    colorscale.add_to(map)

    folium.features.GeoJsonPopup(
        fields=['ids','demand'],
        aliases=['Segment_id', 'Estimated demand'],
        labels=True
    ).add_to(gjson)

    return map

def draw_status_folium(state, links, lengths, linestrings, zoom_start=12, mode='speed'):
    if mode == 'speed':
        v_min = 0
        v_max = 130

    if mode == 'flow':
        v_min = 0
        v_max = 3000
    
    map = folium.Map(
        location=[52.355630000000005,4.873565],
        zoom_start=zoom_start,
    )

    der = np.zeros(sum(lengths))
    start = 0
    for i in range(193):
        der[start:start+lengths[i]] = state[i]
        start = start+lengths[i]

    D = {}
    D['ids'] = np.arange(sum(lengths))
    D['state'] = der

    D = pd.DataFrame.from_dict(D)

    data = gpd.GeoDataFrame(D, geometry=linestrings.geometry)
    data.crs = 'EPSG:4326'
    data = data.drop_duplicates('ids').sort_values('ids')


    for key, value in links.items():
        s_value = np.array(value).T
        folium.features.ColorLine(
            s_value,
            colors = np.ones(len(s_value))*state[int(key)],
            colormap = branca.colormap.LinearColormap(colors=['red', 'white', 'blue'], vmin=v_min, vmax=v_max),
            weight=5,
            opacity=1,
        ).add_to(map)

    if mode == 'speed':
        colorscale = branca.colormap.LinearColormap(colors=['red', 'white', 'blue'], vmin=v_min, vmax=v_max)
    if mode == 'flow':
        colorscale = branca.colormap.LinearColormap(colors=['blue', 'white', 'red'], vmin=v_min, vmax=v_max)
    style_function = lambda feature: {"fillOpacity": 0.5,"weight": 5,"color": colorscale(feature['properties']['state'])}

    gjson = folium.features.GeoJson(data, style_function=style_function,).add_to(map)

    if mode == 'speed':
        colorscale = colorscale.to_step(index=np.arange(v_min, v_max+2, 2))
        colorscale.caption = 'speed (km/h)'
    if mode == 'flow':
        colorscale = colorscale.to_step(index=np.arange(v_min, v_max+20, 20))
        colorscale.caption = 'flow (veh/h/lane)'
    colorscale.add_to(map)

    folium.features.GeoJsonPopup(
        fields=['ids','state'],
        aliases=['Segment_id', 'state'],
        labels=True
    ).add_to(gjson)

    return map

def get_speed_fg(y, links):
    fg = folium.FeatureGroup(name="speed_pred")
    for key, value in links.items():
        s_value = np.array(value).T
        fg.add_child(
            folium.features.ColorLine(
                s_value,
                colors = np.ones(len(s_value))*y[int(key)]*130,
                colormap = branca.colormap.LinearColormap(colors=['red', 'yellow', 'blue'], vmin=0, vmax=130),
                weight=5,
                opacity=1,
            )
        )
    return fg

def get_flow_fg(y, links):
    fg = folium.FeatureGroup(name="speed_pred")
    for key, value in links.items():
        s_value = np.array(value).T
        fg.add_child(
            folium.features.ColorLine(
                s_value,
                colors = np.ones(len(s_value))*y[int(key)]*3000,
                colormap = branca.colormap.LinearColormap(colors=['red', 'yellow', 'blue'], vmin=0, vmax=3000),
                weight=5,
                opacity=1,
            )
        )
    return fg

def draw_prediction(speed, links, zoom_start=11):

    v_min = 0
    v_max = 130
    map = folium.Map(
        location=[52.355630000000005,4.873565],
        zoom_start=zoom_start,
    )

    for key, value in links.items():
        s_value = np.array(value).T
        folium.features.ColorLine(
            s_value,
            colors = np.ones(len(s_value))*speed[int(key)]*130,
            colormap = branca.colormap.LinearColormap(colors=['red', 'yellow', 'blue'], vmin=v_min, vmax=v_max),
            weight=5,
            opacity=1,
        ).add_to(map)


    colorscale = branca.colormap.LinearColormap(colors=['red', 'yellow', 'blue'], vmin=v_min, vmax=v_max)
    colorscale = colorscale.to_step(index=np.arange(v_min, v_max+2, 2))
    colorscale.caption = 'predicted speed (km/h)'
    colorscale.add_to(map)

    return map

def draw_interpretation(links, A, Attention, id, mode='flow'):
    surrounding_id = np.where(A[id]>0)[0]
    atten = Attention[id][surrounding_id]
    vm = np.amax(atten)

    fig, ax = plt.subplots(constrained_layout=True, figsize=(7, 6))
    #im = ax.scatter([])
    for id_, at in zip(surrounding_id, atten):
        value = links[str(id_)]
        if id_ == id:
            ax.plot(value[1], value[0], c=cm.rainbow(at/vm), lw=6, label=format(at, '.2f'))
            im = ax.scatter(value[1], value[0], c=at*np.ones(len(value[1])), s=0., vmin=0, vmax=vm, cmap=cm.rainbow)
        else:
            if mode=='speed':
                ax.plot(value[1], value[0], c=cm.rainbow(at/vm), lw=3, ls='--', label=format(at, '.2f'))
            if mode=='flow':
                ax.plot(value[1], value[0], c=cm.rainbow(at/vm), ls='--', lw=3)
    if mode=='speed':
        labelLines(ax.get_lines(), zorder=2.5, fontsize=10)
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    fig.colorbar(im, orientation='vertical', label='impact coefficience')
    #ax.set_aspect('equal')
    return fig

def dynamic_speed(links, x, y, step):
    ref = [0, 37, 48, 59, 75, 100, 111, 145, 155, 193]

    history = resize(x, (40, 193), anti_aliasing=True)
    prediction = resize(y, (30, 193), anti_aliasing=True)
    #print(x.shape, history.shape)
    heatmap = np.zeros((70,193))
    heatmap[:40] = history
    heatmap[40:40+step*2] = prediction[:step*2]

    current = y[step-1]

    fig1, ax1 = plt.subplots(constrained_layout=True, figsize=(8, 6))
    for key, value in links.items():
        ax1.scatter(value[1], value[0], c=current[int(key)]*np.ones(len(value[1])), s=9,cmap=cm.rainbow_r, vmin=0, vmax=130,)
        if key == '0':
            im1 = ax1.scatter(value[1], value[0], c=current[0]*np.ones(len(value[1])), s=0, cmap=cm.rainbow_r, vmin=0, vmax=130)
    ax1.set_xticks([])
    ax1.set_yticks([])
    fig1.colorbar(im1, orientation='vertical', label='speed (km/h)', fraction=0.046, pad=0.04)
    ax1.set_title('prediction horizon '+str(step*2)+' minutes', fontsize=18)
    #ax1.set_aspect('equal')

    fig2, ax2 = plt.subplots(constrained_layout=True, figsize=(8, 6))

    X = np.arange(0.5,194, 1)
    Y = np.arange(-40.5,30.5, 1)
    im2 = ax2.pcolormesh(Y, X, heatmap.T, cmap='rainbow_r', vmin=0, vmax=130.)

    ax2.vlines(0, ymin=0.5, ymax=193.5, color='white', lw=2)
    # for i in range(1,9):
    #     ax2.hlines(ref[i]-0.5, xmin=-40.5, xmax=29.5, color='black', lw=1.5)

    # ax2.text(26.5, 25, '1', fontsize=15)
    # ax2.text(26.5, 42, '2', fontsize=15)
    # ax2.text(26.5, 53, '3', fontsize=15)
    # ax2.text(26.5, 67, '4', fontsize=15)
    # ax2.text(26.5, 82, '5', fontsize=15)
    # ax2.text(26.5, 105, '6', fontsize=15)
    # ax2.text(26.5, 125, '7', fontsize=15)
    # ax2.text(26.5, 150, '8', fontsize=15)
    # ax2.text(26.5, 171, '9', fontsize=15)
    ax2.text(0.5, 180, 'now', fontsize=15)

    fig2.colorbar(im2, orientation='vertical', label='speed (km/h)', fraction=0.046, pad=0.04)
    ax2.set_title('prediction', fontsize=18)
    return fig1, fig2

def dynamic_flow(links, x, y, step):
    ref = [0, 37, 48, 59, 75, 100, 111, 145, 155, 193]

    history = resize(x, (40, 193), anti_aliasing=True)
    prediction = resize(y, (30, 193), anti_aliasing=True)
    #print(x.shape, history.shape)
    heatmap = np.zeros((70,193))
    heatmap[:40] = history
    heatmap[40:40+step*2] = prediction[:step*2]

    current = y[step-1]

    fig1, ax1 = plt.subplots(constrained_layout=True, figsize=(8, 6))
    for key, value in links.items():
        ax1.scatter(value[1], value[0], c=current[int(key)]*np.ones(len(value[1])), s=9,cmap=cm.rainbow, vmin=0, vmax=3000)
        if key == '0':
            im1 = ax1.scatter(value[1], value[0], c=current[0]*np.ones(len(value[1])), s=0, cmap=cm.rainbow, vmin=0, vmax=3000)
    ax1.set_xticks([])
    ax1.set_yticks([])
    fig1.colorbar(im1, orientation='vertical', label='flow (veh/h/lane)', fraction=0.046, pad=0.04)
    ax1.set_title('prediction horizon '+str(step*2)+' minutes', fontsize=18)
    #ax1.set_aspect('equal')

    fig2, ax2 = plt.subplots(constrained_layout=True, figsize=(8, 6))

    X = np.arange(0.5,194, 1)
    Y = np.arange(-40.5,30.5, 1)
    im2 = ax2.pcolormesh(Y, X, heatmap.T, cmap='rainbow', vmin=0, vmax=3000.)

    ax2.vlines(-0.5, ymin=0.5, ymax=193.5, color='white', lw=1)
    for i in range(1,9):
        ax2.hlines(ref[i]-0.5, xmin=-40.5, xmax=29.5, color='black', lw=1.5)

    ax2.text(26.5, 25, '1', fontsize=15)
    ax2.text(26.5, 42, '2', fontsize=15)
    ax2.text(26.5, 53, '3', fontsize=15)
    ax2.text(26.5, 67, '4', fontsize=15)
    ax2.text(26.5, 82, '5', fontsize=15)
    ax2.text(26.5, 105, '6', fontsize=15)
    ax2.text(26.5, 125, '7', fontsize=15)
    ax2.text(26.5, 150, '8', fontsize=15)
    ax2.text(26.5, 171, '9', fontsize=15)
    ax2.text(-0.5, 180, 'now', fontsize=15)

    fig2.colorbar(im2, orientation='vertical', label='flow (veh/h/lane)', fraction=0.046, pad=0.04)
    ax2.set_title('heatmap', fontsize=18)
    return fig1, fig2

def dynamic_speed_test(links, x, y, step):
    ref = [0, 37, 48, 59, 75, 100, 111, 145, 155, 193]

    heatmap = np.zeros((35,193))
    heatmap[:20] = x
    heatmap[20:20+step] = y[:step]

    current = y[step-1]

    fig1, ax1 = plt.subplots(constrained_layout=True, figsize=(8, 6))
    for key, value in links.items():
        ax1.scatter(value[1], value[0], c=current[int(key)]*np.ones(len(value[1])), s=9,cmap=cm.rainbow_r, vmin=0, vmax=130,)
        if key == '0':
            im1 = ax1.scatter(value[1], value[0], c=current[0]*np.ones(len(value[1])), s=0, cmap=cm.rainbow_r, vmin=0, vmax=130)
    ax1.set_xticks([])
    ax1.set_yticks([])
    fig1.colorbar(im1, orientation='vertical', label='speed (km/h)', fraction=0.046, pad=0.04)
    ax1.set_title('prediction horizon '+str(step*2)+' minutes', fontsize=18)
    #ax1.set_aspect('equal')

    fig2, ax2 = plt.subplots(constrained_layout=True, figsize=(8, 6))

    X = np.arange(0.5,194,1)
    Y = np.arange(-41,31, 2)
    im2 = ax2.pcolormesh(Y, X, heatmap.T, cmap='rainbow_r', vmin=0, vmax=130.)

    ax2.vlines(-0.5, ymin=0.5, ymax=193.5, color='white', lw=1)

    fig2.colorbar(im2, orientation='vertical', label='speed (km/h)', fraction=0.046, pad=0.04)
    ax2.set_title('heatmap', fontsize=18)
    return fig1, fig2

def draw_predictability(demand, links, lengths, linestrings, zoom_start=3):

    v_min = 1#np.amin(demand)*3000
    v_max = 6#np.amax(demand)*3000
    map = folium.Map(
        location=[52.355630000000005,4.873565],
        zoom_start=zoom_start,
    )

    der = np.zeros(sum(lengths))
    start = 0
    for i in range(193):
        der[start:start+lengths[i]] = demand[i]
        start = start+lengths[i]

    D = {}
    D['ids'] = np.arange(sum(lengths))
    D['demand'] = der

    D = pd.DataFrame.from_dict(D)

    data = gpd.GeoDataFrame(D, geometry=linestrings.geometry)
    data.crs = 'EPSG:4326'
    data = data.drop_duplicates('ids').sort_values('ids')


    for key, value in links.items():
        s_value = np.array(value).T
        folium.features.ColorLine(
            s_value,
            colors = np.ones(len(s_value))*demand[int(key)],
            colormap = branca.colormap.LinearColormap(colors=['white', 'red'], vmin=v_min, vmax=v_max),
            weight=5,
            opacity=1,
        ).add_to(map)


    colorscale = branca.colormap.LinearColormap(colors=['white', 'red'], vmin=v_min, vmax=v_max)
    style_function = lambda feature: {"fillOpacity": 0.5,"weight": 5,"color": colorscale(feature['properties']['demand'])}

    gjson = folium.features.GeoJson(data, style_function=style_function,).add_to(map)

    colorscale = colorscale.to_step(index=np.arange(v_min, v_max+0.2, 0.2))
    colorscale.caption = 'Estimated error lower bound (km/h)'
    colorscale.add_to(map)

    folium.features.GeoJsonPopup(
        fields=['ids','demand'],
        aliases=['Segment_id', 'predictability'],
        labels=True
    ).add_to(gjson)

    return map
from dash import Dash, dcc, html, Input, Output
import numpy as np
import plotly.express as px
from pyproj import Transformer
import plotly.graph_objects as go


x1 = -200
x2 = 400
h1 = 61
r1 = 10000


# Create forward rotation matrix
theta = np.radians(29)
c, s = np.cos(theta), np.sin(theta)
rotation_matrix = np.array(((c, -s), (s, c)))

# Create backward rotation matrix
inv_theta = np.radians(-29)
c, s = np.cos(inv_theta), np.sin(inv_theta)
inv_rotation_matrix = np.array(((c, -s), (s, c)))

# Map dependend limits
xscale = 1. * 2000 / (0.48 + 0.35)
yscale = 1. * 2000 / (0.125 + 0.265)
x_min = -4 * xscale
x_max = 1.18 * xscale
y_min = -1.07 * yscale
y_max = 1 * yscale

# Create grid for isohypses
x_iso, y_iso = np.meshgrid(np.arange(x_min, x_max, 10), np.arange(y_min, y_max, 10))
iso_normal = np.stack([x_iso,y_iso], axis=2)

# Tilt grid for isohypses
iso_tilted= np.tensordot(inv_rotation_matrix, iso_normal, axes=([1],[2]))
x_iso_tilted = iso_tilted[0,:,:]
y_iso_tilted = iso_tilted[1,:,:]


lon_brunnen = 824550#13.71639
lat_brunnen = 5759000#51.88775

transformer = Transformer.from_crs("epsg:7416", "epsg:4326")

app = Dash(__name__)


image_filename = 'TUD_logo.png' # replace with your own image


app.layout = html.Div([
    html.H1('Trennstromlinie', style={"text-align":"center"}),
    html.H4('Für einen gespannten, stationären Grundwasserleiter mit 120m Mächtigkeit', style={"text-align":"center"}),
    dcc.Graph(id="graph", style={'height': '70vh'}),
    html.Label('Entnahmerate in l/s'),
    dcc.Slider(id="Q",
            min=0,
            max=1000,
            #marks={i: f"{i}" for i in np.arange(0., 1100., 100)},
            value=100,
        ),
    html.Label('Hydraulische Leitfähigkeit in m/s'),
    dcc.Slider(id="K",
            min=0.0001,
            max=0.001,
            marks={i: f'{i}'for i in np.round(np.arange(0.0001, 0.0011, 0.0001), 6)},
            value=0.0004
        ),

    html.Label('Hydraulischer Hintergrundgradient in m/m'),
    dcc.Slider(id="grad",
            min=0,
            max=0.01,
            #marks={i: f'Label {i}' if i == 1 else str(i) for i in range(1, 6)},
            value=0.0017,
        ),
    html.Hr(),
    html.P([
            html.P(["ⒸReimann und Grabow, 2022"],style={"float": "right"}),
            html.Img(src=app.get_asset_url('TUD_logo.png'),
                    height=50)
        ])
])


def ymax_conf(Q, K, grad, b):
    ymax = Q/(2.*K*np.abs(grad)*b)
    return ymax

def x0_conf(Q, K, grad, b):
    x0 = -Q/(2.*np.pi*K*grad*b)
    return x0

def isohypsen_conf(h1, h2, x1, x2, K, Q, b, x, y):
    h_r1 = 0
    h_iso = (h2-h1)*(x-x1)/(x2-x1) +h1 + Q * np.log(np.sqrt(x**2+y**2) / r1) / (2 * np.pi * K * b) + h_r1
    return h_iso


def getXYtrennstromlinie(Q, K, grad):
    # Umrechnung von l/s in m^3/s
    b = 120
    Q = Q / 1000

    h2 = h1 - grad * (x1 - x2)
    L = x1 - x2

    ymax = ymax_conf(Q, K, grad, b) * 0.95
    #x0 = x0_conf(Q, K, i, b)
    y = np.append(np.arange(-ymax, ymax, 10), ymax)
    x = y/(np.tan(2*np.pi*K*grad*b*y/Q))

    #for i_idx in x:
    #    print(i_idx)

    #mask = x > -30000
    #x = x[mask]
    #y = y[mask]


    h_iso = isohypsen_conf(h1, h2, x1, x2, K, Q, b, x_iso_tilted, y_iso_tilted)

    tsl_normal = np.vstack([x,y])
    tsl_tilted = rotation_matrix.dot(tsl_normal)

    final_x = tsl_tilted[0,:] + lon_brunnen
    final_y = tsl_tilted[1,:] + lat_brunnen

    # remove negatives
    mask = final_x >= 0
    final_x = final_x[mask]
    final_y = final_y[mask]


    tsl_normal = np.vstack([x,y])
    iso_normal = np.stack([x_iso,y_iso], axis=2)
    iso_tilted= np.tensordot(rotation_matrix, iso_normal, axes=([1],[2]))
    tsl_tilted = rotation_matrix.dot(tsl_normal)

    x_plot, y_plot = transformer.transform(final_x,final_y)

    return y_plot, x_plot


y_plot_init, x_plot_init = getXYtrennstromlinie(Q=100, K=0.0004, grad=0.0017)
fig = go.Figure(data=go.Scattermapbox(
    lon = y_plot_init,
    lat = x_plot_init,
    mode = 'lines',
    name = "Trennstromlinie"
    ))

brunnen_x_plot, brunnen_y_plot = transformer.transform(lon_brunnen,lat_brunnen)
fig.add_trace(
    go.Scattermapbox(
        lon = [brunnen_y_plot],
        lat = [brunnen_x_plot],
        mode="markers",
    marker = dict(
        size = 8),
        name = "Brunnen"
))
fig.update_mapboxes(style="open-street-map",
               zoom = 11,
               center={
               "lat":brunnen_x_plot,
               "lon":brunnen_y_plot})
fig.update_layout(uirevision="Don't change"
    )

@app.callback(
    Output("graph", "figure"),
    Input("Q", "value"),
    Input("K", "value"),
    Input("grad", "value"))
    #Input("b", "value"))
def TSL_conf(Q, K, grad):

    y_plot, x_plot = getXYtrennstromlinie(Q, K, grad)
    fig.update_traces(lon = y_plot,
                    lat = x_plot,
                  selector=dict(mode="lines"))

    return fig

server = app.server

if __name__ == '__main__':
    app.run_server(debug=False)

from dash import Dash, dcc, html, Input, Output
import numpy as np
import plotly.express as px
from pyproj import Transformer
import plotly.graph_objects as go


lon_brunnen = 824550#13.71639
lat_brunnen = 5759000#51.88775

transformer = Transformer.from_crs("epsg:7416", "epsg:4326")

app = Dash(__name__)


image_filename = 'TUD_logo.png'


app.layout = html.Div([
    html.H1('Trennstromlinie', style={"text-align":"center"}),
    html.H4('Für einen gespannten, stationären Grundwasserleiter', style={"text-align":"center"}),
    #html.H4('Natürliche Anstromrichtung des Grundwassers ist 241°WSW', style={"text-align":"center"}),
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
            value=0.0006
        ),

    html.Label('Hydraulischer Hintergrundgradient in m/m'),
    dcc.Slider(id="grad",
            min=0,
            max=0.01,
            #marks={i: f'Label {i}' if i == 1 else str(i) for i in range(1, 6)},
            value=0.0017,
        ),
    html.Label('Aquifermächtigkeit in m'),
    dcc.Slider(id="b",
            min=80,
            max=160,
            #marks={i: f'Label {i}' if i == 1 else str(i) for i in range(1, 6)},
            value=120,
        ),
    html.Label('Drehwinkel der Anstromrichtung in °'),
    dcc.Slider(id="winkel",
            min=0,
            max=360,
            marks={int(i): f'{int(i)}'for i in np.linspace(0,360, 7)},
            value=200,
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


def getXYtrennstromlinie(Q, K, grad, b, winkel):
    # Create forward rotation matrix
    theta = np.radians(270-winkel)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array(((c, -s), (s, c)))



    # Umrechnung von l/s in m^3/s
    Q = Q / 1000

    ymax = ymax_conf(Q, K, grad, b) * 0.95
    #x0 = x0_conf(Q, K, i, b)
    y = np.append(np.arange(-ymax, ymax, 10), ymax)
    x = y/(np.tan(2*np.pi*K*grad*b*y/Q))


    tsl_normal = np.vstack([x,y])
    tsl_tilted = rotation_matrix.dot(tsl_normal)

    final_x = tsl_tilted[0,:] + lon_brunnen
    final_y = tsl_tilted[1,:] + lat_brunnen

    # remove negatives
    mask = final_x >= 0
    final_x = final_x[mask]
    final_y = final_y[mask]


    x_plot, y_plot = transformer.transform(final_x,final_y)

    return y_plot, x_plot


y_plot_init, x_plot_init = getXYtrennstromlinie(Q=100, K=0.0006, grad=0.0017, b=120, winkel=200)
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
               zoom = 12,
               center={
               "lat":brunnen_x_plot,
               "lon":brunnen_y_plot})
fig.update_layout(uirevision="Don't change"
    )

@app.callback(
    Output("graph", "figure"),
    Input("Q", "value"),
    Input("K", "value"),
    Input("grad", "value"),
    Input("b", "value"),
    Input("winkel", "value"))
def TSL_conf(Q, K, grad, b, winkel):

    y_plot, x_plot = getXYtrennstromlinie(Q, K, grad,b ,winkel)
    fig.update_traces(lon = y_plot,
                    lat = x_plot,
                  selector=dict(mode="lines"))

    return fig

server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)

"""Folium map of one trip: GPS observations, nearby road network, reference route and
(optionally) a predicted route. Reusable by the API/UI: make_trip_map(trip, pred_arcs=...)."""
import numpy as np
import folium
from candidates import RoadNet

LEGEND = """
<div style="position: fixed; bottom: 24px; left: 24px; z-index: 9999; background: white;
 padding: 8px 12px; border: 1px solid #888; border-radius: 6px; font: 12px/1.5 sans-serif;">
<b>{title}</b><br>
<span style="color:#1f77b4">&#9679;</span> Raw GPS observations<br>
<span style="color:#888">&#9473;</span> Road network (arcs near the track)<br>
<span style="color:#2ca02c">&#9473;</span> Reference route (supplied)<br>
{pred}<span style="color:#2ca02c">&#9679;</span> start &nbsp;<span style="color:#d62728">&#9679;</span> end
</div>"""


def _arc_lines(net, trip, arc_ids):
    nodes, arcs = trip["nodes"], trip["arcs"]
    a, b = nodes[arcs[arc_ids, 0]], nodes[arcs[arc_ids, 1]]
    return [[[p[1], p[0]], [q[1], q[0]]] for p, q in zip(a, b)]   # folium wants (lat, lon)


def make_trip_map(trip, out_html, pred_arcs=None, title=None, radius=100.0, max_points=3000):
    net = RoadNet(trip)
    tr = trip["track"]
    P = net.project_points(tr[:, 0], tr[:, 1])
    _, near = net.candidates(P, radius=radius, k=500, fallback=radius)
    net_arcs = np.unique(near)
    route = trip.get("route")
    m = folium.Map(location=[tr[:, 1].mean(), tr[:, 0].mean()], zoom_start=15, tiles=None)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", name="Light basemap").add_to(m)

    g = folium.FeatureGroup(name=f"Road network ({len(net_arcs)} arcs near track)", show=True)
    folium.PolyLine(_arc_lines(net, trip, net_arcs), color="#888888", weight=2, opacity=0.7).add_to(g)
    g.add_to(m)

    if route is not None:
        g = folium.FeatureGroup(name=f"Reference route ({len(route)} arcs)", show=True)
        # wide, semi-transparent halo so GPS dots stay visible on top of it
        folium.PolyLine(_arc_lines(net, trip, route), color="#2ca02c", weight=9, opacity=0.45,
                        tooltip="reference route").add_to(g)
        g.add_to(m)

    if pred_arcs is not None:
        g = folium.FeatureGroup(name="Predicted route", show=True)
        folium.PolyLine(_arc_lines(net, trip, np.unique(pred_arcs)), color="#d62728", weight=4, opacity=0.9,
                        tooltip="predicted route").add_to(g)
        g.add_to(m)

    g = folium.FeatureGroup(name="GPS observations", show=True)
    step = max(1, len(tr) // max_points)
    for i in range(0, len(tr), step):
        folium.CircleMarker([tr[i, 1], tr[i, 0]], radius=3, color="#ffffff", fill=True, fillColor="#1f77b4",
                            fill_opacity=0.95, weight=0.6, tooltip=f"pt {i}, t={tr[i, 2]:.0f}s").add_to(g)
    g.add_to(m)
    folium.Marker([tr[0, 1], tr[0, 0]], tooltip="start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker([tr[-1, 1], tr[-1, 0]], tooltip="end", icon=folium.Icon(color="red")).add_to(m)

    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    pred_row = '<span style="color:#d62728">&#9473;</span> Predicted route<br>' if pred_arcs is not None else ""
    m.get_root().html.add_child(folium.Element(
        LEGEND.format(title=title or f"Trip {trip['trip_id']}", pred=pred_row)))
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(out_html)
    return out_html

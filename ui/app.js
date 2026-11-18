/* PathSnap evaluator UI. All route data and metrics come from the API. */
"use strict";
const $ = (id) => document.getElementById(id);
const state = {
  api: "",
  trips: [],
  selected: null,
  result: null,
  online: false,
  busy: false,
  connecting: false,
  map: null,
  gps: null,
  gpsRenderer: null,
  route: null,
  endpoints: null,
  bounds: null,
};
const number = new Intl.NumberFormat();
const percent = (v) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}
function setStatus(online, label) {
  state.online = online;
  $("connection-dot").className = `status-dot ${online ? "online" : "offline"}`;
  $("connection-label").textContent = label;
  $("run-button").disabled = !online || !state.selected || state.busy;
}
function panel(open) {
  $("connection-panel").hidden = !open;
  $("connection-toggle").setAttribute("aria-expanded", String(open));
}
function normalizeUrl(value) {
  const url = new URL(value);
  if (
    !["http:", "https:"].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  ) {
    throw new Error(
      "Enter an HTTP or HTTPS API address, for example http://192.168.1.10:8000.",
    );
  }
  return url.href.replace(/\/+$/, "");
}
async function api(path, options = {}) {
  const controller = new AbortController();
  const timeout = setTimeout(
    () => controller.abort(),
    path === "/predict" ? 120000 : 12000,
  );
  try {
    const response = await fetch(`${state.api}${path}`, {
      ...options,
      signal: controller.signal,
    });
    if (!response.ok) {
      let detail;
      try {
        detail = (await response.json()).detail;
      } catch (_) {
        /* e.g. a wrong server URL */
      }
      throw new Error(
        typeof detail === "string"
          ? detail
          : `API returned HTTP ${response.status}. Check the API terminal.`,
      );
    }
    return await response.json();
  } catch (error) {
    if (error.name === "AbortError")
      throw new Error("The API timed out. Check its terminal, then try again.");
    if (error instanceof TypeError)
      throw new Error(
        "Cannot reach the API. Check the address, Wi-Fi and laptop 1’s firewall.",
      );
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}
function setTrip(trip) {
  if (state.busy) return;
  state.selected = trip;
  if (state.result?.trip_id !== trip) resetResult();
  $("workspace-title").textContent = `Trip ${trip}`;
  $("workspace-subtitle").textContent =
    "A GPS recording, ready to find its road.";
  for (const button of $("trip-list").querySelectorAll("button")) {
    button.setAttribute("aria-pressed", String(button.dataset.trip === trip));
  }
  $("run-button").disabled = !state.online;
}
function renderTrips() {
  $("trip-list").replaceChildren();
  $("trip-count").textContent = state.trips.length;
  state.trips.forEach((trip, index) => {
    const button = document.createElement("button");
    button.className = "trip-option";
    button.dataset.trip = trip;
    button.setAttribute("aria-pressed", "false");
    button.setAttribute("aria-label", `Select trip ${trip}`);
    // Only static markup enters innerHTML. API strings enter textContent.
    button.innerHTML =
      '<span class="trip-index"></span><span class="trip-copy"><span class="trip-id"></span><span class="trip-meta">GPS recording</span></span><span class="trip-arrow" aria-hidden="true">↗</span>';
    button.querySelector(".trip-index").textContent = String(
      index + 1,
    ).padStart(2, "0");
    button.querySelector(".trip-id").textContent = trip;
    button.addEventListener("click", () => setTrip(trip));
    $("trip-list").append(button);
  });
  if (!state.trips.length) {
    $("trip-list").textContent =
      "No complete trips found. Check the data folder on the API laptop.";
    state.selected = null;
    $("run-button").disabled = true;
  } else
    setTrip(
      state.trips.includes(state.selected) ? state.selected : state.trips[0],
    );
}
async function connect(value) {
  if (state.busy || state.connecting) return;
  state.connecting = true;
  const submitButton = $("connection-form").querySelector("button");
  submitButton.disabled = true;
  try {
    state.api = normalizeUrl(value);
    setStatus(false, "Connecting");
    showError("");
    resetResult();
    const health = await api("/health");
    if (health.status !== "ok")
      throw new Error("The API is not ready. Check its terminal.");
    const library = await api("/trips");
    if (
      !Array.isArray(library.trips) ||
      library.trips.some((t) => typeof t !== "string")
    ) {
      throw new Error("This address did not return a PathSnap trip library.");
    }
    state.trips = library.trips;
    try {
      localStorage.setItem("pathsnap-api", state.api);
    } catch (_) {
      /* private browser */
    }
    setStatus(true, "API connected");
    renderTrips();
    panel(false);
    $("notice").hidden = health.mode !== "nearest";
    $("notice").textContent =
      "Development mode · Nearest-arc dummy predictions. Trained model and HMM are not active.";
  } catch (error) {
    setStatus(false, "API offline");
    state.trips = [];
    state.selected = null;
    $("trip-count").textContent = "0";
    $("trip-list").textContent = "Connect to the API to load your trips.";
    $("notice").hidden = true;
    showError(error.message);
    panel(true);
  } finally {
    state.connecting = false;
    submitButton.disabled = false;
  }
}
function initMap() {
  if (state.map) return;
  if (!window.L)
    throw new Error(
      "The local map library could not load. Reload the page or check ui/vendor/leaflet.",
    );
  state.map = L.map("map", {
    zoomControl: true,
    preferCanvas: true,
    attributionControl: true,
  });
  state.map.setView([20, 0], 2);
  const tiles = L.tileLayer(
    "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    {
      maxZoom: 20,
      subdomains: "abcd",
      attribution:
        '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
    },
  ).addTo(state.map);
  tiles.on("tileerror", () => {
    $("tile-notice").hidden = false;
  });
  tiles.on("tileload", () => {
    $("tile-notice").hidden = true;
  });
  L.control
    .scale({ imperial: false, position: "bottomright" })
    .addTo(state.map);
}
function resetResult() {
  state.result = null;
  for (const key of ["accuracy", "continuity", "points", "latency"])
    $(key).textContent = "—";
  $("accuracy-note").textContent = "Awaiting a prediction";
  $("matched-note").textContent = "Points in this recording";
  $("map-empty").hidden = false;
  $("map-legend").hidden = true;
  $("map-title").textContent = "Geographical view";
  $("map-caption").textContent = "Your next route, in context.";
  $("map-dot").className = "status-dot";
  $("run-state").textContent = "READY WHEN YOU ARE";
  $("run-notes").innerHTML =
    '<p>One trip, one request. The map and metrics appear here when the API returns your result.</p><span class="notes-arrow" aria-hidden="true">↗</span>';
  $("technical-details").hidden = true;
  $("technical-details").open = false;
  $("export-button").disabled = true;
  $("fit-button").disabled = true;
  $("tile-notice").hidden = true;
  for (const key of ["gps", "route", "endpoints"]) {
    if (state[key]) {
      state.map.removeLayer(state[key]);
      state[key] = null;
    }
  }
  state.bounds = null;
}
function detail(container, label, value) {
  const item = document.createElement("div");
  item.textContent = label;
  const strong = document.createElement("strong");
  strong.textContent = value;
  item.append(strong);
  container.append(item);
}
function renderResult(result) {
  if (
    !Array.isArray(result.gps) ||
    !result.gps.length ||
    !Array.isArray(result.predicted_route)
  ) {
    throw new Error("The API returned an invalid route. Check its terminal.");
  }
  initMap();
  for (const key of ["gps", "route", "endpoints"])
    if (state[key]) state.map.removeLayer(state[key]);
  state.result = result;
  state.route = L.featureGroup(
    result.predicted_route.map((segment) =>
      L.polyline(segment, {
        color: "#c06039",
        weight: 4,
        opacity: 0.88,
        lineCap: "round",
      }).bindTooltip("Predicted road segment"),
    ),
  );
  const gpsRenderer = (state.gpsRenderer ||= L.canvas({ padding: 0.5 }));
  state.gps = L.featureGroup(
    result.gps.map((point, index) =>
      L.circleMarker(point, {
        renderer: gpsRenderer,
        radius: 2.7,
        color: "#427ca8",
        weight: 0.5,
        fillColor: "#427ca8",
        fillOpacity: 0.65,
      }).bindTooltip(
        `GPS point ${index + 1} · predicted arc ${result.predicted_arcs[index] < 0 ? "unmatched" : result.predicted_arcs[index]}`,
      ),
    ),
  );
  if ($("show-route").checked) state.route.addTo(state.map);
  if ($("show-gps").checked) state.gps.addTo(state.map);
  state.endpoints = L.featureGroup([
    L.marker(result.gps[0], {
      icon: L.divIcon({
        className: "endpoint-marker",
        html: "S",
        iconSize: [23, 23],
        iconAnchor: [11, 11],
      }),
    }).bindTooltip("Start of GPS recording"),
    L.marker(result.gps.at(-1), {
      icon: L.divIcon({
        className: "endpoint-marker end",
        html: "E",
        iconSize: [23, 23],
        iconAnchor: [11, 11],
      }),
    }).bindTooltip("End of GPS recording"),
  ]).addTo(state.map);
  state.bounds = L.latLngBounds(result.gps);
  for (const segment of result.predicted_route)
    state.bounds.extend(L.latLngBounds(segment));
  $("map-empty").hidden = true;
  $("map-legend").hidden = false;
  $("map-title").textContent = `Trip ${result.trip_id} / predicted route`;
  $("map-dot").className = "status-dot online";
  $("map-caption").textContent =
    `${number.format(result.predicted_route.length)} predicted road segments · ${number.format(result.n_points)} GPS observations`;
  $("accuracy").textContent = percent(result.metrics?.pointwise_accuracy);
  $("accuracy-note").textContent = result.metrics
    ? "Scored on the API laptop"
    : "Reference unavailable · unscored";
  $("continuity").textContent = percent(result.metrics?.route_continuity);
  $("points").textContent = number.format(result.n_points);
  $("matched-note").textContent =
    `${number.format(result.n_matched)} matched · ${number.format(result.n_points - result.n_matched)} unmatched`;
  const latency = result.timing_ms.total;
  $("latency").replaceChildren(
    document.createTextNode(
      latency >= 1000
        ? (latency / 1000).toFixed(2)
        : Math.round(latency).toString(),
    ),
  );
  const unit = document.createElement("small");
  unit.textContent = latency >= 1000 ? "s" : "ms";
  $("latency").append(unit);
  $("run-state").textContent = "PREDICTION COMPLETE";
  $("run-notes").replaceChildren();
  const grid = document.createElement("div");
  grid.className = "run-note-grid";
  detail(grid, "Prediction method", result.pipeline);
  detail(
    grid,
    "Completed at",
    new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }),
  );
  detail(grid, "Request", result.request_id);
  $("run-notes").append(grid);
  $("technical-content").replaceChildren();
  for (const [label, value] of [
    ["Feature preparation", `${result.timing_ms.features.toFixed(1)} ms`],
    ["Prediction", `${result.timing_ms.predict.toFixed(1)} ms`],
    ["Server scoring", `${result.timing_ms.scoring.toFixed(1)} ms`],
    [
      "Classifier-only accuracy",
      percent(result.metrics?.pointwise_accuracy_classifier_only),
    ],
    [
      "50-point exact windows",
      percent(result.metrics?.sequence_window50_accuracy),
    ],
    [
      "Reference coverage within 60 m",
      percent(result.metrics?.frac_points_reference_covered),
    ],
  ]) {
    detail($("technical-content"), label, value);
  }
  const explanation = document.createElement("p");
  explanation.textContent =
    "Accuracy uses point labels aligned to the supplied route on the server. An unmatched point is incorrect. Exact-window accuracy needs at least 50 points. Timings exclude network transfer and browser rendering.";
  $("technical-content").append(explanation);
  $("technical-details").hidden = false;
  $("export-button").disabled = false;
  $("fit-button").disabled = false;
  $("notice").hidden = result.mode !== "nearest";
  requestAnimationFrame(() => {
    state.map.invalidateSize();
    fitRoute();
  });
}
function fitRoute() {
  if (state.bounds)
    state.map.fitBounds(state.bounds, {
      padding: [35, 45],
      maxZoom: 17,
      animate: false,
    });
}
async function run() {
  if (state.busy || !state.selected) return;
  state.busy = true;
  showError("");
  resetResult();
  $("workspace").setAttribute("aria-busy", "true");
  $("run-button").disabled = true;
  $("run-label").textContent = "Matching trip…";
  $("map-loading").hidden = false;
  $("connection-toggle").disabled = true;
  for (const button of $("trip-list").querySelectorAll("button"))
    button.disabled = true;
  $("run-state").textContent = "REQUEST IN PROGRESS";
  try {
    const result = await api("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trip_id: state.selected }),
    });
    renderResult(result);
  } catch (error) {
    showError(error.message);
    $("run-state").textContent = "RUN UNSUCCESSFUL";
  } finally {
    state.busy = false;
    $("run-button").disabled = !state.online;
    $("workspace").setAttribute("aria-busy", "false");
    $("run-label").textContent = "Run prediction";
    $("map-loading").hidden = true;
    $("connection-toggle").disabled = false;
    for (const button of $("trip-list").querySelectorAll("button"))
      button.disabled = false;
  }
}
$("connection-toggle").addEventListener("click", () =>
  panel($("connection-panel").hidden),
);
$("connection-form").addEventListener("submit", (event) => {
  event.preventDefault();
  connect($("api-url").value);
});
$("run-button").addEventListener("click", run);
$("fit-button").addEventListener("click", fitRoute);
for (const [id, key] of [
  ["show-gps", "gps"],
  ["show-route", "route"],
]) {
  $(id).addEventListener("change", () => {
    if (state[key])
      $(id).checked
        ? state[key].addTo(state.map)
        : state.map.removeLayer(state[key]);
  });
}
$("export-button").addEventListener("click", () => {
  if (!state.result) return;
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(state.result, null, 2)], {
      type: "application/json",
    }),
  );
  const link = document.createElement("a");
  link.href = url;
  link.download = `pathsnap-${state.result.trip_id}.json`;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
let initial;
try {
  initial =
    new URLSearchParams(location.search).get("api") ||
    localStorage.getItem("pathsnap-api");
} catch (_) {
  /* restricted browser */
}
initial ||= location.pathname.startsWith("/ui")
  ? location.origin
  : `${location.protocol}//${location.hostname || "127.0.0.1"}:8000`;
$("api-url").value = initial;
connect(initial);

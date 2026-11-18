/* DOM-level tests of the shipped UI code; these are not visual browser tests. */
const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { JSDOM } = require('jsdom');
const root = path.join(__dirname, '..');
const html = fs.readFileSync(path.join(root, 'ui/index.html'), 'utf8');
const code = fs.readFileSync(path.join(root, 'ui/app.js'), 'utf8');
const result = {
  trip_id: '00000000', request_id: 'test-request', mode: 'nearest', pipeline: 'dummy / nearest arc',
  n_points: 2, n_matched: 1, gps: [[12, 77], [12.001, 77.001]], predicted_arcs: [0, -1],
  predicted_route: [[[12, 77], [12.001, 77.001]]], metrics: null,
  timing_ms: { features: 2, predict: 1, scoring: 1, total: 4 }
};
function leaflet() {
  const layers = new Set();
  const map = { setView() {}, invalidateSize() {}, fitBounds() {}, removeLayer(l) { layers.delete(l); } };
  const layer = () => ({ addTo() { layers.add(this); return this; }, bindTooltip() { return this; }, on() { return this; } });
  return { map: () => map, tileLayer: layer, featureGroup: layer, polyline: layer, canvas: () => ({}),
    circleMarker: layer, marker: layer, divIcon: x => x, control: { scale: layer },
    latLngBounds: () => ({ extend() {} }), layers };
}
async function until(check) {
  for (let i = 0; i < 100; i++) {
    if (check()) return;
    await new Promise(resolve => setTimeout(resolve, 5));
  }
  assert.fail('UI operation did not finish');
}
async function setup(fetcher) {
  const dom = new JSDOM(html, { url: 'http://127.0.0.1:8000/ui/', runScripts: 'outside-only', pretendToBeVisual: true });
  const calls = [];
  dom.window.L = leaflet();
  dom.window.fetch = async (url, options) => {
    calls.push({ url, options });
    return fetcher(url, options);
  };
  dom.window.eval(code);
  await until(() => dom.window.document.getElementById('connection-label').textContent !== 'Connecting');
  return { dom, calls, doc: dom.window.document };
}
const response = value => ({ ok: true, json: async () => value });
const normal = async url => response(url.endsWith('/health') ? { status: 'ok', mode: 'nearest' }
  : url.endsWith('/trips') ? { trips: ['00000000', '00000007'] } : result);

test('loads trip IDs from API, runs selected trip and toggles real result layers', async () => {
  const { dom, doc, calls } = await setup(normal);
  try {
    assert.equal(doc.querySelectorAll('.trip-option').length, 2);
    assert.equal(doc.querySelector('.trip-id').textContent, '00000000');
    assert.equal(doc.getElementById('notice').hidden, false);
    doc.getElementById('run-button').click();
    await until(() => doc.getElementById('run-state').textContent === 'PREDICTION COMPLETE');
    const request = calls.find(call => call.url.endsWith('/predict'));
    assert.deepEqual(JSON.parse(request.options.body), { trip_id: '00000000' });
    assert.equal(doc.getElementById('points').textContent, '2');
    assert.equal(doc.getElementById('accuracy').textContent, '—');
    assert.match(doc.getElementById('accuracy-note').textContent, /unscored/);
    assert.equal(doc.getElementById('map-empty').hidden, true);
    assert.equal(doc.getElementById('map-legend').hidden, false);
    assert.equal(doc.getElementById('export-button').disabled, false);
    const count = dom.window.L.layers.size;
    doc.getElementById('show-route').click();
    assert.equal(dom.window.L.layers.size, count - 1);
    doc.getElementById('show-route').click();
    assert.equal(dom.window.L.layers.size, count);
    doc.querySelectorAll('.trip-option')[1].click();
    assert.match(doc.getElementById('workspace-title').textContent, /00000007/);
    assert.equal(doc.getElementById('map-empty').hidden, false);
    assert.equal(doc.getElementById('points').textContent, '—');
    assert.equal(doc.getElementById('export-button').disabled, true);
  } finally { dom.window.close(); }
});

test('renders server metrics and timings, including a valid zero accuracy', async () => {
  const scored = { ...result, metrics: { pointwise_accuracy: 0, route_continuity: .75,
    pointwise_accuracy_classifier_only: .25, sequence_window50_accuracy: null, frac_points_reference_covered: 1 } };
  const { dom, doc } = await setup(url => url.endsWith('/predict') ? response(scored) : normal(url));
  try {
    doc.getElementById('run-button').click();
    await until(() => doc.getElementById('run-state').textContent === 'PREDICTION COMPLETE');
    assert.equal(doc.getElementById('accuracy').textContent, '0.0%');
    assert.equal(doc.getElementById('continuity').textContent, '75.0%');
    assert.equal(doc.getElementById('latency').textContent, '4ms');
    assert.equal(doc.getElementById('technical-details').hidden, false);
    assert.match(doc.getElementById('run-notes').textContent, /test-request/);
  } finally { dom.window.close(); }
});

test('unreachable API shows a connection error and prevents prediction', async () => {
  const { dom, doc } = await setup(async () => { throw new TypeError('offline'); });
  try {
    assert.equal(doc.getElementById('run-button').disabled, true);
    assert.equal(doc.getElementById('connection-panel').hidden, false);
    assert.equal(doc.getElementById('error').hidden, false);
    assert.equal(doc.getElementById('trip-count').textContent, '0');
  } finally { dom.window.close(); }
});

test('failed prediction clears stale data and leaves a usable retry button', async () => {
  const { dom, doc } = await setup(async url => url.endsWith('/predict')
    ? { ok: false, status: 500, json: async () => ({ detail: 'Check the API terminal.' }) } : normal(url));
  try {
    doc.getElementById('run-button').click();
    await until(() => doc.getElementById('run-state').textContent === 'RUN UNSUCCESSFUL');
    assert.equal(doc.getElementById('error').textContent, 'Check the API terminal.');
    assert.equal(doc.getElementById('points').textContent, '—');
    assert.equal(doc.getElementById('map-loading').hidden, true);
    assert.equal(doc.getElementById('run-button').disabled, false);
  } finally { dom.window.close(); }
});

test('API trip IDs are treated as text rather than inserted as HTML', async () => {
  const id = '<img src=x onerror=alert(1)>';
  const { dom, doc } = await setup(url => url.endsWith('/trips') ? response({ trips: [id] }) : normal(url));
  try {
    assert.equal(doc.querySelector('.trip-id').textContent, id);
    assert.equal(doc.querySelectorAll('.trip-option img').length, 0);
  } finally { dom.window.close(); }
});

// new_fire.js — bbox drawer + form for /new_fire
//
// Layout: an <img> overview underneath an absolutely-positioned <canvas>
// of identical dimensions. The user drags on the canvas to mark the
// VIIRS download AOI; pixel <-> raster-CRS <-> WGS84 conversions are
// driven by the year's overview metadata (geotransform + raster_W/H +
// extent_wgs84).

(function () {
'use strict';

const yearSelect = document.getElementById('nf-year');
const overview = document.getElementById('nf-overview');
const canvas = document.getElementById('nf-canvas');
const coordsEl = document.getElementById('nf-coords');
const clearBtn = document.getElementById('nf-clear-bbox');
const errorsEl = document.getElementById('nf-errors');
const submitBtn = document.getElementById('nf-submit');
const previewBtn = document.getElementById('nf-preview');
const previewStatus = document.getElementById('nf-preview-status');
const previewStages = document.getElementById('nf-preview-stages');
const previewWrap = document.getElementById('nf-preview-wrap');
const previewImg = document.getElementById('nf-preview-img');
const previewMeta = document.getElementById('nf-preview-meta');

// Stages walked client-side while the single /api/fire/preview_hint
// request is in flight. The server is one-shot, but the user wants
// to see *what* it's doing — so we sequence pills on heuristic
// timers (validate is instant, accumulate dominates, rasterize +
// generate are fast tail). Order matches fire_list.handle_api_fire_preview_hint.
const PREVIEW_STAGES = [
    {key: 'validate',   label: 'Validating', delayMs: 0},
    {key: 'accumulate', label: 'Accumulating VIIRS', delayMs: 250},
    {key: 'rasterize',  label: 'Rasterizing', delayMs: 4000},
    {key: 'generate',   label: 'Generating preview', delayMs: 6500},
];

const fields = {
    name: document.getElementById('nf-name'),
    xmin: document.getElementById('nf-xmin'),
    ymin: document.getElementById('nf-ymin'),
    xmax: document.getElementById('nf-xmax'),
    ymax: document.getElementById('nf-ymax'),
    w: document.getElementById('nf-w'),
    e: document.getElementById('nf-e'),
    s: document.getElementById('nf-s'),
    n: document.getElementById('nf-n'),
    start: document.getElementById('nf-start'),
    end: document.getElementById('nf-end'),
    fireDate: document.getElementById('nf-fire-date'),
};

let meta = null;
let bbox = null;          // {x0, y0, x1, y1} in canvas pixels (top-left origin)
let drag = null;          // {kind: 'create'|'move', startMx, startMy, origBbox?}
let lastPreview = null;   // {preview_id, year, start, end, bbox_native} —
                          // sent in the create body when the form still
                          // matches, so the worker can skip accumulate.
// Generation counter — bumped on every input the preview depends on
// (bbox draw/move, year, dates, manual clear). The preview request
// captures this at start; on response we drop the result if the
// generation moved, so a bbox redrawn mid-flight cannot adopt a
// stale preview_id or leave a wrong hint image on screen.
let previewGen = 0;
let previewInflightGen = -1;
let previewStageTimers = [];

// Populate year selector
NF_ALL_YEARS.forEach(y => {
    const opt = document.createElement('option');
    opt.value = String(y);
    opt.textContent = String(y);
    if (y === NF_ACTIVE_YEAR) opt.selected = true;
    yearSelect.appendChild(opt);
});
if (!NF_MULTI_YEAR || !NF_IS_ADMIN) {
    yearSelect.disabled = !NF_MULTI_YEAR ? true : !NF_IS_ADMIN;
    if (NF_MULTI_YEAR && !NF_IS_ADMIN) {
        const note = document.createElement('span');
        note.style.color = '#666';
        note.style.fontSize = '11px';
        note.textContent = '(admin only)';
        yearSelect.parentElement.appendChild(note);
    }
}

yearSelect.addEventListener('change', () => loadYear(parseInt(yearSelect.value, 10)));

function clearErrors() { errorsEl.innerHTML = ''; }
function showErrors(errs) {
    errorsEl.innerHTML = '';
    if (!errs || !errs.length) return;
    const ul = document.createElement('ul');
    errs.forEach(e => {
        const li = document.createElement('li');
        const f = e.field ? `[${e.field}] ` : '';
        li.textContent = f + (e.message || String(e));
        ul.appendChild(li);
    });
    errorsEl.appendChild(ul);
}

async function loadYear(year) {
    clearErrors();
    bbox = null;
    try {
        const r = await fetch(`/api/year/${year}/overview_meta`);
        if (!r.ok) {
            showErrors([{message: `Failed to load year ${year} metadata`}]);
            return;
        }
        meta = await r.json();
    } catch (exc) {
        showErrors([{message: `Network error: ${exc}`}]);
        return;
    }
    overview.src = `/api/year/${year}/overview.png?t=${Date.now()}`;
    overview.onload = () => {
        canvas.width = overview.clientWidth;
        canvas.height = overview.clientHeight;
        // Resize on window resize too
        redraw();
    };
    fields.start.placeholder = meta.default_start || 'YYYY-MM-DD';
    fields.end.placeholder = meta.default_end || 'YYYY-MM-DD';
    if (!fields.start.value) fields.start.value = '';
    if (!fields.end.value) fields.end.value = '';
    refreshFireDatePlaceholder();
}

function refreshFireDatePlaceholder() {
    if (!fields.fireDate) return;
    const eff = fields.end.value.trim() || fields.end.placeholder || '';
    fields.fireDate.placeholder = eff || 'YYYY-MM-DD';
}

if (fields.fireDate) {
    fields.end.addEventListener('input', refreshFireDatePlaceholder);
}

window.addEventListener('resize', () => {
    if (!overview.complete) return;
    canvas.width = overview.clientWidth;
    canvas.height = overview.clientHeight;
    redraw();
});

// ----- Coordinate conversions -----

function canvasToRasterPx(mx, my) {
    if (!meta) return null;
    // Canvas pixels → overview pixels (canvas matches overview client dims)
    const sx = meta.overview_W / canvas.width;
    const sy = meta.overview_H / canvas.height;
    const ovx = mx * sx;
    const ovy = my * sy;
    // Overview px → raster px
    const rx = ovx * (meta.raster_W / meta.overview_W);
    const ry = ovy * (meta.raster_H / meta.overview_H);
    return [rx, ry];
}

function rasterPxToNative(rx, ry) {
    const gt = meta.geotransform;
    const x = gt[0] + rx * gt[1] + ry * gt[2];
    const y = gt[3] + rx * gt[4] + ry * gt[5];
    return [x, y];
}

function canvasToNative(mx, my) {
    const rp = canvasToRasterPx(mx, my);
    if (!rp) return null;
    return rasterPxToNative(rp[0], rp[1]);
}

// Inverse of the affine geotransform — needed when the user types
// raster-CRS bbox coords directly and we have to draw the rectangle on
// the canvas to match.
function nativeToRasterPx(x, y) {
    const gt = meta.geotransform;
    const det = gt[1] * gt[5] - gt[2] * gt[4];
    if (!det) return null;
    const dx = x - gt[0], dy = y - gt[3];
    const rx = (dx * gt[5] - dy * gt[2]) / det;
    const ry = (-dx * gt[4] + dy * gt[1]) / det;
    return [rx, ry];
}

function nativeToCanvas(x, y) {
    if (!meta) return null;
    const rp = nativeToRasterPx(x, y);
    if (!rp) return null;
    const cx = rp[0] * (canvas.width / meta.raster_W);
    const cy = rp[1] * (canvas.height / meta.raster_H);
    return [cx, cy];
}

function nativeBboxToWGS84(xmin, ymin, xmax, ymax) {
    // Linear interpolation through extent_native ↔ extent_wgs84 corners.
    if (!meta || !meta.extent_native || !meta.extent_wgs84) return null;
    const [rxmin, rymin, rxmax, rymax] = meta.extent_native;
    const [w, s, e, n] = meta.extent_wgs84;
    function lerp_x(x) { return w + ((x - rxmin) / (rxmax - rxmin)) * (e - w); }
    function lerp_y(y) { return s + ((y - rymin) / (rymax - rymin)) * (n - s); }
    const W = lerp_x(xmin);
    const E = lerp_x(xmax);
    const S = lerp_y(ymin);
    const N = lerp_y(ymax);
    return [W, S, E, N];
}

// ----- Drawing -----

let bcwsOverlay = null;  // {points: [[x,y],...], polygons: [[[x,y],...]]} in raster-native CRS
// Forward-declared here (assigned further down, alongside the rest of
// the zoom machinery) so drawBcwsOverlay() -- which runs earlier in
// some call paths -- never references it before initialization.
let zoomScale = 1;

async function loadBcwsOverlay() {
    try {
        const r = await fetch('/api/bcws/overlay');
        if (!r.ok) return;
        bcwsOverlay = await r.json();
    } catch (exc) {
        // Non-fatal -- the bbox drawer still works without the overlay.
        bcwsOverlay = null;
    }
    redraw();
}

function drawBcwsOverlay(ctx) {
    if (!bcwsOverlay || !meta) return;
    const polys = bcwsOverlay.polygons || [];
    const pts = bcwsOverlay.points || [];

    // Zoom is applied as a CSS transform: scale() on .nf-zoom-inner
    // (see applyZoom()), which scales the canvas's drawn pixels right
    // along with everything else -- so without compensating here, a
    // 1px line or 4px dot grows on screen as you zoom in. Dividing by
    // the current zoomScale keeps their ON-SCREEN size pinned to
    // whatever it is at zoomScale=1 (i.e. "Reset zoom"), regardless
    // of how far in/out the user has zoomed.
    const z = (typeof zoomScale === 'number' && zoomScale > 0) ? zoomScale : 1;
    const lineWidthPx = 1 / z;     // exactly 1px wide at reset-zoom, always
    const pointRadiusPx = 4 / z;   // same on-screen radius at any zoom

    ctx.strokeStyle = 'rgba(220, 0, 0, 0.9)';
    ctx.fillStyle = 'rgba(220, 0, 0, 0.18)';
    ctx.lineWidth = lineWidthPx;
    polys.forEach((ring) => {
        if (!ring || ring.length < 2) return;
        ctx.beginPath();
        ring.forEach(([x, y], i) => {
            const cp = nativeToCanvas(x, y);
            if (!cp) return;
            if (i === 0) ctx.moveTo(cp[0], cp[1]);
            else ctx.lineTo(cp[0], cp[1]);
        });
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
    });

    ctx.fillStyle = 'rgba(220, 0, 0, 0.95)';
    pts.forEach(([x, y]) => {
        const cp = nativeToCanvas(x, y);
        if (!cp) return;
        ctx.beginPath();
        ctx.arc(cp[0], cp[1], pointRadiusPx, 0, 2 * Math.PI);
        ctx.fill();
    });
}

function redraw() {
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    drawBcwsOverlay(ctx);
    if (!bbox) return;
    const x = Math.min(bbox.x0, bbox.x1);
    const y = Math.min(bbox.y0, bbox.y1);
    const w = Math.abs(bbox.x1 - bbox.x0);
    const h = Math.abs(bbox.y1 - bbox.y0);
    ctx.fillStyle = 'rgba(255, 220, 70, 0.18)';
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = 'rgba(255, 180, 0, 0.95)';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
}

function updateReadout() {
    if (!bbox || !meta) return;
    const [xa, ya] = canvasToNative(bbox.x0, bbox.y0);
    const [xb, yb] = canvasToNative(bbox.x1, bbox.y1);
    const xmin = Math.min(xa, xb), xmax = Math.max(xa, xb);
    const ymin = Math.min(ya, yb), ymax = Math.max(ya, yb);
    fields.xmin.value = xmin.toFixed(2);
    fields.ymin.value = ymin.toFixed(2);
    fields.xmax.value = xmax.toFixed(2);
    fields.ymax.value = ymax.toFixed(2);
    const wgs = nativeBboxToWGS84(xmin, ymin, xmax, ymax);
    if (wgs) {
        fields.w.value = wgs[0].toFixed(6);
        fields.s.value = wgs[1].toFixed(6);
        fields.e.value = wgs[2].toFixed(6);
        fields.n.value = wgs[3].toFixed(6);
    }
}

function clearReadout() {
    Object.values(fields).forEach(f => {
        if (f.readOnly) f.value = '';
    });
    // Raster-CRS inputs are editable now, so clearReadout has to wipe
    // them explicitly when the bbox is dropped.
    ['xmin', 'ymin', 'xmax', 'ymax'].forEach(k => {
        if (fields[k]) fields[k].value = '';
    });
}

// Apply user-typed raster-CRS bbox values: rebuild the canvas rectangle
// and refresh the WGS84 readout. Returns true on success, false on
// invalid input (the user can keep editing without losing the rest).
function applyTypedRasterBbox() {
    if (!meta) return false;
    const xmin = parseFloat(fields.xmin.value);
    const ymin = parseFloat(fields.ymin.value);
    const xmax = parseFloat(fields.xmax.value);
    const ymax = parseFloat(fields.ymax.value);
    if (![xmin, ymin, xmax, ymax].every(Number.isFinite)) return false;
    if (xmin >= xmax || ymin >= ymax) return false;
    const tl = nativeToCanvas(xmin, ymax);  // upper-left corner
    const br = nativeToCanvas(xmax, ymin);  // lower-right corner
    if (!tl || !br) return false;
    bbox = {x0: tl[0], y0: tl[1], x1: br[0], y1: br[1]};
    redraw();
    // Refresh WGS84 readout (raster-CRS values are already what the
    // user typed — leave them alone so we don't fight their cursor).
    const wgs = nativeBboxToWGS84(xmin, ymin, xmax, ymax);
    if (wgs) {
        fields.w.value = wgs[0].toFixed(6);
        fields.s.value = wgs[1].toFixed(6);
        fields.e.value = wgs[2].toFixed(6);
        fields.n.value = wgs[3].toFixed(6);
    }
    invalidatePreview();
    return true;
}

// Wire each raster-CRS field. ``input`` fires on every keystroke so
// the WGS84 readout and the canvas rectangle track the user's edits
// in real time. We only touch the *other* fields, never the one being
// edited, so we don't fight the user's cursor.
['xmin', 'ymin', 'xmax', 'ymax'].forEach(k => {
    if (!fields[k]) return;
    fields[k].addEventListener('input', applyTypedRasterBbox);
});

// ----- Mouse handlers -----

function getMousePos(ev) {
    // getBoundingClientRect returns the post-CSS-transform size, so
    // dividing by it remaps cursor pixels back into canvas-buffer
    // pixels regardless of the current zoom scale.
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) {
        return [ev.clientX - rect.left, ev.clientY - rect.top];
    }
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    return [(ev.clientX - rect.left) * sx, (ev.clientY - rect.top) * sy];
}

function bboxContains(b, mx, my) {
    if (!b) return false;
    const x0 = Math.min(b.x0, b.x1), x1 = Math.max(b.x0, b.x1);
    const y0 = Math.min(b.y0, b.y1), y1 = Math.max(b.y0, b.y1);
    return mx >= x0 && mx <= x1 && my >= y0 && my <= y1;
}

canvas.addEventListener('mousedown', (ev) => {
    if (!meta) return;
    const [mx, my] = getMousePos(ev);
    if (bbox && bboxContains(bbox, mx, my)) {
        drag = {kind: 'move', startMx: mx, startMy: my,
                origBbox: Object.assign({}, bbox)};
    } else {
        bbox = {x0: mx, y0: my, x1: mx, y1: my};
        drag = {kind: 'create'};
        redraw();
    }
});

canvas.addEventListener('mousemove', (ev) => {
    if (!meta) return;
    const [mx, my] = getMousePos(ev);
    // Live coord readout in toolbar
    const native = canvasToNative(mx, my);
    if (native) {
        const wgs = nativeBboxToWGS84(native[0], native[1],
                                       native[0] + 0.001, native[1] + 0.001);
        const lon = wgs ? wgs[0].toFixed(4) : '?';
        const lat = wgs ? wgs[1].toFixed(4) : '?';
        coordsEl.textContent =
            `cursor: x=${native[0].toFixed(0)} y=${native[1].toFixed(0)}  ` +
            `(lon ${lon}, lat ${lat})`;
    }
    if (!drag) return;
    if (drag.kind === 'create') {
        bbox.x1 = mx;
        bbox.y1 = my;
    } else if (drag.kind === 'move') {
        const dx = mx - drag.startMx;
        const dy = my - drag.startMy;
        bbox.x0 = drag.origBbox.x0 + dx;
        bbox.y0 = drag.origBbox.y0 + dy;
        bbox.x1 = drag.origBbox.x1 + dx;
        bbox.y1 = drag.origBbox.y1 + dy;
    }
    redraw();
    updateReadout();
});

window.addEventListener('mouseup', () => {
    if (drag) {
        const wasDragging = drag;
        drag = null;
        if (bbox && Math.abs(bbox.x0 - bbox.x1) < 2
                 && Math.abs(bbox.y0 - bbox.y1) < 2) {
            bbox = null;
            clearReadout();
            redraw();
        } else {
            updateReadout();
        }
        // Any actual drag invalidates the preview cache.
        if (wasDragging) invalidatePreview();
    }
});

clearBtn.addEventListener('click', () => {
    bbox = null;
    clearReadout();
    redraw();
    invalidatePreview();
});

// Drop the cached preview reference when the user changes any input
// that would feed the create body. Bumping previewGen also poisons
// any in-flight preview request: when its response lands, the
// generation mismatch makes us discard the preview_id, hide the
// (stale) hint image, and keep the user from confirming a fire
// whose form bbox no longer matches the previewed one.
function invalidatePreview() {
    previewGen += 1;
    clearPreviewStageTimers();
    const hadCommitted = !!lastPreview;
    const hadInflight = previewInflightGen >= 0;
    lastPreview = null;
    if (hadCommitted || hadInflight) {
        if (previewWrap) previewWrap.style.display = 'none';
        if (previewImg) previewImg.removeAttribute('src');
        if (previewMeta) previewMeta.textContent = '';
        if (previewStages) previewStages.innerHTML = '';
        if (previewStatus) {
            previewStatus.textContent =
                hadInflight
                    ? 'Preview canceled — bbox/dates changed. Click again.'
                    : 'Preview is stale — click again.';
        }
    }
}
fields.start.addEventListener('input', invalidatePreview);
fields.end.addEventListener('input', invalidatePreview);
yearSelect.addEventListener('change', invalidatePreview);

function clearPreviewStageTimers() {
    for (const t of previewStageTimers) clearTimeout(t);
    previewStageTimers = [];
}

function renderPreviewStages(activeIdx) {
    if (!previewStages) return;
    previewStages.innerHTML = '';
    for (let i = 0; i < PREVIEW_STAGES.length; i++) {
        const pill = document.createElement('span');
        pill.className = 'progress-stage';
        if (i < activeIdx) pill.classList.add('stage-done');
        else if (i === activeIdx) pill.classList.add('stage-active');
        pill.textContent = PREVIEW_STAGES[i].label;
        previewStages.appendChild(pill);
    }
}

// ----- Preview Hint -----

function buildBodyForPreview() {
    const xmin = parseFloat(fields.xmin.value);
    const ymin = parseFloat(fields.ymin.value);
    const xmax = parseFloat(fields.xmax.value);
    const ymax = parseFloat(fields.ymax.value);
    return {
        year: parseInt(yearSelect.value, 10),
        bbox_native: [xmin, ymin, xmax, ymax],
        start: fields.start.value.trim(),
        end: fields.end.value.trim(),
    };
}

previewBtn.addEventListener('click', async () => {
    clearErrors();
    if (!bbox) {
        showErrors([{field: 'bbox_native',
                     message: 'Draw a bounding box on the overview first.'}]);
        return;
    }
    if (!meta) {
        showErrors([{field: 'year',
                     message: 'Year metadata not loaded yet.'}]);
        return;
    }
    // Capture the generation under which this request runs. If the
    // user changes the bbox/year/dates while we await, invalidatePreview
    // bumps previewGen and we drop our result on the floor below.
    const myGen = previewGen;
    previewInflightGen = myGen;
    previewBtn.disabled = true;
    previewWrap.style.display = 'none';
    previewMeta.textContent = '';
    if (previewImg) previewImg.removeAttribute('src');
    previewStatus.textContent = 'Working …';
    clearPreviewStageTimers();
    renderPreviewStages(0);
    for (let i = 1; i < PREVIEW_STAGES.length; i++) {
        const idx = i;
        previewStageTimers.push(setTimeout(() => {
            // A later request can have superseded us; only walk stages
            // for the request the user is actually waiting on.
            if (previewInflightGen === myGen) renderPreviewStages(idx);
        }, PREVIEW_STAGES[i].delayMs));
    }
    try {
        const body = buildBodyForPreview();
        const r = await fetch('/api/fire/preview_hint', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'fetch',
            },
            body: JSON.stringify(body),
        });
        const j = await r.json().catch(() => ({}));
        if (myGen !== previewGen) return;  // user moved on — discard
        if (!r.ok) {
            if (j.errors) showErrors(j.errors);
            else showErrors([{message: j.error || `HTTP ${r.status}`}]);
            previewStatus.textContent = '';
            clearPreviewStageTimers();
            if (previewStages) previewStages.innerHTML = '';
            return;
        }
        if (j.errors && j.errors.length) {
            showErrors(j.errors);
            previewStatus.textContent = '';
            clearPreviewStageTimers();
            if (previewStages) previewStages.innerHTML = '';
            return;
        }
        clearPreviewStageTimers();
        renderPreviewStages(PREVIEW_STAGES.length);  // all done
        previewWrap.style.display = '';
        previewImg.src = j.views.hint + '?t=' + Date.now();
        const start = j.start || '?';
        const end = j.end || '?';
        const area = (typeof j.area_ha === 'number')
            ? j.area_ha.toFixed(2) + ' ha' : '?';
        previewMeta.textContent =
            `Range: ${start} → ${end}   |   ` +
            `VIIRS hint area (within bbox): ${area}`;
        previewStatus.textContent = 'Preview ready (will be reused on Confirm).';
        lastPreview = {
            preview_id: j.preview_id,
            year: body.year,
            start: j.start,
            end: j.end,
            bbox_native: j.bbox_native || body.bbox_native,
        };
    } catch (exc) {
        if (myGen !== previewGen) return;
        showErrors([{message: `Network error: ${exc}`}]);
        previewStatus.textContent = '';
        clearPreviewStageTimers();
        if (previewStages) previewStages.innerHTML = '';
    } finally {
        if (previewInflightGen === myGen) previewInflightGen = -1;
        previewBtn.disabled = false;
    }
});

// ----- Submit -----

function bboxClose(a, b, tol = 1e-3) {
    if (!a || !b || a.length !== 4 || b.length !== 4) return false;
    for (let i = 0; i < 4; i++) {
        if (Math.abs(parseFloat(a[i]) - parseFloat(b[i])) > tol) return false;
    }
    return true;
}

submitBtn.addEventListener('click', async () => {
    clearErrors();
    if (!bbox) {
        showErrors([{field: 'bbox_native', message: 'Draw a bounding box on the overview first.'}]);
        return;
    }
    if (!meta) {
        showErrors([{field: 'year', message: 'Year metadata not loaded yet.'}]);
        return;
    }
    const xmin = parseFloat(fields.xmin.value);
    const ymin = parseFloat(fields.ymin.value);
    const xmax = parseFloat(fields.xmax.value);
    const ymax = parseFloat(fields.ymax.value);
    const body = {
        name: fields.name.value.trim(),
        year: parseInt(yearSelect.value, 10),
        bbox_native: [xmin, ymin, xmax, ymax],
        start: fields.start.value.trim(),
        end: fields.end.value.trim(),
        fire_date: (fields.fireDate
                    ? fields.fireDate.value.trim() : ''),
    };
    // Reuse the last preview's accumulate result if the form still
    // matches it exactly. The server re-validates, so a mismatch is a
    // silent fallback — never a hard error.
    if (lastPreview
            && lastPreview.year === body.year
            && bboxClose(lastPreview.bbox_native, body.bbox_native)) {
        // The server resolves empty start/end via the overview meta's
        // defaults, so test against the *placeholder* date when the
        // input is blank.
        const effStart = body.start || (meta.default_start || '');
        const effEnd = body.end || (meta.default_end || '');
        if (effStart === lastPreview.start && effEnd === lastPreview.end) {
            body.preview_id = lastPreview.preview_id;
        }
    }
    submitBtn.disabled = true;
    try {
        const r = await fetch('/api/fire/create', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Requested-With': 'fetch',
            },
            body: JSON.stringify(body),
        });
        if (r.status === 202) {
            const j = await r.json();
            window.location.href = '/';
            return;
        }
        const j = await r.json().catch(() => ({}));
        if (j.errors) {
            showErrors(j.errors);
        } else {
            showErrors([{message: j.error || `HTTP ${r.status}`}]);
        }
    } catch (exc) {
        showErrors([{message: `Network error: ${exc}`}]);
    } finally {
        submitBtn.disabled = false;
    }
});

// ----- Wheel zoom + reset -----
//
// We scale ``.nf-zoom-inner`` (which contains both the overview <img>
// and the bbox <canvas>) via CSS transform. The canvas draw buffer
// stays at its natural pixel size — getMousePos divides by
// getBoundingClientRect to remap cursor px → canvas px regardless of
// scale, so bbox math is unaffected.

const zoomWrap = document.getElementById('nf-canvas-wrap');
const zoomInner = document.getElementById('nf-zoom-inner');
const zoomResetBtn = document.getElementById('nf-zoom-reset');
let zoomTx = 0, zoomTy = 0;
const ZOOM_MIN = 1, ZOOM_MAX = 32;

function applyZoom() {
    if (!zoomInner) return;
    zoomInner.style.transform =
        `translate(${zoomTx}px, ${zoomTy}px) scale(${zoomScale})`;
}

function resetZoom() {
    zoomScale = 1;
    zoomTx = 0;
    zoomTy = 0;
    applyZoom();
    redraw();
}

function clampPan() {
    // Stop the scaled content from wandering beyond the wrap edges. At
    // scale=1 both bounds collapse to 0; otherwise we let the user pan
    // anywhere up to where the content edge meets the viewport edge.
    if (!zoomWrap || !zoomInner) return;
    const wrapW = zoomWrap.clientWidth;
    const wrapH = zoomWrap.clientHeight;
    const innerW = zoomInner.offsetWidth * zoomScale;
    const innerH = zoomInner.offsetHeight * zoomScale;
    const minTx = Math.min(0, wrapW - innerW);
    const minTy = Math.min(0, wrapH - innerH);
    if (zoomTx > 0) zoomTx = 0;
    if (zoomTy > 0) zoomTy = 0;
    if (zoomTx < minTx) zoomTx = minTx;
    if (zoomTy < minTy) zoomTy = minTy;
}

if (zoomWrap && zoomInner) {
    // ``passive: false`` is required to call preventDefault on a wheel
    // event and stop the page from scrolling while the user zooms.
    zoomWrap.addEventListener('wheel', (ev) => {
        ev.preventDefault();
        const rect = zoomWrap.getBoundingClientRect();
        // Cursor position relative to the wrap (the scale anchor frame).
        const cx = ev.clientX - rect.left;
        const cy = ev.clientY - rect.top;
        // Same point in unscaled-content coordinates.
        const contentX = (cx - zoomTx) / zoomScale;
        const contentY = (cy - zoomTy) / zoomScale;
        // Standard exponential zoom feel: ~10% per notch.
        const factor = Math.exp(-ev.deltaY * 0.0015);
        let next = zoomScale * factor;
        if (next < ZOOM_MIN) next = ZOOM_MIN;
        if (next > ZOOM_MAX) next = ZOOM_MAX;
        // Re-anchor so the content point under the cursor stays under
        // the cursor.
        zoomTx = cx - contentX * next;
        zoomTy = cy - contentY * next;
        zoomScale = next;
        clampPan();
        applyZoom();
        redraw();
    }, {passive: false});
}

if (zoomResetBtn) zoomResetBtn.addEventListener('click', resetZoom);

// ----- BCWS points + polygons overlay -----

const bcwsRefreshBtn = document.getElementById('nf-bcws-refresh');
const bcwsStatusEl = document.getElementById('nf-bcws-status');

if (bcwsRefreshBtn) {
    bcwsRefreshBtn.addEventListener('click', async () => {
        bcwsRefreshBtn.disabled = true;
        if (bcwsStatusEl) bcwsStatusEl.textContent = 'Downloading BCWS data...';
        try {
            const r = await fetch('/api/bcws/refresh', {method: 'POST'});
            const j = await r.json().catch(() => ({}));
            if (!r.ok) {
                if (bcwsStatusEl) {
                    bcwsStatusEl.textContent =
                        `Failed: ${j.error || r.statusText}`;
                }
            } else {
                if (bcwsStatusEl) {
                    bcwsStatusEl.textContent =
                        `Updated: ${j.n_points} point(s), `
                        + `${j.n_polygons} polygon(s)`;
                }
                await loadBcwsOverlay();
            }
        } catch (exc) {
            if (bcwsStatusEl) bcwsStatusEl.textContent = `Network error: ${exc}`;
        } finally {
            bcwsRefreshBtn.disabled = false;
        }
    });
}

// ----- Boot -----

loadYear(NF_ACTIVE_YEAR);
loadBcwsOverlay();
})();

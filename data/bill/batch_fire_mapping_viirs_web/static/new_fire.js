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
const zoomFireBtn = document.getElementById('nf-zoom-fire');
const previewStatus = document.getElementById('nf-preview-status');
const previewStages = document.getElementById('nf-preview-stages');
const previewWrap = document.getElementById('nf-preview-wrap');
const previewImg = document.getElementById('nf-preview-img');
const previewMeta = document.getElementById('nf-preview-meta');
const bandCaptionEl = document.getElementById('nf-band-caption');
const resolutionCaptionEl = document.getElementById('nf-resolution-caption');

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
};

let meta = null;
// {x0, y0, x1, y1} -- two opposite corners of the AOI box, in RASTER-NATIVE
// CRS units (metres), NOT screen/canvas pixels. Storing it in native CRS
// (the same space VIIRS/BCWS overlay points live in) means it's projected
// to screen pixels fresh on every redraw via nativeToCanvas(), exactly
// like every other overlay -- so it automatically tracks window resizes,
// zoom, and pan instead of staying pinned at a stale absolute pixel
// position. (Previously this stored raw screen-pixel mouse coordinates,
// which is why the box visually drifted away from the underlying image
// whenever the window was reshaped: the image's on-screen position/size
// changed but the box's cached pixel coords did not.)
let bbox = null;
let drag = null;          // {kind: 'create'|'move', startNative, origBbox?}
// Right-button pan gesture: {lastX, lastY} in CLIENT (CSS pixel)
// coordinates. Kept separate from `drag` so a pan can never be
// confused with an AOI gesture in either direction.
let panState = null;
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

    // Try sessionStorage first — avoids a network round-trip on
    // window-switch / page-revisit when the stack file hasn't changed.
    // The cache key is the ETag the server returned last time; if the
    // server's ETag changes (new stack file), the 200 response replaces
    // the cached entry automatically.
    const ssKey = `nf_meta_${year}`;
    let metaJson = null;
    let cachedEtag = null;
    try {
        const cached = sessionStorage.getItem(ssKey);
        if (cached) {
            const parsed = JSON.parse(cached);
            metaJson = parsed.meta;
            cachedEtag = parsed.etag || null;
        }
    } catch (_) {}

    try {
        const headers = {};
        if (cachedEtag) headers['If-None-Match'] = cachedEtag;
        const r = await fetch(`/api/year/${year}/overview_meta`, {headers});
        if (r.status === 304 && metaJson) {
            // Server confirmed nothing changed — use cached meta as-is.
            meta = metaJson;
        } else if (r.ok) {
            meta = await r.json();
            // Store fresh copy + new ETag for next load.
            const newEtag = r.headers.get('ETag') || null;
            try {
                sessionStorage.setItem(ssKey, JSON.stringify(
                    {meta, etag: newEtag}));
            } catch (_) {}
        } else {
            showErrors([{message: `Failed to load year ${year} metadata`}]);
            return;
        }
    } catch (exc) {
        if (metaJson) {
            // Network error but we have a cached copy — use it.
            meta = metaJson;
        } else {
            showErrors([{message: `Network error: ${exc}`}]);
            return;
        }
    }
    // Render the R:/G:/B: band-name caption (bold) showing exactly
    // which bands of the active stack file are being visualized.
    if (bandCaptionEl) {
        bandCaptionEl.innerHTML = '';
        (meta.rgb_band_names || []).forEach((line) => {
            const b = document.createElement('b');
            b.textContent = line;
            bandCaptionEl.appendChild(b);
            bandCaptionEl.appendChild(document.createElement('br'));
        });
    }
    if (resolutionCaptionEl) {
        const res = meta.overview_resolution_m;
        resolutionCaptionEl.textContent = (typeof res === 'number')
            ? `Overview sampled at ~${res.toFixed(0)}m/px `
              + `(native ${meta.native_resolution_m.toFixed(0)}m/px).`
            : '';
    }
    // Cache-bust on the raster's own cache_key (mtime+size), not on
    // wall-clock time -- using Date.now() forced a full re-fetch of a
    // 50-100MB+ overview PNG on every single page load/year-switch,
    // even when the underlying stack file hadn't changed at all. The
    // server already computes this exact key to decide whether ITS
    // OWN on-disk cache is stale (overview_is_fresh() in overview.py);
    // reusing it here means the browser's HTTP cache can actually
    // serve a repeat load from disk/memory instead of re-downloading,
    // and only refetches when the stack file is genuinely different.
    const cacheKey = meta.cache_key
        ? `${meta.cache_key.st_mtime_ns}_${meta.cache_key.st_size}`
        : Date.now();  // no cache_key in the metadata -- fall back
                       // to always-fresh rather than risk showing a
                       // stale image with no way to detect staleness.
    loadOverviewPyramid(year, cacheKey);
    fields.start.placeholder = meta.default_start || 'YYYY-MM-DD';
    fields.end.placeholder = meta.default_end || 'YYYY-MM-DD';
    if (!fields.start.value) fields.start.value = '';
    if (!fields.end.value) fields.end.value = '';
}

// ----- Two-level overview pyramid loading -----
//
// The overview PNG is generated at two resolutions server-side:
//   overview_low.png -- 2000px tall, arrives quickly
//   overview.png     -- full size (longest edge up to 9090)
//
// Both are fetched concurrently. The map stays hidden behind a
// progress bar until the low level arrives; then the interface comes
// up immediately on that image while the full-size one continues
// downloading behind a second progress bar. When the full-size image
// lands it replaces the low one in place.
//
// The swap is seamless because none of the coordinate math depends on
// the image's intrinsic pixel size: canvasToRasterPx() divides by
// meta.overview_W and immediately multiplies by raster_W/overview_W,
// so overview_W cancels out entirely, and nativeToCanvas() only ever
// uses overviewBufferW() -- the image's *rendered* CSS width. Both
// levels share an aspect ratio, so both render into the same CSS box
// and every stored coordinate (bbox, overlays) stays valid across the
// swap without recomputation.

let overviewPyramidGen = 0;
let overviewObjectUrls = [];
// Intrinsic width of whichever level is currently displayed. Used to
// compute the "zoom to 1:1" target for click-to-zoom.
let overviewNaturalW = 0;

function _fmtBytes(b) {
    if (b >= 1048576) return (b / 1048576).toFixed(1) + ' MB';
    if (b >= 1024) return (b / 1024).toFixed(0) + ' kB';
    return b + ' B';
}

function _fmtEta(sec) {
    if (!isFinite(sec) || sec < 0) return 'estimating\u2026';
    if (sec < 1) return 'under a second';
    if (sec < 60) return `~${Math.ceil(sec)}s`;
    const m = Math.floor(sec / 60);
    const s = Math.ceil(sec % 60);
    return `~${m}m ${s}s`;
}

// Ground sample distance, formatted for a caption. Adaptive precision
// so a 200m/px overview reads "200m" rather than "200.0m", while a
// fine one still shows a useful fraction.
function _fmtResM(m) {
    if (typeof m !== 'number' || !isFinite(m) || m <= 0) return null;
    if (m >= 10) return `${Math.round(m)}m`;
    if (m >= 1) return `${m.toFixed(1)}m`;
    return `${m.toFixed(2)}m`;
}

// Fetch an image with byte-level progress, writing running totals into
// `track` so a separate 100ms ticker can render them. Returns an
// object URL for the downloaded bytes.
async function fetchImageTracked(url, track) {
    track.startedAt = performance.now();
    track.received = 0;
    track.total = 0;
    track.done = false;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const lenHdr = resp.headers.get('Content-Length');
    track.total = lenHdr ? (parseInt(lenHdr, 10) || 0) : 0;
    if (!resp.body || typeof resp.body.getReader !== 'function') {
        // Streaming unavailable -- fall back to a plain blob read. No
        // incremental progress, but the bar still completes correctly.
        const blob = await resp.blob();
        track.received = track.total = blob.size;
        track.done = true;
        return URL.createObjectURL(blob);
    }
    const reader = resp.body.getReader();
    const chunks = [];
    for (;;) {
        const {done, value} = await reader.read();
        if (done) break;
        chunks.push(value);
        track.received += value.length;
    }
    track.done = true;
    return URL.createObjectURL(new Blob(chunks, {type: 'image/png'}));
}

function startProgressTicker(track, barEl, etaEl, label) {
    if (!barEl || !etaEl) return null;
    const tick = () => {
        const elapsed = (performance.now() - track.startedAt) / 1000;
        let frac;
        if (track.done) {
            frac = 1;
        } else if (track.total > 0) {
            frac = Math.min(0.999, track.received / track.total);
        } else {
            // Unknown content length: creep toward 100% asymptotically
            // so the bar still shows life rather than sitting at zero.
            frac = 1 - Math.exp(-elapsed / 6);
        }
        barEl.style.width = (frac * 100).toFixed(1) + '%';
        if (track.done) {
            etaEl.textContent = `${label}: complete.`;
            return;
        }
        if (track.total > 0 && track.received > 0 && elapsed > 0.2) {
            const rate = track.received / elapsed;   // bytes/sec
            const eta = (track.total - track.received) / rate;
            etaEl.textContent =
                `${label}: ${_fmtBytes(track.received)} of `
                + `${_fmtBytes(track.total)} `
                + `(${(frac * 100).toFixed(0)}%) \u2014 ETA ${_fmtEta(eta)}`;
        } else {
            etaEl.textContent = `${label}: starting\u2026`;
        }
    };
    tick();
    return setInterval(tick, 100);
}

// Point the <img> at `objUrl`, resolving once it has actually decoded
// (so clientWidth/naturalWidth are meaningful immediately after).
function setOverviewImage(objUrl) {
    return new Promise((resolve, reject) => {
        overview.onload = () => resolve();
        overview.onerror = () => reject(new Error('image decode failed'));
        overview.src = objUrl;
    });
}

async function loadOverviewPyramid(year, cacheKey) {
    const gen = ++overviewPyramidGen;

    const initialBlock = document.getElementById('nf-initial-loading');
    const initialBar   = document.getElementById('nf-initial-bar');
    const initialEta   = document.getElementById('nf-initial-eta');
    const hiBlock      = document.getElementById('nf-highres-loading');
    const hiBar        = document.getElementById('nf-highres-bar');
    const hiEta        = document.getElementById('nf-highres-eta');

    const lowUrl  = `/api/year/${year}/overview_low.png?v=${cacheKey}`;
    const highUrl = `/api/year/${year}/overview.png?v=${cacheKey}`;

    // Release object URLs from a previous year so the blobs can be
    // garbage collected.
    overviewObjectUrls.forEach(u => {
        try { URL.revokeObjectURL(u); } catch (_) {}
    });
    overviewObjectUrls = [];

    // Hide the map; show the initial progress bar in its place.
    if (zoomWrap) zoomWrap.style.display = 'none';
    if (initialBlock) initialBlock.style.display = '';
    if (hiBlock) hiBlock.style.display = 'none';

    const lowTrack = {};
    const highTrack = {};

    // Kick BOTH downloads off at once -- the server is threaded, so the
    // full-size fetch makes progress while the low level is still in
    // flight. Only the low level gets a progress bar at this stage; it
    // is far smaller and always finishes first.
    const lowPromise = fetchImageTracked(lowUrl, lowTrack);
    const highPromise = fetchImageTracked(highUrl, highTrack);
    // Nothing awaits highPromise until later; swallow rejections now so
    // a failure can never surface as an unhandled promise rejection.
    highPromise.catch(() => {});

    const lowTicker = startProgressTicker(
        lowTrack, initialBar, initialEta, 'Loading map preview');

    let lowObj = null;
    try {
        lowObj = await lowPromise;
    } catch (exc) {
        if (initialEta) {
            initialEta.textContent =
                `Preview unavailable (${exc.message}) \u2014 waiting for `
                + `full-resolution image\u2026`;
        }
    }
    if (gen !== overviewPyramidGen) {
        if (lowTicker) clearInterval(lowTicker);
        return;
    }

    if (lowObj) {
        overviewObjectUrls.push(lowObj);
        try {
            await setOverviewImage(lowObj);
        } catch (_) {
            lowObj = null;   // fall through and wait for full size
        }
    }
    if (gen !== overviewPyramidGen) {
        if (lowTicker) clearInterval(lowTicker);
        return;
    }
    if (lowTicker) clearInterval(lowTicker);

    // Bring the map up on the low-resolution level. The wrap must be
    // visible *before* sizeCanvasToWrap(), since clientWidth/Height
    // read 0 while it is display:none.
    if (lowObj) {
        if (initialBlock) initialBlock.style.display = 'none';
        if (zoomWrap) zoomWrap.style.display = '';
        overviewNaturalW = overview.naturalWidth || 0;
        sizeCanvasToWrap();
        redraw();
    }

    const showHiBar = lowObj && hiBlock && !highTrack.done;
    if (showHiBar) hiBlock.style.display = '';
    // Name the level by its actual ground sample distance rather than
    // calling it "full resolution": the full-size overview is still a
    // downsample of the stack (e.g. ~200m/px against 20m/px native),
    // so "full resolution" would overstate what is arriving.
    const hiResTxt = _fmtResM(meta && meta.overview_resolution_m);
    const hiLabel = hiResTxt
        ? `Loading ${hiResTxt}/px overview`
        : 'Loading higher-resolution overview';
    const hiTicker = showHiBar
        ? startProgressTicker(highTrack, hiBar, hiEta, hiLabel)
        : null;

    let highObj = null;
    try {
        highObj = await highPromise;
    } catch (exc) {
        if (hiTicker) clearInterval(hiTicker);
        if (hiBlock) hiBlock.style.display = 'none';
        if (!lowObj) {
            if (initialBlock) initialBlock.style.display = 'none';
            showErrors([{message: `Failed to load overview: ${exc.message}`}]);
        }
        return;
    }
    if (gen !== overviewPyramidGen) {
        if (hiTicker) clearInterval(hiTicker);
        try { URL.revokeObjectURL(highObj); } catch (_) {}
        return;
    }

    // ---- Seamless swap ----
    // Record the rendered width before swapping. Both levels share an
    // aspect ratio and both normally exceed the container width, so
    // max-width:100% renders them into an identical box and the ratio
    // below is exactly 1. The compensation is a safety net for small
    // rasters where the low level is narrower than the container: it
    // rescales the pan offsets so the same ground point stays put.
    const wBefore = overview.clientWidth || 0;
    overviewObjectUrls.push(highObj);
    try {
        await setOverviewImage(highObj);
    } catch (_) {
        if (hiTicker) clearInterval(hiTicker);
        if (hiBlock) hiBlock.style.display = 'none';
        return;
    }
    if (gen !== overviewPyramidGen) {
        if (hiTicker) clearInterval(hiTicker);
        return;
    }
    const wAfter = overview.clientWidth || 0;
    if (wBefore > 0 && wAfter > 0 && Math.abs(wAfter - wBefore) > 0.5) {
        const ratio = wAfter / wBefore;
        zoomTx *= ratio;
        zoomTy *= ratio;
        clampPan();
        applyZoom();
    }
    overviewNaturalW = overview.naturalWidth || 0;

    // If the low level never rendered, the map is still hidden --
    // bring it up now on the full-size image.
    if (!lowObj) {
        if (initialBlock) initialBlock.style.display = 'none';
        if (zoomWrap) zoomWrap.style.display = '';
    }
    if (hiTicker) clearInterval(hiTicker);
    if (hiBlock) hiBlock.style.display = 'none';
    sizeCanvasToWrap();
    redraw();
}

// The canvas sits OUTSIDE .nf-zoom-inner (see new_fire.html) so the
// CSS zoom transform never touches it -- only the <img> gets CSS-
// scaled (acceptable: it's already a downsampled raster, blurring on
// zoom is imperceptible). The canvas is sized once to match its wrap
// box 1:1 in real pixels and stays that size regardless of zoom;
// everything drawn on it is redrawn fresh at the current zoom/pan via
// bufferToScreen() below, so it rasterizes crisp at any zoom level
// instead of being a bitmap that gets stretched and blurred.
//
// Declared here (rather than down with the rest of the zoom
// machinery) so sizeCanvasToWrap() -- called from callbacks that can
// run before the rest of the file finishes its top-to-bottom pass --
// never reads it before initialization.
const zoomWrap = document.getElementById('nf-canvas-wrap');

function sizeCanvasToWrap() {
    if (!zoomWrap) return;
    canvas.width = zoomWrap.clientWidth;
    canvas.height = zoomWrap.clientHeight;
}

window.addEventListener('resize', () => {
    if (!overview.complete) return;
    sizeCanvasToWrap();
    redraw();
});

// ----- Coordinate conversions -----
//
// Two coordinate spaces meet here:
//   "buffer" space  -- pixels in the unscaled overview image, i.e.
//                       what canvasToRasterPx/nativeToRasterPx already
//                       worked in before zoom was added. Independent
//                       of zoomScale/pan.
//   "screen" space  -- actual on-screen canvas pixels, after applying
//                       the same pan+zoom the CSS transform used to
//                       apply to the whole element. This is the space
//                       mouse events and ctx.draw* calls use now.
// bufferToScreen / screenToBuffer convert between them; every other
// function below is unchanged and still operates in buffer space.

function bufferToScreen(bx, by) {
    return [bx * zoomScale + zoomTx, by * zoomScale + zoomTy];
}

function screenToBuffer(sx, sy) {
    return [(sx - zoomTx) / zoomScale, (sy - zoomTy) / zoomScale];
}

function canvasToRasterPx(mx, my) {
    if (!meta) return null;
    // mx, my arrive in screen space (raw canvas pixels, since the
    // canvas is never CSS-scaled) -- undo pan/zoom to get back to
    // buffer space before the existing overview_W/H-based math below.
    const [bx, by] = screenToBuffer(mx, my);
    // Buffer pixels → overview pixels (buffer matches overview dims 1:1)
    const sx = meta.overview_W / overviewBufferW();
    const sy = meta.overview_H / overviewBufferH();
    const ovx = bx * sx;
    const ovy = by * sy;
    // Overview px → raster px
    const rx = ovx * (meta.raster_W / meta.overview_W);
    const ry = ovy * (meta.raster_H / meta.overview_H);
    return [rx, ry];
}

// The buffer's pixel dimensions are simply the overview image's
// natural/intrinsic size (meta.overview_W/H) -- this no longer has
// anything to do with canvas.width/height, which is now the wrap's
// fixed on-screen size, not the image's pixel size.
// The buffer's pixel dimensions are the overview <img>'s actual
// RENDERED size (its CSS box, e.g. clientWidth/Height) -- NOT
// meta.overview_W/H, which is the PNG's native/source pixel
// resolution (can be many times larger, e.g. up to max_dim=9090).
// canvasToRasterPx below already divides by this to get to overview
// pixels, so it must match whatever size the image is ACTUALLY
// displayed at, same as the original (pre-zoom-rewrite) code did via
// canvas.width = overview.clientWidth.
function overviewBufferW() { return overview.clientWidth || 1; }
function overviewBufferH() { return overview.clientHeight || 1; }

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
    // raster px → buffer px (buffer matches overview's native size)
    const bx = rp[0] * (overviewBufferW() / meta.raster_W);
    const by = rp[1] * (overviewBufferH() / meta.raster_H);
    // buffer px → screen px (apply current pan/zoom so drawing lands
    // at its true on-screen position, redrawn crisp every time rather
    // than relying on a CSS transform to visually stretch a bitmap)
    return bufferToScreen(bx, by);
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
let viirsOverlay = null;  // {points: [[x,y],...], det_dts: [...], native_resolution_m} in raster-native CRS
const viirsToggle = document.getElementById('nf-viirs-toggle');
// Forward-declared here (assigned further down, alongside the rest of
// the zoom machinery) so drawBcwsOverlay() -- which runs earlier in
// some call paths -- never references it before initialization.
let zoomScale = 1;

async function loadViirsOverlay() {
    try {
        const r = await fetch('/api/viirs/overlay');
        if (!r.ok) return;
        viirsOverlay = await r.json();
        console.log('[viirs] points:', (viirsOverlay.points || []).length,
                    'native_resolution_m:', viirsOverlay.native_resolution_m);
    } catch (exc) {
        viirsOverlay = null;
    }
    redraw();
}

function drawViirsOverlay(ctx) {
    if (!viirsOverlay || !meta) return;
    if (viirsToggle && !viirsToggle.checked) return;
    const pts = viirsOverlay.points || [];
    if (!pts.length) return;

    // VIIRS detection pixel radius, in native CRS metres, scaled to
    // however the overview/zoom is currently rendering -- computed
    // dynamically from two native-CRS points a known distance apart,
    // rather than any hardcoded pixel ratio, so it stays correct
    // regardless of the overview's sampling resolution or current
    // zoom level.
    const resM = viirsOverlay.native_resolution_m || 375.0;
    const radiusM = resM / 2;

    // Magenta so detections stand out clearly against the green/yellow
    // vegetation and blue water of the false-color overview, instead of
    // blending into the foliage the way the previous green did.
    ctx.fillStyle = 'rgba(255, 0, 220, 0.55)';
    ctx.strokeStyle = 'rgba(200, 0, 170, 0.9)';
    ctx.lineWidth = 1;

    pts.forEach(([x, y]) => {
        const c0 = nativeToCanvas(x, y);
        const c1 = nativeToCanvas(x + radiusM, y);
        if (!c0 || !c1) return;
        const screenRadius = Math.hypot(c1[0] - c0[0], c1[1] - c0[1]);
        if (screenRadius <= 0) return;
        ctx.beginPath();
        ctx.arc(c0[0], c0[1], screenRadius, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
    });
}

async function loadBcwsOverlay() {
    try {
        const r = await fetch('/api/bcws/overlay');
        if (!r.ok) return;
        bcwsOverlay = await r.json();
        console.log('[bcws] points:', (bcwsOverlay.points || []).length,
                    'polygons:', (bcwsOverlay.polygons || []).length);
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

    // Under the old design the canvas itself was CSS-scaled by the
    // zoom transform, so a constant ctx.lineWidth/marker size would
    // visually grow with zoom -- everything here had to be divided by
    // zoomScale to compensate. That's no longer true: the canvas now
    // lives outside the zoom transform and is redrawn fresh every
    // frame via nativeToCanvas() -> bufferToScreen(), which already
    // bakes the current pan/zoom into each point's screen position.
    // A plain constant width/size here is therefore CORRECT and
    // crisp at any zoom level by construction -- no compensation
    // needed, and dividing by zoomScale would be wrong (it would
    // make lines thinner, not constant, as you zoom in).
    const lineWidthPx = 1;
    const markerHalfPx = 5;

    ctx.strokeStyle = 'rgba(220, 0, 0, 0.9)';
    ctx.fillStyle = 'rgba(220, 0, 0, 0.18)';
    ctx.lineWidth = lineWidthPx;
    polys.forEach((ring) => {
        if (!ring || ring.length < 3) return;
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

    // Points as X markers (two crossed line segments) rather than
    // filled circles -- circles at small/clamped radius were
    // visually blending together and made it impossible to tell
    // whether polygons were rendering underneath them. An X's
    // strokes are the same constant width as the polygon outlines,
    // and an X's open center doesn't occlude a polygon edge sitting
    // right under a point the way a filled disc does.
    ctx.strokeStyle = 'rgba(220, 0, 0, 0.95)';
    ctx.lineWidth = lineWidthPx;
    pts.forEach(([x, y]) => {
        const cp = nativeToCanvas(x, y);
        if (!cp) return;
        const [cx, cy] = cp;
        ctx.beginPath();
        ctx.moveTo(cx - markerHalfPx, cy - markerHalfPx);
        ctx.lineTo(cx + markerHalfPx, cy + markerHalfPx);
        ctx.moveTo(cx + markerHalfPx, cy - markerHalfPx);
        ctx.lineTo(cx - markerHalfPx, cy + markerHalfPx);
        ctx.stroke();
    });

    // Draw fire number labels next to polygons and points.
    const polyNums = bcwsOverlay.polygon_fire_nums || [];
    const ptNums = bcwsOverlay.point_fire_nums || [];
    ctx.font = '10px sans-serif';
    ctx.textBaseline = 'bottom';
    polys.forEach((ring, i) => {
        if (!ring || ring.length < 1 || !polyNums[i]) return;
        // Label at centroid of the polygon's first ring.
        let sx = 0, sy = 0;
        ring.forEach(([x, y]) => { sx += x; sy += y; });
        const cp = nativeToCanvas(sx / ring.length, sy / ring.length);
        if (!cp) return;
        const label = polyNums[i];
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        const tw = ctx.measureText(label).width;
        ctx.fillRect(cp[0] - 1, cp[1] - 11, tw + 2, 12);
        ctx.fillStyle = '#ff4444';
        ctx.fillText(label, cp[0], cp[1]);
    });
    pts.forEach(([x, y], i) => {
        if (!ptNums[i]) return;
        const cp = nativeToCanvas(x, y);
        if (!cp) return;
        const label = ptNums[i];
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        const tw = ctx.measureText(label).width;
        ctx.fillRect(cp[0] + markerHalfPx + 2, cp[1] - 11, tw + 2, 12);
        ctx.fillStyle = '#ff4444';
        ctx.fillText(label, cp[0] + markerHalfPx + 3, cp[1]);
    });
}

// Projects a native-CRS bbox {x0,y0,x1,y1} to a screen-pixel rect
// {x0,y0,x1,y1} via nativeToCanvas, the same projection every other
// overlay point goes through -- so the box is always positioned
// relative to wherever the underlying image is currently rendered,
// regardless of window size, zoom, or pan.
function nativeBboxScreenRect(b) {
    if (!b || !meta) return null;
    const c0 = nativeToCanvas(b.x0, b.y0);
    const c1 = nativeToCanvas(b.x1, b.y1);
    if (!c0 || !c1) return null;
    return {x0: c0[0], y0: c0[1], x1: c1[0], y1: c1[1]};
}

// True once the overview <img> has actually decoded and has real pixel
// dimensions. Before that, overviewBufferW()/H() fall back to a bogus
// 1px width, which would briefly place every overlay point/marker at a
// garbage position near the canvas origin. Drawing is skipped in that
// window; the overview's own onload handler triggers a fresh redraw()
// once it's actually ready, so nothing is lost -- this just avoids ever
// painting the wrong thing in between.
function overviewReady() {
    return !!(overview.naturalWidth && overview.naturalHeight);
}

function redraw() {
    // Always resync first -- the wrap's size can be 0x0 until the
    // overview <img> has actually loaded (the wrap is unsized CSS
    // that shrinks to its content), so whichever of loadYear()'s
    // image load / loadBcwsOverlay()'s fetch happens to resolve
    // first cannot be trusted to have already sized the canvas
    // correctly. Cheap to call every redraw; only a real resize if
    // the wrap's dimensions actually changed since last time.
    sizeCanvasToWrap();
    const ctx = canvas.getContext('2d');
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!overviewReady()) return;  // see overviewReady() -- avoid drawing
                                    // against not-yet-decoded image dims
    drawViirsOverlay(ctx);
    drawBcwsOverlay(ctx);
    if (!bbox) return;
    const r = nativeBboxScreenRect(bbox);
    if (!r) return;
    const x = Math.min(r.x0, r.x1);
    const y = Math.min(r.y0, r.y1);
    const w = Math.abs(r.x1 - r.x0);
    const h = Math.abs(r.y1 - r.y0);
    ctx.fillStyle = 'rgba(255, 220, 70, 0.18)';
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = 'rgba(255, 180, 0, 0.95)';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
}

function updateReadout() {
    if (!bbox || !meta) return;
    const xmin = Math.min(bbox.x0, bbox.x1), xmax = Math.max(bbox.x0, bbox.x1);
    const ymin = Math.min(bbox.y0, bbox.y1), ymax = Math.max(bbox.y0, bbox.y1);
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
    bbox = {x0: xmin, y0: ymax, x1: xmax, y1: ymin};
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
    // The canvas is no longer inside the CSS zoom transform (see
    // new_fire.html / sizeCanvasToWrap()), so canvas.width/height
    // already equals rect.width/height in the normal case -- sx/sy
    // below are always ~1.0 in practice. Left in as a defensive
    // fallback in case some other CSS (browser zoom, devtools
    // scaling, etc.) ever puts the canvas's rendered size out of
    // sync with its buffer size; costs nothing when they match.
    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) {
        return [ev.clientX - rect.left, ev.clientY - rect.top];
    }
    const sx = canvas.width / rect.width;
    const sy = canvas.height / rect.height;
    return [(ev.clientX - rect.left) * sx, (ev.clientY - rect.top) * sy];
}

function bboxContains(b, mx, my) {
    const r = nativeBboxScreenRect(b);
    if (!r) return false;
    const x0 = Math.min(r.x0, r.x1), x1 = Math.max(r.x0, r.x1);
    const y0 = Math.min(r.y0, r.y1), y1 = Math.max(r.y0, r.y1);
    return mx >= x0 && mx <= x1 && my >= y0 && my <= y1;
}

canvas.addEventListener('mousedown', (ev) => {
    if (!meta) return;
    // Right button starts a pan. Handled entirely separately from the
    // AOI gestures below, and tracked on `window` (further down) so a
    // pan that wanders off the canvas keeps working until the button
    // comes back up.
    if (ev.button === 2) {
        ev.preventDefault();
        panState = {lastX: ev.clientX, lastY: ev.clientY};
        canvas.style.cursor = 'grabbing';
        return;
    }
    // Everything below is a left-button gesture. Ignore middle/other
    // buttons rather than letting them start an AOI drag -- previously
    // there was no button check at all, so a right-click would create
    // a degenerate bbox (destroying any existing one) and then trip
    // the click-to-zoom toggle on release.
    if (ev.button !== 0) return;
    const [mx, my] = getMousePos(ev);
    // Remember where the press started, and what the bbox looked like
    // beforehand. If the pointer never really moves, mouseup treats the
    // gesture as a plain click (zoom toggle) and restores this snapshot
    // -- otherwise a stray click would silently destroy the user's AOI,
    // since the 'create' branch below overwrites `bbox` immediately.
    const downInfo = {
        downScreen: [mx, my],
        bboxBefore: bbox ? Object.assign({}, bbox) : null,
        moved: false,
    };
    if (bbox && bboxContains(bbox, mx, my)) {
        const startNative = canvasToNative(mx, my);
        if (!startNative) return;
        drag = Object.assign({kind: 'move', startNative,
                              origBbox: Object.assign({}, bbox)},
                             downInfo);
    } else {
        const nat = canvasToNative(mx, my);
        if (!nat) return;
        bbox = {x0: nat[0], y0: nat[1], x1: nat[0], y1: nat[1]};
        drag = Object.assign({kind: 'create'}, downInfo);
        redraw();
    }
});

// ----- Click-to-zoom (no-drag left click) -----
//
// A press-and-release with no meaningful pointer movement toggles
// between "fit" and "1:1 with the overview image's own pixels",
// centred on wherever was clicked. Anything with actual drag in it is
// an AOI gesture and never reaches here.
const CLICK_SLOP_PX = 3;

function overviewFullResScale() {
    // Scale at which one overview-image pixel maps to one screen pixel.
    const shown = overview.clientWidth || 0;
    if (!shown || !overviewNaturalW) return null;
    const s = overviewNaturalW / shown;
    if (!isFinite(s) || s <= 1) return null;   // already at/above 1:1
    return Math.min(s, ZOOM_MAX);
}

function toggleClickZoom(mx, my) {
    const full = overviewFullResScale();
    if (full === null) {
        // Nothing meaningful to zoom into (image already displayed at
        // or above its native resolution) -- just reset if zoomed.
        if (zoomScale > 1) resetZoom();
        return;
    }
    // Treat "close enough to full" as fully zoomed in, so the second
    // click reliably zooms back out even after wheel adjustments.
    if (zoomScale >= full * 0.999) {
        resetZoom();
        return;
    }
    if (!zoomWrap) return;
    // Content point under the click, then re-anchor it to the centre of
    // the viewport at the new scale.
    const contentX = (mx - zoomTx) / zoomScale;
    const contentY = (my - zoomTy) / zoomScale;
    zoomScale = full;
    zoomTx = zoomWrap.clientWidth / 2 - contentX * zoomScale;
    zoomTy = zoomWrap.clientHeight / 2 - contentY * zoomScale;
    clampPan();
    applyZoom();
    redraw();
}

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
    // Once the pointer travels past the slop radius this is a real
    // drag, not a click -- latch that so mouseup routes it to the AOI
    // logic instead of the zoom toggle.
    if (!drag.moved && drag.downScreen) {
        const ddx = mx - drag.downScreen[0];
        const ddy = my - drag.downScreen[1];
        if ((ddx * ddx + ddy * ddy) > (CLICK_SLOP_PX * CLICK_SLOP_PX)) {
            drag.moved = true;
        }
    }
    if (drag.kind === 'create') {
        const nat = canvasToNative(mx, my);
        if (nat) { bbox.x1 = nat[0]; bbox.y1 = nat[1]; }
    } else if (drag.kind === 'move') {
        const cur = canvasToNative(mx, my);
        if (cur) {
            const dx = cur[0] - drag.startNative[0];
            const dy = cur[1] - drag.startNative[1];
            bbox.x0 = drag.origBbox.x0 + dx;
            bbox.y0 = drag.origBbox.y0 + dy;
            bbox.x1 = drag.origBbox.x1 + dx;
            bbox.y1 = drag.origBbox.y1 + dy;
        }
    }
    redraw();
    updateReadout();
});

window.addEventListener('mouseup', (ev) => {
    // Only the left button finishes an AOI gesture. Without this, a
    // right-press/release while a left-drag was still held would
    // prematurely commit the AOI.
    if (ev && ev.button !== 0) return;
    if (drag) {
        const wasDragging = drag;
        drag = null;

        // No meaningful movement => this was a click, not a drag.
        // Put the AOI back exactly as it was and use the gesture to
        // toggle zoom instead. Nothing about the bbox changed, so the
        // cached preview stays valid and is deliberately not
        // invalidated here.
        if (!wasDragging.moved && wasDragging.downScreen) {
            bbox = wasDragging.bboxBefore
                ? Object.assign({}, wasDragging.bboxBefore) : null;
            if (bbox) updateReadout(); else clearReadout();
            redraw();
            toggleClickZoom(wasDragging.downScreen[0],
                            wasDragging.downScreen[1]);
            return;
        }

        const r = bbox ? nativeBboxScreenRect(bbox) : null;
        const tooSmall = !r
            || (Math.abs(r.x0 - r.x1) < 2 && Math.abs(r.y0 - r.y1) < 2);
        if (!bbox || tooSmall) {
            bbox = null;
            clearReadout();
            redraw();
        } else {
            updateReadout();
        }
        // Any actual drag invalidates the preview cache.
        invalidatePreview();
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
    const sizeErr = aoiTooLarge();
    if (sizeErr) {
        showErrors([{field: 'bbox_native', message: sizeErr}]);
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
            `${j.hint_source === 'redwins_post' ? 'Red wins (post)' : 'VIIRS'} ` +
            `hint area (within bbox): ${area}`;
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

function aoiTooLarge() {
    // Returns an error message string if the drawn AOI exceeds the
    // admin-configured max_aoi_fraction of the full-res raster area,
    // or null if it's within limits.
    if (!bbox || !meta) return null;
    const gt = meta.geotransform;
    if (!gt) return null;
    const pixW = Math.abs(gt[1]);  // native CRS metres per pixel
    const pixH = Math.abs(gt[5]);
    const rW = meta.raster_W || 1;
    const rH = meta.raster_H || 1;
    const maxFrac = meta.max_aoi_fraction || 0.10;
    const maxPixels = Math.floor(rW * rH * maxFrac);

    const xmin = Math.min(bbox.x0, bbox.x1);
    const xmax = Math.max(bbox.x0, bbox.x1);
    const ymin = Math.min(bbox.y0, bbox.y1);
    const ymax = Math.max(bbox.y0, bbox.y1);
    const aoiW = Math.ceil(Math.abs(xmax - xmin) / pixW);
    const aoiH = Math.ceil(Math.abs(ymax - ymin) / pixH);
    const aoiPixels = aoiW * aoiH;
    if (aoiPixels > maxPixels) {
        const pct = (maxFrac * 100).toFixed(0);
        return `AOI is too large: ${aoiW}×${aoiH} = ${aoiPixels.toLocaleString()} pixels `
             + `(limit: ${pct}% of ${rW}×${rH} = ${maxPixels.toLocaleString()} pixels). `
             + `Draw a smaller rectangle.`;
    }
    return null;
}

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
    const sizeErr = aoiTooLarge();
    if (sizeErr) {
        showErrors([{field: 'bbox_native', message: sizeErr}]);
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
// We scale ``.nf-zoom-inner`` (which now contains only the overview
// <img>) via CSS transform for pan/zoom. The canvas lives OUTSIDE
// that transform (see new_fire.html) and is redrawn fresh at the
// current zoom/pan via bufferToScreen()/screenToBuffer() instead of
// being visually stretched -- this is what keeps thin lines and
// fixed-size markers crisp at any zoom level rather than blurring.

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

// ----- Right-button drag to pan -----
//
// zoomTx/zoomTy are consumed by a CSS ``translate(...px)``, so they
// live in the same CSS-pixel space as clientX/clientY. Adding the raw
// pointer delta therefore moves the content exactly 1:1 with the
// mouse, in the same direction, with no scale factor to apply -- at
// any zoom level.
//
// Tracked on `window` rather than the canvas so a pan that runs off
// the edge of the map keeps following the pointer until the button is
// released, instead of stalling at the boundary of the element.

if (zoomWrap) {
    // Without this the browser menu opens on right-press and the
    // matching mouseup never arrives, leaving the pan stuck on.
    zoomWrap.addEventListener('contextmenu', (ev) => ev.preventDefault());
}

window.addEventListener('mousemove', (ev) => {
    if (!panState) return;
    const dx = ev.clientX - panState.lastX;
    const dy = ev.clientY - panState.lastY;
    panState.lastX = ev.clientX;
    panState.lastY = ev.clientY;
    zoomTx += dx;
    zoomTy += dy;
    // clampPan() is what implements "provided we haven't reached the
    // limit": once an edge of the scaled content meets the viewport
    // edge, the offset stops advancing on that axis, so the pan simply
    // stalls at the boundary rather than dragging the map off-screen.
    // At scale 1 both bounds collapse to 0, so panning is a no-op when
    // not zoomed in -- which is the intended behaviour.
    clampPan();
    applyZoom();
    redraw();
});

function endPan() {
    if (!panState) return;
    panState = null;
    canvas.style.cursor = '';
}

window.addEventListener('mouseup', (ev) => {
    if (ev.button === 2) endPan();
});

// If the window loses focus mid-drag (alt-tab, another app steals the
// pointer) the mouseup can be delivered elsewhere and never reach us,
// which would otherwise leave the map panning on the next mousemove.
window.addEventListener('blur', endPan);

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

// ----- Zoom to Fire# -----
//
// When the Name field contains a BCWS fire number (1 letter + 5 digits,
// e.g. G80280), clicking "Zoom to Fire#" finds it in the loaded BCWS
// overlay data and zooms the overview to center on it at max zoom.
// Searches points first; falls back to polygons if not found in points.

const _FIRE_NUM_RE = /^[A-Za-z]\d{5}$/;

if (zoomFireBtn) {
    zoomFireBtn.addEventListener('click', () => {
        const rawName = (fields.name.value || '').trim().toUpperCase();
        if (!_FIRE_NUM_RE.test(rawName)) return;  // not a fire number pattern
        if (!bcwsOverlay || !meta) return;

        const ptNums = bcwsOverlay.point_fire_nums || [];
        const polyNums = bcwsOverlay.polygon_fire_nums || [];
        const pts = bcwsOverlay.points || [];
        const polys = bcwsOverlay.polygons || [];

        // Search points first
        let targetNativeX = null, targetNativeY = null;
        const ptIdx = ptNums.indexOf(rawName);
        if (ptIdx >= 0 && pts[ptIdx]) {
            targetNativeX = pts[ptIdx][0];
            targetNativeY = pts[ptIdx][1];
        } else {
            // Search polygons — center of bounding box
            const polyIdx = polyNums.indexOf(rawName);
            if (polyIdx >= 0 && polys[polyIdx] && polys[polyIdx].length >= 2) {
                const ring = polys[polyIdx];
                let minX = Infinity, minY = Infinity;
                let maxX = -Infinity, maxY = -Infinity;
                for (const [px, py] of ring) {
                    if (px < minX) minX = px;
                    if (py < minY) minY = py;
                    if (px > maxX) maxX = px;
                    if (py > maxY) maxY = py;
                }
                targetNativeX = (minX + maxX) / 2;
                targetNativeY = (minY + maxY) / 2;
            }
        }

        if (targetNativeX === null) {
            alert(rawName + ' not found in BCWS points or polys data');
            return;
        }

        // Convert native CRS coords to buffer-space pixel position
        const rp = nativeToRasterPx(targetNativeX, targetNativeY);
        if (!rp) return;
        const bufX = rp[0] * (overviewBufferW() / meta.raster_W);
        const bufY = rp[1] * (overviewBufferH() / meta.raster_H);

        // Zoom to max and center on the target point
        zoomScale = ZOOM_MAX;
        // Pan so the target buffer point lands at the center of the wrap
        const wrapCx = (zoomWrap ? zoomWrap.clientWidth : canvas.width) / 2;
        const wrapCy = (zoomWrap ? zoomWrap.clientHeight : canvas.height) / 2;
        zoomTx = wrapCx - bufX * zoomScale;
        zoomTy = wrapCy - bufY * zoomScale;
        clampPan();
        applyZoom();
        redraw();
    });
}

// ----- Boot -----

// Size the canvas up front, synchronously, before either async load
// below has a chance to resolve and call redraw() against a
// still-zero-sized canvas. Previously this only happened inside
// overview.onload, which was fine when the image always took a
// moment to load -- but now that the overview PNG is browser-cached
// (see the cache_key change above), it can load near-instantly, and
// loadBcwsOverlay()'s fetch could resolve and redraw() first, onto a
// canvas that was still 0x0 -- which is exactly why markers stopped
// appearing once caching made the image load "too fast".
//
// With pyramid loading the wrap starts hidden, so this first call
// sizes to 0x0 and redraw() short-circuits on overviewReady() until an
// image has actually decoded. loadOverviewPyramid() re-runs
// sizeCanvasToWrap() immediately after it un-hides the wrap, which is
// the call that establishes the real canvas size; this one just keeps
// the pre-reveal state well-defined.
sizeCanvasToWrap();

// loadYear() must complete (and set `meta`) before loadBcwsOverlay()
// draws anything -- drawBcwsOverlay() bails out silently if `meta`
// is still null, and nothing else was forcing a later redraw once it
// became available. Running these in sequence instead of
// concurrently removes that race entirely rather than relying on
// timing happening to work out.
(async () => {
    await loadYear(NF_ACTIVE_YEAR);
    await loadViirsOverlay();
    await loadBcwsOverlay();
})();

if (viirsToggle) {
    viirsToggle.addEventListener('change', () => redraw());
}
})();

document.addEventListener('DOMContentLoaded', () => {
    const $ = (id) => document.getElementById(id);

    // 1x1 transparent GIF: assigning it to an <img> aborts any MJPEG stream it was holding open.
    const BLANK_IMG = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
    const DETECT_INTERVAL_MS = 500;
    const RESULT_HOLD_MS = 6000;
    const ENROLL_FRAMES = 5;
    const ENROLL_FRAME_GAP_MS = 250;

    // Why a face in a scan wasn't matched to anyone (reason codes from /api/attendance).
    const UNKNOWN_REASONS = {
        no_match: 'Not registered',
        ambiguous: 'Uncertain match, not marked',
        duplicate: 'Uncertain match, not marked',
        too_small: 'Too far from camera',
        too_dark: 'Too dark to recognise',
        blurry: 'Too blurry to recognise',
    };

    // ------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------
    let currentStream = null;
    let activeCameraSource = 'drone'; // 'drone' | 'webcam'
    let droneMode = 'live';            // 'live' | 'video' (recorded footage simulating the drone)
    let activeView = 'attendance';     // 'attendance' | 'register' | 'records'
    let students = [];
    let attendanceLogs = [];
    let scanInFlight = false;
    let autoScan = false;              // attendance view scans + marks continuously instead of on button press
    let overlayHoldUntil = 0;         // live detection won't redraw over scan results until then
    let lastTodayKey = todayKey();

    const views = {
        attendance: {
            img: $('att-drone-img'), video: $('att-video'), canvas: $('att-canvas'),
            overlay: $('att-overlay-canvas'), guide: $('att-face-overlay'),
            viewport: $('att-viewport'), label: $('att-stream-label'), chip: $('att-detect-chip'),
            lastDraw: null,
        },
        register: {
            img: $('reg-drone-img'), video: $('reg-video'), canvas: $('reg-canvas'),
            overlay: $('reg-overlay-canvas'), guide: $('reg-face-overlay'),
            viewport: $('reg-viewport'), label: $('reg-stream-label'), chip: $('reg-detect-chip'),
            lastDraw: null,
        },
    };

    const PAGE_META = {
        '#attendance-pane': { view: 'attendance', hash: '', title: 'Mark Attendance', subtitle: 'Scan the live feed to recognise and check in every face in view.' },
        '#register-pane': { view: 'register', hash: '#register', title: 'Register Student', subtitle: 'Enrol a new student by capturing their face from the live feed.' },
        '#records-pane': { view: 'records', hash: '#records', title: 'Records', subtitle: 'Registered students, today\'s turnout and the full attendance log.' },
    };

    const STATUS = {
        marked: { tone: 'success', label: 'Marked present', icon: 'bi-check-circle-fill', stroke: '#34d399', pill: '#047857', text: '#ffffff' },
        already_marked: { tone: 'warning', label: 'Already marked', icon: 'bi-clock-history', stroke: '#fbbf24', pill: '#fbbf24', text: '#1f2937' },
        unknown: { tone: 'neutral', label: 'Unknown', icon: 'bi-question-circle-fill', stroke: '#cbd5e1', pill: '#334155', text: '#ffffff' },
    };
    const LIVE_STROKE = '#22d3ee';
    const LIVE_PILL = '#0e7490';

    // ------------------------------------------------------------------
    // Elements
    // ------------------------------------------------------------------
    const pageTitle = $('page-title');
    const pageSubtitle = $('page-subtitle');
    const topbarDate = $('topbar-date');
    const themeToggle = $('theme-toggle');
    const toastStack = $('toast-stack');

    const sbPresent = $('sb-present');
    const sbTotal = $('sb-total');
    const sbMeter = $('sb-meter');
    const sbSourceDot = $('sb-source-dot');
    const sbSourceLabel = $('sb-source-label');
    const navStudentCount = $('nav-student-count');

    const btnMarkAttendance = $('btn-mark-attendance');
    const autoScanToggle = $('auto-scan-toggle');
    const attIdleState = $('att-idle-state');
    const attResultState = $('att-result-state');
    const attStatusBadge = $('att-status-badge');
    const attFacesList = $('att-faces-list');
    const attLastScan = $('att-last-scan');
    const attCountMarked = $('att-count-marked');
    const attCountAlready = $('att-count-already');
    const attCountUnknown = $('att-count-unknown');
    const todayFeed = $('today-feed');
    const todayCount = $('today-count');

    const regForm = $('reg-form');
    const regName = $('reg-name');
    const regId = $('reg-id');
    const regAlert = $('reg-alert');
    const btnRegister = $('btn-register-student');
    const regIdCard = $('reg-id-card');
    const regIdPhoto = $('reg-id-photo');
    const regIdName = $('reg-id-name');
    const regIdReg = $('reg-id-reg');
    const regIdStatus = $('reg-id-status');

    const studentsTableBody = $('students-table-body');
    const attendanceTableBody = $('attendance-table-body');
    const btnRefreshStudents = $('btn-refresh-students');
    const btnRefreshLogs = $('btn-refresh-logs');
    const btnExportLogs = $('btn-export-logs');
    const studentSearch = $('student-search');
    const logSearch = $('log-search');
    const logDateFilter = $('log-date-filter');
    const studentsCaption = $('students-caption');
    const logsCaption = $('logs-caption');

    const cameraSourceRadios = document.querySelectorAll('input[name="cameraSource"]');
    const droneConfigForm = $('drone-config-form');
    const droneUrlInput = $('drone-url-input');
    const droneConfigAlert = $('drone-config-alert');
    const droneUploadForm = $('drone-upload-form');
    const droneVideoInput = $('drone-video-input');
    const droneUploadAlert = $('drone-upload-alert');

    // ------------------------------------------------------------------
    // Small helpers
    // ------------------------------------------------------------------
    function esc(value) {
        return String(value ?? '').replace(/[&<>"']/g, (ch) => (
            { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]
        ));
    }

    function initials(name) {
        const parts = String(name || '').trim().split(/\s+/).filter(Boolean);
        if (!parts.length) return '?';
        const first = parts[0][0] || '';
        const last = parts.length > 1 ? parts[parts.length - 1][0] : '';
        return (first + last).toUpperCase();
    }

    function hueFor(text) {
        let h = 0;
        for (const ch of String(text || '')) h = (h * 31 + ch.charCodeAt(0)) % 360;
        return h;
    }

    function avatarHtml(name) {
        return `<span class="avatar" style="--h:${hueFor(name)}" aria-hidden="true">${esc(initials(name))}</span>`;
    }

    function pad2(n) {
        return String(n).padStart(2, '0');
    }

    function todayKey() {
        const d = new Date();
        return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
    }

    // Attendance timestamps are server-local "YYYY-MM-DD HH:MM:SS".
    function parseTimestamp(ts) {
        if (!ts) return null;
        const d = new Date(String(ts).replace(' ', 'T'));
        return Number.isNaN(d.getTime()) ? null : d;
    }

    function formatTime(ts, withSeconds = false) {
        const d = parseTimestamp(ts);
        if (!d) return '—';
        return d.toLocaleTimeString([], withSeconds
            ? { hour: '2-digit', minute: '2-digit', second: '2-digit' }
            : { hour: '2-digit', minute: '2-digit' });
    }

    function formatDateKey(key) {
        const d = parseTimestamp(`${key} 00:00:00`);
        if (!d) return key;
        return d.toLocaleDateString([], { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' });
    }

    // Wraps case-insensitive matches of `query` (already lower-cased) in <mark>, escaping the rest.
    function highlight(text, query) {
        const raw = String(text ?? '');
        if (!query) return esc(raw);
        const lower = raw.toLowerCase();
        let out = '';
        let from = 0;
        let idx;
        while ((idx = lower.indexOf(query, from)) !== -1) {
            out += `${esc(raw.slice(from, idx))}<mark>${esc(raw.slice(idx, idx + query.length))}</mark>`;
            from = idx + query.length;
        }
        return out + esc(raw.slice(from));
    }

    function plural(n, one, many) {
        return `${n} ${n === 1 ? one : (many || `${one}s`)}`;
    }

    function setAlert(el, tone, icon, message) {
        el.className = `app-alert tone-${tone}`;
        el.innerHTML = `<i class="bi ${icon}"></i><span>${esc(message)}</span>`;
    }

    function hideAlert(el) {
        el.className = 'app-alert d-none';
        el.innerHTML = '';
    }

    function setBusy(btn, busy, label) {
        if (busy) {
            btn.dataset.idleHtml = btn.innerHTML;
            btn.disabled = true;
            btn.innerHTML = `<span class="spinner-border spinner-border-sm" aria-hidden="true"></span><span>${esc(label)}</span>`;
        } else {
            btn.disabled = false;
            if (btn.dataset.idleHtml) btn.innerHTML = btn.dataset.idleHtml;
        }
    }

    function toast(title, message, tone = 'info') {
        const icons = {
            success: 'bi-check-circle-fill', warning: 'bi-exclamation-triangle-fill',
            danger: 'bi-x-octagon-fill', info: 'bi-info-circle-fill',
        };
        const el = document.createElement('div');
        el.className = `toast app-toast tone-${tone}`;
        el.setAttribute('role', tone === 'danger' ? 'alert' : 'status');
        el.setAttribute('aria-live', tone === 'danger' ? 'assertive' : 'polite');
        el.setAttribute('aria-atomic', 'true');
        el.innerHTML = `
            <div class="toast-body">
                <span class="toast-icon"><i class="bi ${icons[tone] || icons.info}"></i></span>
                <div class="flex-grow-1">
                    <div class="toast-title">${esc(title)}</div>
                    ${message ? `<div class="toast-text">${esc(message)}</div>` : ''}
                </div>
                <button type="button" class="btn-close" data-bs-dismiss="toast" aria-label="Close"></button>
            </div>`;
        toastStack.appendChild(el);
        el.addEventListener('hidden.bs.toast', () => el.remove());
        new bootstrap.Toast(el, { delay: 4500 }).show();
    }

    function flash(viewport) {
        viewport.classList.remove('flash');
        void viewport.offsetWidth; // restart the animation
        viewport.classList.add('flash');
        setTimeout(() => viewport.classList.remove('flash'), 500);
    }

    // ------------------------------------------------------------------
    // Theme + clock
    // ------------------------------------------------------------------
    function applyTheme(theme) {
        document.documentElement.setAttribute('data-bs-theme', theme);
        const dark = theme === 'dark';
        themeToggle.innerHTML = dark ? '<i class="bi bi-sun"></i>' : '<i class="bi bi-moon-stars"></i>';
        themeToggle.setAttribute('aria-label', dark ? 'Switch to light theme' : 'Switch to dark theme');
    }

    applyTheme(document.documentElement.getAttribute('data-bs-theme') === 'dark' ? 'dark' : 'light');

    themeToggle.addEventListener('click', () => {
        const next = document.documentElement.getAttribute('data-bs-theme') === 'dark' ? 'light' : 'dark';
        applyTheme(next);
        try { localStorage.setItem('fas-theme', next); } catch (e) { /* storage unavailable */ }
    });

    function tickClock() {
        const now = new Date();
        const time = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        document.querySelectorAll('.hud-clock').forEach((el) => { el.textContent = time; });
        topbarDate.textContent = now.toLocaleDateString([], { weekday: 'long', day: 'numeric', month: 'short' });

        // Roll "today" stats over at midnight.
        const key = todayKey();
        if (key !== lastTodayKey) {
            lastTodayKey = key;
            refreshData();
        }
    }
    tickClock();
    setInterval(tickClock, 1000);

    // ------------------------------------------------------------------
    // Stream settings modal
    // ------------------------------------------------------------------
    async function loadDroneConfig() {
        try {
            const res = await fetch('/api/drone/config');
            const data = await res.json();
            if (data.success && data.stream_url) droneUrlInput.value = data.stream_url;
            if (data.success) {
                droneMode = data.mode || 'live';
                renderSourceLabels();
            }
        } catch (err) {
            console.error('Error loading drone config:', err);
        }
    }
    loadDroneConfig();

    // Points the UI at a newly configured drone source and reloads the visible feed.
    function applyDroneSource(data) {
        droneMode = data.mode || 'live';
        if (data.stream_url) droneUrlInput.value = data.stream_url;
        renderSourceLabels();
        overlayHoldUntil = 0;
        Object.values(views).forEach((view) => {
            clearOverlay(view);
            clearViewportMessage(view);
            setDroneStream(view, activeCameraSource === 'drone' && activeView === viewKey(view), true);
        });
    }

    document.querySelectorAll('.preset-chip').forEach((chip) => {
        chip.addEventListener('click', () => {
            droneUrlInput.value = chip.dataset.preset;
            droneUrlInput.focus();
        });
    });

    $('droneConfigModal').addEventListener('show.bs.modal', () => {
        hideAlert(droneConfigAlert);
        hideAlert(droneUploadAlert);
        loadDroneConfig();
    });

    droneConfigForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const submitBtn = droneConfigForm.querySelector('button[type="submit"]');
        setBusy(submitBtn, true, 'Reconnecting…');
        try {
            const res = await fetch('/api/drone/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ stream_url: droneUrlInput.value.trim() }),
            });
            const data = await res.json();
            if (data.success) {
                setAlert(droneConfigAlert, 'success', 'bi-check-circle-fill', data.message);
                applyDroneSource(data);
                toast('Stream reconnected', data.stream_url, 'success');
            } else {
                setAlert(droneConfigAlert, 'danger', 'bi-exclamation-octagon-fill', data.message || 'Invalid stream URL.');
            }
        } catch (err) {
            setAlert(droneConfigAlert, 'danger', 'bi-x-octagon-fill', 'Failed to update the drone stream configuration.');
        } finally {
            setBusy(submitBtn, false);
        }
    });

    droneUploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const file = droneVideoInput.files[0];
        if (!file) return;
        const submitBtn = droneUploadForm.querySelector('button[type="submit"]');
        hideAlert(droneUploadAlert);
        setBusy(submitBtn, true, 'Uploading…');
        try {
            const body = new FormData();
            body.append('video', file);
            const res = await fetch('/api/drone/upload', { method: 'POST', body });
            const data = await res.json();
            if (data.success) {
                droneUploadForm.reset();
                bootstrap.Modal.getOrCreateInstance($('droneConfigModal')).hide();
                applyDroneSource(data);
                // Show the clip straight away, even if the webcam was selected.
                if (activeCameraSource !== 'drone') $('srcDrone').click();
                toast('Simulation started', data.message, 'success');
            } else {
                setAlert(droneUploadAlert, 'danger', 'bi-exclamation-octagon-fill', data.message || 'Upload failed.');
            }
        } catch (err) {
            setAlert(droneUploadAlert, 'danger', 'bi-x-octagon-fill', 'Failed to upload the video.');
        } finally {
            setBusy(submitBtn, false);
        }
    });

    // ------------------------------------------------------------------
    // Camera sources
    // ------------------------------------------------------------------
    function viewKey(view) {
        return Object.keys(views).find((k) => views[k] === view);
    }

    function showViewportMessage(view, icon, title, text) {
        clearViewportMessage(view);
        const box = document.createElement('div');
        box.className = 'viewport-message';
        box.innerHTML = `
            <div class="viewport-message-card">
                <div class="vm-icon"><i class="bi ${icon}"></i></div>
                <div class="vm-title">${esc(title)}</div>
                <p class="vm-text">${esc(text)}</p>
            </div>`;
        view.viewport.appendChild(box);
    }

    function clearViewportMessage(view) {
        const box = view.viewport.querySelector('.viewport-message');
        if (box) box.remove();
    }

    // Only the visible view holds the MJPEG connection open, so the server isn't
    // decoding the same drone stream for hidden tabs.
    function setDroneStream(view, on, forceReload = false) {
        const streaming = view.img.dataset.streaming === '1';
        if (on && (!streaming || forceReload)) {
            view.img.dataset.streaming = '1';
            view.img.src = `/video_feed?t=${Date.now()}`;
        } else if (!on && streaming) {
            view.img.dataset.streaming = '0';
            view.img.src = BLANK_IMG;
        }
    }

    Object.values(views).forEach((view) => {
        view.img.addEventListener('error', () => {
            if (view.img.dataset.streaming !== '1') return;
            showViewportMessage(view, 'bi-wifi-off', 'Drone stream unreachable',
                'The server video feed could not be loaded. Check Stream settings or switch to the webcam.');
        });
    });

    function showWebcamError(view, err) {
        let text = (err && (err.message || err.name)) || 'Permission denied or no camera found.';
        if (!window.isSecureContext) {
            text += ' Browsers block camera access on plain http:// LAN addresses — open this page via https:// or as "localhost" on the server machine.';
        }
        showViewportMessage(view, 'bi-camera-video-off', 'Webcam unavailable', text);
    }

    async function startWebcam(view) {
        clearViewportMessage(view);
        if (currentStream && currentStream.active) {
            view.video.srcObject = currentStream;
            return;
        }
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            showWebcamError(view, { message: 'Camera API not available in this browser context.' });
            return;
        }
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
                audio: false,
            });
            // The user may have switched source or tab while the permission prompt was open.
            const target = views[activeView];
            if (activeCameraSource !== 'webcam' || !target) {
                stream.getTracks().forEach((t) => t.stop());
                return;
            }
            currentStream = stream;
            target.video.srcObject = stream;
            clearViewportMessage(target);
        } catch (err) {
            console.error('Webcam access error:', err);
            showWebcamError(view, err);
        }
    }

    function stopWebcam() {
        if (currentStream) {
            currentStream.getTracks().forEach((t) => t.stop());
            currentStream = null;
        }
        Object.values(views).forEach((view) => { view.video.srcObject = null; });
    }

    function renderSourceLabels() {
        const isDrone = activeCameraSource === 'drone';
        const simulated = isDrone && droneMode === 'video';

        sbSourceLabel.textContent = !isDrone ? 'Local webcam' : simulated ? 'Simulated drone' : 'Drone stream';
        sbSourceDot.className = `status-dot ${isDrone ? 'is-live' : 'is-cam'}`;

        Object.values(views).forEach((view) => {
            view.label.className = `source-pill${isDrone ? '' : ' is-webcam'}`;
            view.label.innerHTML = !isDrone
                ? '<i class="bi bi-webcam"></i>Local webcam'
                : simulated
                    ? '<i class="bi bi-film"></i>Simulated drone feed'
                    : '<i class="bi bi-airplane-engines"></i>Drone feed';
            view.viewport.querySelector('.hud-source').textContent = !isDrone ? 'Webcam' : simulated ? 'Drone sim' : 'Drone';
        });
    }

    function syncCameraViews() {
        const isDrone = activeCameraSource === 'drone';
        renderSourceLabels();

        Object.entries(views).forEach(([key, view]) => {
            view.img.classList.toggle('d-none', !isDrone);
            view.video.classList.toggle('d-none', isDrone);
            clearViewportMessage(view);
            setDroneStream(view, isDrone && activeView === key);
        });

        if (isDrone || !views[activeView]) {
            stopWebcam();
        } else {
            startWebcam(views[activeView]);
        }
    }

    cameraSourceRadios.forEach((radio) => {
        radio.addEventListener('change', (e) => {
            activeCameraSource = e.target.value;
            overlayHoldUntil = 0;
            Object.values(views).forEach((view) => clearOverlay(view));
            syncCameraViews();
        });
    });

    // Grabs a frame from the active source. Returns { image, width, height } or null.
    async function getCaptureFrame(view) {
        if (activeCameraSource === 'drone') {
            try {
                const res = await fetch('/api/drone/snapshot');
                const data = await res.json();
                if (data.success && data.image) {
                    const size = await imageSize(data.image);
                    return { image: data.image, ...size };
                }
            } catch (err) {
                console.error('Drone snapshot error:', err);
            }
            return null;
        }

        const video = view.video;
        if (!video.videoWidth || !video.videoHeight) return null;
        view.canvas.width = video.videoWidth;
        view.canvas.height = video.videoHeight;
        view.canvas.getContext('2d').drawImage(video, 0, 0, video.videoWidth, video.videoHeight);
        return { image: view.canvas.toDataURL('image/jpeg', 0.85), width: video.videoWidth, height: video.videoHeight };
    }

    function imageSize(src) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => resolve({ width: img.naturalWidth || 640, height: img.naturalHeight || 480 });
            img.onerror = () => resolve({ width: 640, height: 480 });
            img.src = src;
        });
    }

    function noFrameMessage() {
        return activeCameraSource === 'drone'
            ? 'Could not grab a frame from the drone stream. Check Stream settings and try again.'
            : 'The webcam is not ready yet. Allow camera access and try again.';
    }

    // ------------------------------------------------------------------
    // Overlay drawing
    // ------------------------------------------------------------------
    // Media is drawn with object-fit: contain, so frame pixels map into a letterboxed rect.
    function prepareOverlay(view, frameW, frameH) {
        const canvas = view.overlay;
        const rect = canvas.getBoundingClientRect();
        const dpr = window.devicePixelRatio || 1;
        canvas.width = Math.max(1, Math.round(rect.width * dpr));
        canvas.height = Math.max(1, Math.round(rect.height * dpr));
        const ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        ctx.clearRect(0, 0, rect.width, rect.height);

        const fw = frameW || 640;
        const fh = frameH || 480;
        const scale = Math.min(rect.width / fw, rect.height / fh);
        return {
            ctx,
            width: rect.width,
            height: rect.height,
            scale,
            offX: (rect.width - fw * scale) / 2,
            offY: (rect.height - fh * scale) / 2,
        };
    }

    function clearOverlay(view) {
        view.lastDraw = null;
        const ctx = view.overlay.getContext('2d');
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.clearRect(0, 0, view.overlay.width, view.overlay.height);
        view.guide.classList.remove('face-detected');
    }

    function pathRoundRect(ctx, x, y, w, h, r) {
        ctx.beginPath();
        if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
        else ctx.rect(x, y, w, h);
    }

    function drawPill(g, text, x, y, fill, color, align = 'left') {
        const { ctx } = g;
        ctx.save();
        ctx.font = '600 12px Inter, system-ui, sans-serif';
        const h = 22;
        const w = ctx.measureText(text).width + 18;
        let left = align === 'center' ? x - w / 2 : x;
        let top = y - h;
        left = Math.max(6, Math.min(left, g.width - w - 6));
        top = Math.max(6, Math.min(top, g.height - h - 6));
        ctx.shadowColor = 'rgba(0, 0, 0, 0.35)';
        ctx.shadowBlur = 8;
        ctx.fillStyle = fill;
        pathRoundRect(ctx, left, top, w, h, 11);
        ctx.fill();
        ctx.shadowBlur = 0;
        ctx.fillStyle = color;
        ctx.textBaseline = 'middle';
        ctx.fillText(text, left + 9, top + h / 2 + 0.5);
        ctx.restore();
    }

    function drawBrackets(ctx, x, y, w, h, color) {
        const len = Math.max(10, Math.min(w, h) * 0.24);
        ctx.save();
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.08;
        ctx.fillRect(x, y, w, h);
        ctx.globalAlpha = 0.4;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, w, h);
        ctx.globalAlpha = 1;
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.shadowColor = color;
        ctx.shadowBlur = 10;
        ctx.beginPath();
        ctx.moveTo(x, y + len); ctx.lineTo(x, y); ctx.lineTo(x + len, y);
        ctx.moveTo(x + w - len, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + len);
        ctx.moveTo(x + w, y + h - len); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w - len, y + h);
        ctx.moveTo(x + len, y + h); ctx.lineTo(x, y + h); ctx.lineTo(x, y + h - len);
        ctx.stroke();
        ctx.restore();
    }

    // Live tracking: circular reticle around each detected face.
    function drawLiveFaces(view, faces, frameW, frameH) {
        view.lastDraw = () => drawLiveFaces(view, faces, frameW, frameH);
        const g = prepareOverlay(view, frameW, frameH);
        const { ctx } = g;
        view.guide.classList.toggle('face-detected', faces.length > 0);

        faces.forEach((f, i) => {
            const cx = g.offX + f.cx * g.scale;
            const cy = g.offY + f.cy * g.scale;
            const r = Math.max(f.w, f.h) * g.scale * 0.65;

            ctx.save();
            ctx.beginPath();
            ctx.arc(cx, cy, r, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(34, 211, 238, 0.07)';
            ctx.fill();
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = 'rgba(34, 211, 238, 0.5)';
            ctx.stroke();

            ctx.lineWidth = 3;
            ctx.lineCap = 'round';
            ctx.strokeStyle = LIVE_STROKE;
            ctx.shadowColor = LIVE_STROKE;
            ctx.shadowBlur = 12;
            for (let k = 0; k < 4; k++) {
                const mid = Math.PI / 4 + (k * Math.PI) / 2;
                ctx.beginPath();
                ctx.arc(cx, cy, r, mid - 0.32, mid + 0.32);
                ctx.stroke();
            }
            ctx.restore();

            drawPill(g, faces.length > 1 ? `Face ${i + 1}` : 'Face', cx, cy - r - 8, LIVE_PILL, '#ffffff', 'center');
        });
    }

    // Scan results: colour-coded corner brackets with the recognised name.
    function drawResultFaces(view, faces, frameW, frameH) {
        view.lastDraw = () => drawResultFaces(view, faces, frameW, frameH);
        const g = prepareOverlay(view, frameW, frameH);
        view.guide.classList.toggle('face-detected', faces.length > 0);

        faces.forEach((f) => {
            const s = STATUS[f.status] || STATUS.unknown;
            const x = g.offX + f.box.x * g.scale;
            const y = g.offY + f.box.y * g.scale;
            const w = f.box.w * g.scale;
            const h = f.box.h * g.scale;
            drawBrackets(g.ctx, x, y, w, h, s.stroke);
            drawPill(g, f.name, x, y - 6, s.pill, s.text);
        });
    }

    const resizeObserver = new ResizeObserver(() => {
        Object.values(views).forEach((view) => { if (view.lastDraw) view.lastDraw(); });
    });
    Object.values(views).forEach((view) => resizeObserver.observe(view.viewport));

    function setDetectChip(view, state, text) {
        view.chip.className = `hud-chip hud-bl detect-chip${state ? ` is-${state}` : ''}`;
        view.chip.querySelector('.detect-text').textContent = text;
    }

    function updateDetectChip(view, count) {
        if (count === null) {
            setDetectChip(view, 'off', activeCameraSource === 'drone' ? 'No drone signal' : 'Waiting for webcam');
        } else if (count === 0) {
            setDetectChip(view, '', 'Searching for faces…');
        } else if (view === views.register && count > 1) {
            setDetectChip(view, 'warn', `${count} faces · largest is enrolled`);
        } else {
            setDetectChip(view, 'found', `${plural(count, 'face')} in view`);
        }
    }

    // ------------------------------------------------------------------
    // Background face detection loop (chained, so slow responses never pile up)
    // ------------------------------------------------------------------
    async function detectionTick() {
        const view = views[activeView];
        // With auto-scan on, the attendance view runs full recognition instead of plain detection.
        const auto = autoScan && view === views.attendance;
        try {
            if (auto && !document.hidden && !scanInFlight) {
                await runScan(true);
            } else if (!auto && view && !document.hidden && !scanInFlight && Date.now() >= overlayHoldUntil) {
                const frame = await getCaptureFrame(view);
                const stillCurrent = () => views[activeView] === view && !scanInFlight && Date.now() >= overlayHoldUntil;
                if (!frame) {
                    if (stillCurrent()) {
                        clearOverlay(view);
                        updateDetectChip(view, null);
                    }
                } else {
                    const res = await fetch('/api/detect_face', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ image: frame.image }),
                    });
                    const data = await res.json();
                    if (stillCurrent()) {
                        const faces = (data.success && data.detected && data.faces) || [];
                        drawLiveFaces(view, faces, frame.width, frame.height);
                        updateDetectChip(view, faces.length);
                    }
                }
            }
        } catch (err) {
            // Background polling: ignore transient errors.
        } finally {
            setTimeout(detectionTick, DETECT_INTERVAL_MS);
        }
    }

    // ------------------------------------------------------------------
    // Mark attendance
    // ------------------------------------------------------------------
    function renderFaceResultRow(f, i) {
        const s = STATUS[f.status] || STATUS.unknown;
        const avatar = f.status === 'unknown'
            ? '<span class="avatar is-unknown" aria-hidden="true"><i class="bi bi-person"></i></span>'
            : avatarHtml(f.name);
        let sub = f.reg_no ? `<span class="mono">${esc(f.reg_no)}</span>` : (UNKNOWN_REASONS[f.reason] || 'Not registered');
        if (f.status === 'already_marked' && f.timestamp) sub += ` · in at ${esc(formatTime(f.timestamp))}`;
        else if (f.status === 'marked' && f.timestamp) sub += ` · ${esc(formatTime(f.timestamp))}`;
        return `
            <div class="face-item tone-${s.tone}" style="animation-delay:${i * 60}ms">
                ${avatar}
                <div class="face-meta">
                    <div class="face-row">
                        <div class="face-name">${esc(f.name)}</div>
                        <span class="status-pill tone-${s.tone}"><i class="bi ${s.icon}"></i>${s.label}</span>
                    </div>
                    <div class="face-sub">${sub}</div>
                </div>
            </div>`;
    }

    let lastFacesKey = '';

    function showScanState(tone, icon, message, faces, quiet = false) {
        attIdleState.classList.add('d-none');
        attResultState.classList.remove('d-none');
        setAlert(attStatusBadge, tone, icon, message);

        const counts = { marked: 0, already_marked: 0, unknown: 0 };
        faces.forEach((f) => { counts[f.status] = (counts[f.status] || 0) + 1; });
        attCountMarked.textContent = counts.marked;
        attCountAlready.textContent = counts.already_marked;
        attCountUnknown.textContent = counts.unknown;
        // Auto-scan repeats every few seconds: only rebuild (and re-animate) the list when it changes.
        const key = faces.map((f) => `${f.name}:${f.status}`).sort().join('|');
        if (!quiet || key !== lastFacesKey) attFacesList.innerHTML = faces.map(renderFaceResultRow).join('');
        lastFacesKey = key;
        attLastScan.textContent = `Last scan at ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}`;
        return counts;
    }

    // One recognition pass over the current frame. A manual scan (the button) reports every
    // outcome; an auto-scan runs from the background loop and stays quiet unless someone new is marked.
    async function runScan(auto) {
        const view = views.attendance;
        scanInFlight = true;
        if (!auto) {
            setBusy(btnMarkAttendance, true, 'Scanning…');
            view.viewport.classList.add('is-scanning');
        }

        try {
            const frame = await getCaptureFrame(view);
            if (!frame) {
                if (auto) {
                    clearOverlay(view);
                    updateDetectChip(view, null);
                } else {
                    showScanState('danger', 'bi-x-octagon-fill', noFrameMessage(), []);
                }
                return;
            }
            if (!auto) flash(view.viewport);

            const response = await fetch('/api/attendance', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: frame.image }),
            });
            const data = await response.json();
            const faces = (data.success && data.faces) || [];

            if (data.success && faces.length > 0) {
                const counts = showScanState(
                    faces.some((f) => f.status === 'marked') ? 'success'
                        : faces.some((f) => f.status === 'already_marked') ? 'neutral' : 'warning',
                    faces.some((f) => f.status === 'marked') ? 'bi-check-circle-fill' : 'bi-info-circle-fill',
                    data.message, faces, auto,
                );
                drawResultFaces(view, faces, frame.width, frame.height);
                overlayHoldUntil = Date.now() + RESULT_HOLD_MS;
                setDetectChip(view, 'result', `${auto ? 'Auto-scan' : 'Scan result'} · ${plural(faces.length, 'face')}`);
                if (counts.marked > 0) {
                    const names = faces.filter((f) => f.status === 'marked').map((f) => f.name);
                    const who = names.length <= 3 ? names.join(', ') : plural(names.length, 'student');
                    toast('Attendance marked', `${who} checked in just now.`, 'success');
                }
                if (!auto || counts.marked > 0) refreshData();
            } else if (auto) {
                clearOverlay(view);
                updateDetectChip(view, data.success ? 0 : null);
            } else if (data.success) {
                showScanState('warning', 'bi-exclamation-triangle-fill', data.message || 'No faces detected in the frame.', []);
                clearOverlay(view);
            } else {
                showScanState('danger', 'bi-x-octagon-fill', data.message || 'Face scan error.', []);
                clearOverlay(view);
            }
        } catch (err) {
            if (auto) throw err;
            console.error(err);
            showScanState('danger', 'bi-x-octagon-fill', 'Error processing attendance. Please try again.', []);
        } finally {
            scanInFlight = false;
            if (!auto) {
                view.viewport.classList.remove('is-scanning');
                setBusy(btnMarkAttendance, false);
            }
        }
    }

    btnMarkAttendance.addEventListener('click', () => runScan(false));

    autoScanToggle.addEventListener('change', () => {
        autoScan = autoScanToggle.checked;
        views.attendance.viewport.classList.toggle('is-auto', autoScan);
        if (autoScan) setDetectChip(views.attendance, 'result', 'Auto-scan on');
    });

    // ------------------------------------------------------------------
    // Registration
    // ------------------------------------------------------------------
    function renderIdCardDraft() {
        const name = regName.value.trim();
        const reg = regId.value.trim();
        regIdCard.classList.remove('is-registered');
        regIdName.textContent = name || 'Student name';
        regIdReg.textContent = reg || 'REG-0000-000';
        regIdPhoto.innerHTML = name ? esc(initials(name)) : '<i class="bi bi-person"></i>';
        regIdStatus.innerHTML = '<i class="bi bi-pencil"></i>Draft';
    }

    function renderIdCardRegistered(student, photo) {
        regIdCard.classList.add('is-registered');
        regIdName.textContent = student.name;
        regIdReg.textContent = student.reg_no;
        regIdPhoto.innerHTML = photo
            ? `<img src="${esc(photo)}" alt="Captured face of ${esc(student.name)}">`
            : esc(initials(student.name));
        regIdStatus.innerHTML = '<i class="bi bi-patch-check-fill"></i>Registered';
    }

    [regName, regId].forEach((input) => {
        input.addEventListener('input', () => {
            input.classList.remove('is-invalid');
            renderIdCardDraft();
        });
    });

    regForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const name = regName.value.trim();
        const regNo = regId.value.trim();

        regName.classList.toggle('is-invalid', !name);
        regId.classList.toggle('is-invalid', !regNo);
        if (!name || !regNo) {
            (name ? regId : regName).focus();
            return;
        }

        const view = views.register;
        hideAlert(regAlert);
        setBusy(btnRegister, true, 'Capturing…');
        view.viewport.classList.add('is-scanning');
        scanInFlight = true;

        try {
            // Several frames -> several face samples, so recognition copes with small
            // changes in pose and expression instead of hinging on one snapshot.
            const frames = [];
            for (let i = 0; i < ENROLL_FRAMES; i++) {
                if (i > 0) await new Promise((r) => setTimeout(r, ENROLL_FRAME_GAP_MS));
                btnRegister.querySelector('span:last-child').textContent = `Capturing ${i + 1}/${ENROLL_FRAMES}…`;
                const f = await getCaptureFrame(view);
                if (f) frames.push(f);
            }
            const frame = frames[0];
            if (!frame) {
                setAlert(regAlert, 'danger', 'bi-x-octagon-fill', noFrameMessage());
                return;
            }
            flash(view.viewport);
            btnRegister.querySelector('span:last-child').textContent = 'Registering…';

            const response = await fetch('/api/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name, reg_no: regNo, images: frames.map((f) => f.image) }),
            });
            const data = await response.json();

            if (data.success) {
                setAlert(regAlert, 'success', 'bi-check-circle-fill', data.message);
                renderIdCardRegistered(data.student || { name, reg_no: regNo }, data.cropped_face);
                regForm.reset();
                if (data.face_box) {
                    drawResultFaces(view, [{ box: data.face_box, status: 'marked', name }], frame.width, frame.height);
                    overlayHoldUntil = Date.now() + 4000;
                    setDetectChip(view, 'result', 'Face enrolled');
                }
                toast('Student registered', `${name} (${regNo}) can now be recognised.`, 'success');
                refreshData();
            } else {
                setAlert(regAlert, 'danger', 'bi-exclamation-octagon-fill', data.message || 'Registration failed.');
            }
        } catch (err) {
            console.error(err);
            setAlert(regAlert, 'danger', 'bi-x-octagon-fill', 'Failed to register student.');
        } finally {
            scanInFlight = false;
            view.viewport.classList.remove('is-scanning');
            setBusy(btnRegister, false);
        }
    });

    // ------------------------------------------------------------------
    // Records: data, stats, tables
    // ------------------------------------------------------------------
    function presentTodayIds() {
        const today = todayKey();
        return new Set(attendanceLogs.filter((a) => a.date === today).map((a) => a.student_id));
    }

    function renderStats() {
        const today = todayKey();
        const total = students.length;
        const present = presentTodayIds().size;
        const rate = total ? Math.min(100, Math.round((present / total) * 100)) : 0;
        const days = new Set(attendanceLogs.map((a) => a.date)).size;

        $('stat-students').textContent = total;
        $('stat-present').textContent = present;
        $('stat-present-sub').textContent = `of ${total} registered`;
        $('stat-rate').textContent = `${rate}%`;
        $('stat-rate-meter').style.width = `${rate}%`;
        $('stat-logs').textContent = attendanceLogs.length;
        $('stat-logs-sub').textContent = `across ${plural(days, 'day')}`;

        sbPresent.textContent = present;
        sbTotal.textContent = total;
        sbMeter.style.width = `${rate}%`;
        navStudentCount.textContent = total;

        const todayLogs = attendanceLogs.filter((a) => a.date === today);
        todayCount.textContent = todayLogs.length;
        todayFeed.innerHTML = todayLogs.length
            ? todayLogs.slice(0, 50).map((a) => `
                <li>
                    ${avatarHtml(a.name)}
                    <div class="feed-meta">
                        <div class="feed-name">${esc(a.name)}</div>
                        <div class="feed-sub mono">${esc(a.reg_no)}</div>
                    </div>
                    <time class="feed-time mono" datetime="${esc(a.timestamp)}">${esc(formatTime(a.timestamp))}</time>
                </li>`).join('')
            : `<li class="d-block"><div class="empty-state">
                    <div class="empty-icon is-small"><i class="bi bi-cup-hot"></i></div>
                    <div class="empty-title">No check-ins yet today</div>
                    <p class="empty-text">Recognised students will appear here as they are scanned.</p>
               </div></li>`;
    }

    function emptyRow(colspan, icon, title, text, action = '') {
        return `<tr class="empty-row"><td colspan="${colspan}" class="empty-cell">
            <div class="empty-state">
                <div class="empty-icon is-small"><i class="bi ${icon}"></i></div>
                <div class="empty-title">${esc(title)}</div>
                <p class="empty-text">${esc(text)}</p>
                ${action}
            </div></td></tr>`;
    }

    function renderStudents() {
        const q = studentSearch.value.trim().toLowerCase();
        const present = presentTodayIds();
        const rows = students.filter((s) => !q
            || String(s.name).toLowerCase().includes(q)
            || String(s.reg_no).toLowerCase().includes(q));

        studentsCaption.textContent = q
            ? `${rows.length} of ${plural(students.length, 'student')}`
            : plural(students.length, 'student');

        if (!students.length) {
            studentsTableBody.innerHTML = emptyRow(3, 'bi-people', 'No students registered yet',
                'Enrol the first student to start recognising faces.',
                '<button type="button" class="btn btn-primary btn-sm mt-3" data-goto="register-tab"><i class="bi bi-person-plus me-1"></i>Register a student</button>');
            return;
        }
        if (!rows.length) {
            studentsTableBody.innerHTML = emptyRow(3, 'bi-search', 'No matches', `No student matches “${studentSearch.value.trim()}”.`);
            return;
        }
        studentsTableBody.innerHTML = rows.map((s) => {
            const isPresent = present.has(s.id);
            return `
                <tr>
                    <td>
                        <div class="person-cell">
                            ${avatarHtml(s.name)}
                            <div>
                                <div class="person-name">${highlight(s.name, q)}</div>
                                <div class="person-sub">ID ${esc(s.id)}</div>
                            </div>
                        </div>
                    </td>
                    <td class="mono">${highlight(s.reg_no, q)}</td>
                    <td>${isPresent
                        ? '<span class="status-pill tone-success"><i class="bi bi-check-circle-fill"></i>Present</span>'
                        : '<span class="status-pill tone-neutral"><i class="bi bi-dash-circle"></i>Absent</span>'}</td>
                </tr>`;
        }).join('');
    }

    function filteredLogs() {
        const q = logSearch.value.trim().toLowerCase();
        const dateFilter = logDateFilter.value;
        const today = todayKey();
        return attendanceLogs.filter((a) => {
            if (dateFilter === 'today' && a.date !== today) return false;
            if (dateFilter !== 'all' && dateFilter !== 'today' && a.date !== dateFilter) return false;
            if (!q) return true;
            return String(a.name).toLowerCase().includes(q) || String(a.reg_no).toLowerCase().includes(q);
        });
    }

    function syncDateFilterOptions() {
        const today = todayKey();
        const selected = logDateFilter.value;
        const dates = [...new Set(attendanceLogs.map((a) => a.date))]
            .filter((d) => d && d !== today)
            .sort()
            .reverse();
        logDateFilter.innerHTML = '<option value="all">All dates</option><option value="today">Today</option>'
            + dates.map((d) => `<option value="${esc(d)}">${esc(formatDateKey(d))}</option>`).join('');
        logDateFilter.value = [...logDateFilter.options].some((o) => o.value === selected) ? selected : 'all';
    }

    function renderLogs() {
        const q = logSearch.value.trim().toLowerCase();
        const rows = filteredLogs();
        const today = todayKey();

        logsCaption.textContent = rows.length === attendanceLogs.length
            ? plural(attendanceLogs.length, 'entry', 'entries')
            : `${rows.length} of ${plural(attendanceLogs.length, 'entry', 'entries')}`;

        if (!attendanceLogs.length) {
            attendanceTableBody.innerHTML = emptyRow(4, 'bi-journal', 'No attendance logged yet',
                'Run a scan from Mark Attendance and check-ins will be recorded here.',
                '<button type="button" class="btn btn-primary btn-sm mt-3" data-goto="attendance-tab"><i class="bi bi-camera-video me-1"></i>Start scanning</button>');
            return;
        }
        if (!rows.length) {
            attendanceTableBody.innerHTML = emptyRow(4, 'bi-funnel', 'No matching entries', 'Try a different search or date.');
            return;
        }
        attendanceTableBody.innerHTML = rows.map((a) => `
            <tr>
                <td>
                    <div class="person-cell">
                        ${avatarHtml(a.name)}
                        <div class="person-name">${highlight(a.name, q)}</div>
                    </div>
                </td>
                <td class="mono">${highlight(a.reg_no, q)}</td>
                <td>${a.date === today ? '<span class="status-pill tone-success">Today</span>' : `<span class="cell-muted">${esc(formatDateKey(a.date))}</span>`}</td>
                <td class="mono cell-muted">${esc(formatTime(a.timestamp, true))}</td>
            </tr>`).join('');
    }

    async function refreshData(triggerBtn) {
        const icon = triggerBtn && triggerBtn.querySelector('.bi-arrow-clockwise');
        if (icon) icon.classList.add('is-spinning');
        try {
            const [sRes, aRes] = await Promise.allSettled([
                fetch('/api/students').then((r) => r.json()),
                fetch('/api/attendance').then((r) => r.json()),
            ]);
            if (sRes.status === 'fulfilled' && sRes.value.success) students = sRes.value.students || [];
            if (aRes.status === 'fulfilled' && aRes.value.success) attendanceLogs = aRes.value.attendance || [];
            if (sRes.status === 'rejected' || aRes.status === 'rejected') {
                if (triggerBtn) toast('Could not refresh', 'The server did not respond. Is it still running?', 'danger');
            }
        } finally {
            syncDateFilterOptions();
            renderStats();
            renderStudents();
            renderLogs();
            if (icon) setTimeout(() => icon.classList.remove('is-spinning'), 300);
        }
    }

    function exportLogsCsv() {
        const rows = filteredLogs();
        if (!rows.length) {
            toast('Nothing to export', 'No log entries match the current filters.', 'warning');
            return;
        }
        // Quote every cell and neutralise leading formula characters for spreadsheet safety.
        const cell = (v) => {
            let s = String(v ?? '');
            if (/^[=+\-@\t\r]/.test(s)) s = `'${s}`;
            return `"${s.replace(/"/g, '""')}"`;
        };
        const lines = [['Reg No', 'Name', 'Date', 'Time'].map(cell).join(',')].concat(
            rows.map((a) => [a.reg_no, a.name, a.date, String(a.timestamp || '').split(' ')[1] || ''].map(cell).join(',')),
        );
        const blob = new Blob([`﻿${lines.join('\r\n')}`], { type: 'text/csv;charset=utf-8' });
        const filter = logDateFilter.value === 'today' ? todayKey() : logDateFilter.value;
        const link = document.createElement('a');
        link.href = URL.createObjectURL(blob);
        link.download = `attendance-${filter === 'all' ? 'all' : filter}.csv`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        setTimeout(() => URL.revokeObjectURL(link.href), 1000);
        toast('Export ready', `${plural(rows.length, 'entry', 'entries')} saved to ${link.download}`, 'success');
    }

    btnRefreshStudents.addEventListener('click', () => refreshData(btnRefreshStudents));
    btnRefreshLogs.addEventListener('click', () => refreshData(btnRefreshLogs));
    btnExportLogs.addEventListener('click', exportLogsCsv);
    studentSearch.addEventListener('input', renderStudents);
    logSearch.addEventListener('input', renderLogs);
    logDateFilter.addEventListener('change', renderLogs);

    // ------------------------------------------------------------------
    // Navigation
    // ------------------------------------------------------------------
    document.querySelectorAll('#appTabs [data-bs-toggle="pill"]').forEach((tabEl) => {
        tabEl.addEventListener('shown.bs.tab', (event) => {
            const meta = PAGE_META[event.target.getAttribute('data-bs-target')];
            if (!meta) return;
            activeView = meta.view;
            pageTitle.textContent = meta.title;
            pageSubtitle.textContent = meta.subtitle;
            document.title = `${meta.title} · Facial Attendance System`;
            history.replaceState(null, '', meta.hash || window.location.pathname + window.location.search);

            overlayHoldUntil = 0;
            Object.values(views).forEach((view) => clearOverlay(view));
            syncCameraViews();
            if (activeView === 'records') refreshData();
        });
    });

    document.addEventListener('click', (e) => {
        const target = e.target.closest('[data-goto]');
        if (target) bootstrap.Tab.getOrCreateInstance($(target.dataset.goto)).show();
    });

    // ------------------------------------------------------------------
    // Boot
    // ------------------------------------------------------------------
    renderIdCardDraft();
    refreshData();

    // Deep links (#register, #records) open that section; its shown.bs.tab handler syncs the cameras.
    const initialTab = { '#register': 'register-tab', '#records': 'records-tab' }[window.location.hash];
    if (initialTab) bootstrap.Tab.getOrCreateInstance($(initialTab)).show();
    else syncCameraViews();

    detectionTick();
});

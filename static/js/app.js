document.addEventListener('DOMContentLoaded', () => {
    let currentStream = null;
    let activeCameraSource = 'drone'; // 'drone' or 'webcam'

    const attVideo = document.getElementById('att-video');
    const attDroneImg = document.getElementById('att-drone-img');
    const attCanvas = document.getElementById('att-canvas');
    const attStreamLabel = document.getElementById('att-stream-label');

    const btnMarkAttendance = document.getElementById('btn-mark-attendance');
    const attIdleState = document.getElementById('att-idle-state');
    const attResultState = document.getElementById('att-result-state');
    const attStatusBadge = document.getElementById('att-status-badge');
    const attStudentInfo = document.getElementById('att-student-info');
    const attStudentName = document.getElementById('att-student-name');
    const attStudentReg = document.getElementById('att-student-reg');
    const attTime = document.getElementById('att-time');

    const regVideo = document.getElementById('reg-video');
    const regDroneImg = document.getElementById('reg-drone-img');
    const regCanvas = document.getElementById('reg-canvas');
    const regStreamLabel = document.getElementById('reg-stream-label');

    const regForm = document.getElementById('reg-form');
    const regName = document.getElementById('reg-name');
    const regId = document.getElementById('reg-id');
    const regAlert = document.getElementById('reg-alert');
    const btnRegister = document.getElementById('btn-register-student');

    const studentsTableBody = document.getElementById('students-table-body');
    const attendanceTableBody = document.getElementById('attendance-table-body');
    const btnRefreshStudents = document.getElementById('btn-refresh-students');
    const btnRefreshLogs = document.getElementById('btn-refresh-logs');

    const cameraSourceRadios = document.querySelectorAll('input[name="cameraSource"]');
    const droneStatusBadge = document.getElementById('drone-status-badge');
    const droneConfigForm = document.getElementById('drone-config-form');
    const droneUrlInput = document.getElementById('drone-url-input');
    const droneConfigAlert = document.getElementById('drone-config-alert');

    // Fetch current drone stream config
    async function loadDroneConfig() {
        try {
            const res = await fetch('/api/drone/config');
            const data = await res.json();
            if (data.success && data.stream_url) {
                droneUrlInput.value = data.stream_url;
            }
        } catch (err) {
            console.error("Error loading drone config:", err);
        }
    }
    loadDroneConfig();

    // Drone config form submit
    droneConfigForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const newUrl = droneUrlInput.value.trim();
        try {
            const res = await fetch('/api/drone/config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ stream_url: newUrl })
            });
            const data = await res.json();
            droneConfigAlert.classList.remove('d-none');
            if (data.success) {
                droneConfigAlert.className = 'alert alert-success';
                droneConfigAlert.textContent = data.message;
                // Refresh drone img src
                attDroneImg.src = '/video_feed?t=' + new Date().getTime();
                regDroneImg.src = '/video_feed?t=' + new Date().getTime();
            } else {
                droneConfigAlert.className = 'alert alert-danger';
                droneConfigAlert.textContent = data.message;
            }
        } catch (err) {
            droneConfigAlert.classList.remove('d-none');
            droneConfigAlert.className = 'alert alert-danger';
            droneConfigAlert.textContent = 'Failed to update drone stream configuration.';
        }
    });

    // Initialize local webcam stream
    async function startWebcam(videoElement) {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
        }
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
                audio: false
            });
            currentStream = stream;
            videoElement.srcObject = stream;
        } catch (err) {
            console.error("Webcam access error:", err);
        }
    }

    function stopWebcam() {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            currentStream = null;
        }
    }

    // Camera Source switch handler
    cameraSourceRadios.forEach(radio => {
        radio.addEventListener('change', (e) => {
            activeCameraSource = e.target.value;
            updateCameraViews();
        });
    });

    function updateCameraViews() {
        const activeTab = document.querySelector('.nav-link.active').getAttribute('id');

        if (activeCameraSource === 'drone') {
            droneStatusBadge.className = 'badge bg-success px-3 py-2 fs-6';
            droneStatusBadge.innerHTML = '<i class="bi bi-broadcast me-1"></i>Drone Stream Source Active';

            attDroneImg.classList.remove('d-none');
            attVideo.classList.add('d-none');
            attStreamLabel.className = 'badge bg-primary';
            attStreamLabel.textContent = 'Drone Feed';

            regDroneImg.classList.remove('d-none');
            regVideo.classList.add('d-none');
            regStreamLabel.className = 'badge bg-success';
            regStreamLabel.textContent = 'Drone Feed';

            stopWebcam();
        } else {
            droneStatusBadge.className = 'badge bg-secondary px-3 py-2 fs-6';
            droneStatusBadge.innerHTML = '<i class="bi bi-webcam me-1"></i>Local Webcam Active';

            attDroneImg.classList.add('d-none');
            attVideo.classList.remove('d-none');
            attStreamLabel.className = 'badge bg-secondary';
            attStreamLabel.textContent = 'Local Webcam';

            regDroneImg.classList.add('d-none');
            regVideo.classList.remove('d-none');
            regStreamLabel.className = 'badge bg-secondary';
            regStreamLabel.textContent = 'Local Webcam';

            if (activeTab === 'attendance-tab') {
                startWebcam(attVideo);
            } else if (activeTab === 'register-tab') {
                startWebcam(regVideo);
            }
        }
    }

    // Switch tab handler
    const tabEls = document.querySelectorAll('button[data-bs-toggle="pill"]');
    tabEls.forEach(tabEl => {
        tabEl.addEventListener('shown.bs.tab', (event) => {
            const targetId = event.target.getAttribute('data-bs-target');
            if (activeCameraSource === 'webcam') {
                if (targetId === '#attendance-pane') {
                    startWebcam(attVideo);
                } else if (targetId === '#register-pane') {
                    startWebcam(regVideo);
                } else if (targetId === '#records-pane') {
                    stopWebcam();
                    loadStudents();
                    loadAttendanceLogs();
                }
            } else {
                if (targetId === '#records-pane') {
                    loadStudents();
                    loadAttendanceLogs();
                }
            }
        });
    });

    // Capture frame base64 from video element or drone API
    async function getCaptureFrame(videoElement, canvasElement) {
        if (activeCameraSource === 'drone') {
            try {
                const res = await fetch('/api/drone/snapshot');
                const data = await res.json();
                if (data.success && data.image) {
                    return data.image;
                }
            } catch (e) {
                console.error("Drone snapshot error:", e);
            }
        }

        // Fallback or webcam capture
        const width = videoElement.videoWidth || 640;
        const height = videoElement.videoHeight || 480;
        canvasElement.width = width;
        canvasElement.height = height;
        const context = canvasElement.getContext('2d');
        context.drawImage(videoElement, 0, 0, width, height);
        return canvasElement.toDataURL('image/jpeg', 0.85);
    }

    // Function to draw dynamic circular face overlays
    function drawFaceOverlay(overlayCanvas, overlayDiv, faces, frameWidth, frameHeight) {
        if (!overlayCanvas) return;
        const ctx = overlayCanvas.getContext('2d');
        const container = overlayCanvas.parentElement;
        const displayWidth = container.clientWidth || 640;
        const displayHeight = container.clientHeight || 360;

        overlayCanvas.width = displayWidth;
        overlayCanvas.height = displayHeight;
        ctx.clearRect(0, 0, displayWidth, displayHeight);

        if (!faces || faces.length === 0) {
            if (overlayDiv) overlayDiv.classList.remove('face-detected');
            return;
        }

        if (overlayDiv) overlayDiv.classList.add('face-detected');

        const scaleX = displayWidth / (frameWidth || 640);
        const scaleY = displayHeight / (frameHeight || 480);

        faces.forEach(face => {
            const cx = face.cx * scaleX;
            const cy = face.cy * scaleY;
            const radius = Math.max(face.w * scaleX, face.h * scaleY) * 0.65;

            // Draw glowing circular face boundary
            ctx.beginPath();
            ctx.arc(cx, cy, radius, 0, 2 * Math.PI, false);
            ctx.lineWidth = 3;
            ctx.strokeStyle = '#28a745';
            ctx.shadowColor = '#28a745';
            ctx.shadowBlur = 12;
            ctx.stroke();

            // Draw inner crosshair / center target indicator
            ctx.shadowBlur = 0;
            ctx.beginPath();
            ctx.arc(cx, cy, 4, 0, 2 * Math.PI, false);
            ctx.fillStyle = '#28a745';
            ctx.fill();
        });
    }

    // Periodic face detection indicator loop
    let detectionInterval = null;
    function startAutoFaceDetection() {
        if (detectionInterval) clearInterval(detectionInterval);
        detectionInterval = setInterval(async () => {
            const activeTab = document.querySelector('.nav-link.active')?.getAttribute('id');
            let videoEl, canvasEl, overlayCanvasEl, overlayDivEl;

            if (activeTab === 'attendance-tab') {
                videoEl = attVideo;
                canvasEl = attCanvas;
                overlayCanvasEl = document.getElementById('att-overlay-canvas');
                overlayDivEl = document.getElementById('att-face-overlay');
            } else if (activeTab === 'register-tab') {
                videoEl = regVideo;
                canvasEl = regCanvas;
                overlayCanvasEl = document.getElementById('reg-overlay-canvas');
                overlayDivEl = document.getElementById('reg-face-overlay');
            } else {
                return;
            }

            try {
                const imgB64 = await getCaptureFrame(videoEl, canvasEl);
                if (!imgB64) return;

                const response = await fetch('/api/detect_face', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: imgB64 })
                });
                const data = await response.json();
                if (data.success && data.detected && data.faces) {
                    drawFaceOverlay(overlayCanvasEl, overlayDivEl, data.faces, canvasEl.width || 640, canvasEl.height || 480);
                } else {
                    drawFaceOverlay(overlayCanvasEl, overlayDivEl, [], 640, 480);
                }
            } catch (err) {
                // Ignore background interval detection error
            }
        }, 1500);
    }

    startAutoFaceDetection();

    // Handle Mark Attendance
    btnMarkAttendance.addEventListener('click', async () => {
        btnMarkAttendance.disabled = true;
        btnMarkAttendance.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Scanning...';

        const imageB64 = await getCaptureFrame(attVideo, attCanvas);

        try {
            const response = await fetch('/api/attendance', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imageB64 })
            });

            const data = await response.json();

            attIdleState.classList.add('d-none');
            attResultState.classList.remove('d-none');

            if (data.success && data.registered) {
                attStatusBadge.className = 'alert alert-success fw-bold text-start mb-3';
                attStatusBadge.innerHTML = `<i class="bi bi-check-circle-fill me-2"></i>${data.message}`;

                attStudentName.textContent = data.student.name;
                attStudentReg.textContent = data.student.reg_no;
                attTime.textContent = data.attendance.timestamp;
                attStudentInfo.classList.remove('d-none');
            } else if (!data.registered) {
                attStatusBadge.className = 'alert alert-danger fw-bold text-start mb-3';
                attStatusBadge.innerHTML = `<i class="bi bi-exclamation-triangle-fill me-2"></i>Student is not registered.`;
                attStudentInfo.classList.add('d-none');
            } else {
                attStatusBadge.className = 'alert alert-warning fw-bold text-start mb-3';
                attStatusBadge.innerHTML = `<i class="bi bi-info-circle-fill me-2"></i>${data.message || 'Face scan error'}`;
                attStudentInfo.classList.add('d-none');
            }
        } catch (err) {
            attIdleState.classList.add('d-none');
            attResultState.classList.remove('d-none');
            attStatusBadge.className = 'alert alert-danger fw-bold text-start mb-3';
            attStatusBadge.innerHTML = `<i class="bi bi-x-circle-fill me-2"></i>Error processing attendance. Please try again.`;
            attStudentInfo.classList.add('d-none');
        } finally {
            btnMarkAttendance.disabled = false;
            btnMarkAttendance.innerHTML = '<i class="bi bi-person-check me-2"></i>Scan Face & Mark Attendance';
        }
    });

    // Handle Student Registration
    regForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        const name = regName.value.trim();
        const regNo = regId.value.trim();

        if (!name || !regNo) {
            regAlert.className = 'alert alert-danger';
            regAlert.textContent = 'Please enter both Name and Registration Number.';
            regAlert.classList.remove('d-none');
            return;
        }

        btnRegister.disabled = true;
        btnRegister.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Registering...';

        const imageB64 = await getCaptureFrame(regVideo, regCanvas);

        try {
            const response = await fetch('/api/register', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name, reg_no: regNo, image: imageB64 })
            });

            const data = await response.json();

            regAlert.classList.remove('d-none');
            if (data.success) {
                regAlert.className = 'alert alert-success';
                regAlert.innerHTML = `<i class="bi bi-check-circle me-1"></i>${data.message}`;
                regForm.reset();
            } else {
                regAlert.className = 'alert alert-danger';
                regAlert.innerHTML = `<i class="bi bi-exclamation-circle me-1"></i>${data.message}`;
            }
        } catch (err) {
            regAlert.classList.remove('d-none');
            regAlert.className = 'alert alert-danger';
            regAlert.innerHTML = `<i class="bi bi-x-circle me-1"></i>Failed to register student.`;
        } finally {
            btnRegister.disabled = false;
            btnRegister.innerHTML = '<i class="bi bi-person-check-fill me-2"></i>Capture & Register';
        }
    });

    // Fetch and display registered students
    async function loadStudents() {
        try {
            const res = await fetch('/api/students');
            const data = await res.json();
            if (data.success && data.students.length > 0) {
                studentsTableBody.innerHTML = data.students.map(s => `
                    <tr>
                        <td>${s.id}</td>
                        <td class="fw-semibold">${s.reg_no}</td>
                        <td>${s.name}</td>
                    </tr>
                `).join('');
            } else {
                studentsTableBody.innerHTML = '<tr><td colspan="3" class="text-center text-muted py-3">No students registered yet.</td></tr>';
            }
        } catch (err) {
            console.error(err);
        }
    }

    // Fetch and display attendance logs
    async function loadAttendanceLogs() {
        try {
            const res = await fetch('/api/attendance');
            const data = await res.json();
            if (data.success && data.attendance.length > 0) {
                attendanceTableBody.innerHTML = data.attendance.map(a => `
                    <tr>
                        <td class="fw-semibold">${a.reg_no}</td>
                        <td>${a.name}</td>
                        <td><small class="text-muted">${a.timestamp}</small></td>
                    </tr>
                `).join('');
            } else {
                attendanceTableBody.innerHTML = '<tr><td colspan="3" class="text-center text-muted py-3">No attendance logs yet.</td></tr>';
            }
        } catch (err) {
            console.error(err);
        }
    }

    btnRefreshStudents.addEventListener('click', loadStudents);
    btnRefreshLogs.addEventListener('click', loadAttendanceLogs);
});

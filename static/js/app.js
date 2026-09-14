document.addEventListener('DOMContentLoaded', () => {
    let currentStream = null;

    const attVideo = document.getElementById('att-video');
    const attCanvas = document.getElementById('att-canvas');
    const btnMarkAttendance = document.getElementById('btn-mark-attendance');
    const attIdleState = document.getElementById('att-idle-state');
    const attResultState = document.getElementById('att-result-state');
    const attStatusBadge = document.getElementById('att-status-badge');
    const attStudentInfo = document.getElementById('att-student-info');
    const attStudentName = document.getElementById('att-student-name');
    const attStudentReg = document.getElementById('att-student-reg');
    const attTime = document.getElementById('att-time');

    const regVideo = document.getElementById('reg-video');
    const regCanvas = document.getElementById('reg-canvas');
    const regForm = document.getElementById('reg-form');
    const regName = document.getElementById('reg-name');
    const regId = document.getElementById('reg-id');
    const regAlert = document.getElementById('reg-alert');
    const btnRegister = document.getElementById('btn-register-student');

    const studentsTableBody = document.getElementById('students-table-body');
    const attendanceTableBody = document.getElementById('attendance-table-body');
    const btnRefreshStudents = document.getElementById('btn-refresh-students');
    const btnRefreshLogs = document.getElementById('btn-refresh-logs');

    // Initialize camera stream
    async function startCamera(videoElement) {
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
            console.error("Camera access error:", err);
        }
    }

    // Switch tab handler
    const tabEls = document.querySelectorAll('button[data-bs-toggle="pill"]');
    tabEls.forEach(tabEl => {
        tabEl.addEventListener('shown.bs.tab', (event) => {
            const targetId = event.target.getAttribute('data-bs-target');
            if (targetId === '#attendance-pane') {
                startCamera(attVideo);
            } else if (targetId === '#register-pane') {
                startCamera(regVideo);
            } else if (targetId === '#records-pane') {
                if (currentStream) {
                    currentStream.getTracks().forEach(track => track.stop());
                }
                loadStudents();
                loadAttendanceLogs();
            }
        });
    });

    // Start default tab camera (Attendance tab)
    startCamera(attVideo);

    // Capture frame base64 from video element
    function captureFrame(videoElement, canvasElement) {
        const width = videoElement.videoWidth || 640;
        const height = videoElement.videoHeight || 480;
        canvasElement.width = width;
        canvasElement.height = height;
        const context = canvasElement.getContext('2d');
        context.drawImage(videoElement, 0, 0, width, height);
        return canvasElement.toDataURL('image/jpeg', 0.85);
    }

    // Handle Mark Attendance
    btnMarkAttendance.addEventListener('click', async () => {
        btnMarkAttendance.disabled = true;
        btnMarkAttendance.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Scanning...';

        const imageB64 = captureFrame(attVideo, attCanvas);

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
                // Registered student successfully recognized & marked
                attStatusBadge.className = 'alert alert-success fw-bold text-start mb-3';
                attStatusBadge.innerHTML = `<i class="bi bi-check-circle-fill me-2"></i>${data.message}`;

                attStudentName.textContent = data.student.name;
                attStudentReg.textContent = data.student.reg_no;
                attTime.textContent = data.attendance.timestamp;
                attStudentInfo.classList.remove('d-none');
            } else if (!data.registered) {
                // Student NOT registered or not recognized
                attStatusBadge.className = 'alert alert-danger fw-bold text-start mb-3';
                attStatusBadge.innerHTML = `<i class="bi bi-exclamation-triangle-fill me-2"></i>Student is not registered.`;
                attStudentInfo.classList.add('d-none');
            } else {
                // Other error (e.g., no face detected)
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

        const imageB64 = captureFrame(regVideo, regCanvas);

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

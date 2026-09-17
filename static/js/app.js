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
    const attInfLatency = document.getElementById('att-inf-latency');
    const attTxLink = document.getElementById('att-tx-link');
    const attBlockNum = document.getElementById('att-block-num');
    const attGasUsed = document.getElementById('att-gas-used');
    const attChainLatency = document.getElementById('att-chain-latency');

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
    const btnRefreshBenchmarks = document.getElementById('btn-refresh-benchmarks');

    // Admin elements
    const admTermAddr = document.getElementById('adm-term-addr');
    const admContractAddr = document.getElementById('adm-contract-addr');
    const admBlockNum = document.getElementById('adm-block-num');
    const admNodeStatus = document.getElementById('adm-node-status');

    // Benchmark elements
    const bmInfLatency = document.getElementById('bm-inf-latency');
    const bmChainLatency = document.getElementById('bm-chain-latency');
    const bmAccuracy = document.getElementById('bm-accuracy');
    const bmFar = document.getElementById('bm-far');
    const bmFrr = document.getElementById('bm-frr');
    const bmGasSavings = document.getElementById('bm-gas-savings');
    const bmGasIndiv = document.getElementById('bm-gas-indiv');
    const bmGasBatch = document.getElementById('bm-gas-batch');
    const bmTotalTx = document.getElementById('bm-total-tx');

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
            } else if (targetId === '#admin-pane') {
                if (currentStream) {
                    currentStream.getTracks().forEach(track => track.stop());
                }
                loadWeb3Status();
                loadBenchmarks();
            }
        });
    });

    startCamera(attVideo);
    loadWeb3Status();

    function captureFrame(videoElement, canvasElement) {
        const width = videoElement.videoWidth || 640;
        const height = videoElement.videoHeight || 480;
        canvasElement.width = width;
        canvasElement.height = height;
        const context = canvasElement.getContext('2d');
        context.drawImage(videoElement, 0, 0, width, height);
        return canvasElement.toDataURL('image/jpeg', 0.85);
    }

    btnMarkAttendance.addEventListener('click', async () => {
        btnMarkAttendance.disabled = true;
        btnMarkAttendance.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Scanning & Dispatching On-Chain...';

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
                attStatusBadge.className = 'alert alert-success fw-bold text-start mb-3';
                attStatusBadge.innerHTML = `<i class="bi bi-check-circle-fill me-2"></i>${data.message}`;

                attStudentName.textContent = data.student.name;
                attStudentReg.textContent = data.student.reg_no;
                attTime.textContent = data.attendance.timestamp;
                attInfLatency.textContent = data.inference_latency_ms || '--';

                if (data.web3) {
                    const txHash = data.web3.tx_hash || 'N/A';
                    attTxLink.textContent = txHash;
                    attTxLink.href = data.explorer_link || `https://sepolia.etherscan.io/tx/${txHash}`;
                    attBlockNum.textContent = data.web3.block_number || '--';
                    attGasUsed.textContent = data.web3.gas_used ? data.web3.gas_used.toLocaleString() : '--';
                    attChainLatency.textContent = data.web3.latency_ms || '--';
                }

                attStudentInfo.classList.remove('d-none');
            } else if (!data.registered) {
                attStatusBadge.className = 'alert alert-danger fw-bold text-start mb-3';
                attStatusBadge.innerHTML = `<i class="bi bi-exclamation-triangle-fill me-2"></i>${data.message || 'Student is not registered.'}`;
                attStudentInfo.classList.add('d-none');
            } else {
                attStatusBadge.className = 'alert alert-warning fw-bold text-start mb-3';
                attStatusBadge.innerHTML = `<i class="bi bi-shield-exclamation me-2"></i>${data.message || 'Face scan error'}`;
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
        btnRegister.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Extracting Embedding...';

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
            regAlert.innerHTML = `<i class="bi bi-x-circle me-1"></i>Failed to register student vector.`;
        } finally {
            btnRegister.disabled = false;
            btnRegister.innerHTML = '<i class="bi bi-person-check-fill me-2"></i>Capture & Register Vector';
        }
    });

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
                        <td><span class="badge ${s.has_embedding ? 'bg-success' : 'bg-secondary'}">${s.has_embedding ? '128-d Vector Saved' : 'No Vector'}</span></td>
                    </tr>
                `).join('');
            } else {
                studentsTableBody.innerHTML = '<tr><td colspan="4" class="text-center text-muted py-3">No students registered yet.</td></tr>';
            }
        } catch (err) {
            console.error(err);
        }
    }

    async function loadAttendanceLogs() {
        try {
            const res = await fetch('/api/attendance');
            const data = await res.json();
            if (data.success && data.attendance.length > 0) {
                attendanceTableBody.innerHTML = data.attendance.map(a => {
                    const txHash = a.tx_hash ? a.tx_hash : 'N/A';
                    const link = a.tx_hash ? `https://sepolia.etherscan.io/tx/${a.tx_hash}` : '#';
                    return `
                    <tr>
                        <td class="fw-semibold">${a.reg_no}<br><small class="text-muted">${a.name}</small></td>
                        <td><small class="text-muted">${a.timestamp}</small></td>
                        <td>
                            ${a.tx_hash ? `<a href="${link}" target="_blank" class="font-monospace small text-truncate d-inline-block" style="max-width: 150px;">${txHash}</a>` : '<span class="text-muted small">Pending</span>'}
                        </td>
                        <td><span class="badge bg-light text-dark border">${a.block_number || '--'}</span></td>
                        <td><small>${a.gas_used ? a.gas_used.toLocaleString() : '--'}</small></td>
                    </tr>
                `}).join('');
            } else {
                attendanceTableBody.innerHTML = '<tr><td colspan="5" class="text-center text-muted py-3">No attendance logs yet.</td></tr>';
            }
        } catch (err) {
            console.error(err);
        }
    }

    async function loadWeb3Status() {
        try {
            const res = await fetch('/api/web3/status');
            const data = await res.json();
            if (data.success && data.web3_status) {
                const s = data.web3_status;
                if (admTermAddr) admTermAddr.textContent = s.terminal_address || 'N/A';
                if (admContractAddr) admContractAddr.textContent = s.contract_address || 'N/A';
                if (admBlockNum) admBlockNum.textContent = s.block_number !== undefined ? s.block_number : '--';
                if (admNodeStatus) {
                    admNodeStatus.textContent = s.connected ? 'Connected' : 'Disconnected';
                    admNodeStatus.className = s.connected ? 'fs-4 fw-bold text-success' : 'fs-4 fw-bold text-danger';
                }
            }
        } catch (err) {
            console.error("Web3 status error:", err);
        }
    }

    async function loadBenchmarks() {
        try {
            const res = await fetch('/api/benchmarks');
            const data = await res.json();
            if (data.success && data.benchmarks) {
                const bm = data.benchmarks;
                if (bmInfLatency) bmInfLatency.textContent = `${bm.inference_latency.avg_ms} ms`;
                if (bmChainLatency) bmChainLatency.textContent = `${bm.on_chain_latency.avg_ms} ms`;
                if (bmAccuracy) bmAccuracy.textContent = bm.match_accuracy.accuracy;
                if (bmFar) bmFar.textContent = bm.match_accuracy.FAR;
                if (bmFrr) bmFrr.textContent = bm.match_accuracy.FRR;
                if (bmGasSavings) bmGasSavings.textContent = `${bm.gas_analysis.gas_savings_percent}%`;
                if (bmGasIndiv) bmGasIndiv.textContent = `${bm.gas_analysis.individual_call_avg_gas.toLocaleString()} gas`;
                if (bmGasBatch) bmGasBatch.textContent = `${bm.gas_analysis.batched_call_avg_gas.toLocaleString()} gas`;
                if (bmTotalTx) bmTotalTx.textContent = bm.total_transactions;
            }
        } catch (err) {
            console.error("Benchmarks load error:", err);
        }
    }

    btnRefreshStudents.addEventListener('click', loadStudents);
    btnRefreshLogs.addEventListener('click', loadAttendanceLogs);
    if (btnRefreshBenchmarks) btnRefreshBenchmarks.addEventListener('click', loadBenchmarks);
});

// State variables
let currentTab = 'overview';
let pollInterval = null;
let isSimulationActive = false;
let accuracyChart = null;
let lossChart = null;
let diseaseChart = null;
let allSimulations = [];
let allReports = [];

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initCharts();
    loadDashboardData();
    startPolling();

    // Event Listeners
    document.getElementById('simConfigForm').addEventListener('submit', handleStartSimulation);
    document.getElementById('btnStopSimulation').addEventListener('click', handleStopSimulation);
    document.getElementById('btnStopMonitor').addEventListener('click', handleStopSimulation);
    document.getElementById('btnRefreshData').addEventListener('click', loadDashboardData);
    document.getElementById('btnClearLogs').addEventListener('click', () => {
        document.getElementById('terminalLogs').innerHTML = '<div class="terminal-line text-muted">[Logs cleared]</div>';
    });
    document.getElementById('selectReportHistory').addEventListener('change', (e) => {
        if (e.target.value) loadReportDetail(e.target.value);
    });
    document.getElementById('btnDownloadReport').addEventListener('click', downloadCurrentReport);
});

// Navigation
function initNavigation() {
    const tabs = document.querySelectorAll('.nav-tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const tabName = tab.getAttribute('data-tab');
            switchTab(tabName);
        });
    });
}

function switchTab(tabName) {
    currentTab = tabName;
    document.querySelectorAll('.nav-tab').forEach(t => {
        t.classList.toggle('active', t.getAttribute('data-tab') === tabName);
    });
    document.querySelectorAll('.tab-pane').forEach(p => {
        p.classList.toggle('active', p.id === `pane-${tabName}`);
    });

    if (tabName === 'analytics') {
        loadReportsList();
    } else if (tabName === 'dataset') {
        loadDatasetInfo();
    }
}

// Charts Initialization
function initCharts() {
    const chartDefaults = {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        scales: {
            x: {
                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                ticks: { color: '#94a3b8' }
            },
            y: {
                grid: { color: 'rgba(255, 255, 255, 0.05)' },
                ticks: { color: '#94a3b8' }
            }
        },
        plugins: {
            legend: {
                labels: { color: '#f8fafc', font: { family: 'Inter', size: 12 } }
            }
        }
    };

    // 1. Live Accuracy Chart
    const ctxAcc = document.getElementById('liveAccuracyChart').getContext('2d');
    accuracyChart = new Chart(ctxAcc, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            ...chartDefaults,
            scales: {
                ...chartDefaults.scales,
                y: {
                    ...chartDefaults.scales.y,
                    title: { display: true, text: 'Accuracy (%)', color: '#94a3b8' },
                    suggestedMin: 80,
                    suggestedMax: 100
                }
            }
        }
    });

    // 2. Live Loss Chart
    const ctxLoss = document.getElementById('liveLossChart').getContext('2d');
    lossChart = new Chart(ctxLoss, {
        type: 'line',
        data: {
            labels: [],
            datasets: []
        },
        options: {
            ...chartDefaults,
            scales: {
                ...chartDefaults.scales,
                y: {
                    ...chartDefaults.scales.y,
                    title: { display: true, text: 'Loss', color: '#94a3b8' },
                    suggestedMin: 0
                }
            }
        }
    });

    // 3. Disease Distribution Chart
    const ctxDisease = document.getElementById('diseaseDistributionChart').getContext('2d');
    diseaseChart = new Chart(ctxDisease, {
        type: 'doughnut',
        data: {
            labels: [],
            datasets: [{
                data: [],
                backgroundColor: [
                    '#06b6d4',
                    '#10b981',
                    '#8b5cf6',
                    '#f59e0b',
                    '#f43f5e',
                    '#3b82f6'
                ],
                borderWidth: 2,
                borderColor: '#0f172a'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'right', labels: { color: '#f8fafc', font: { family: 'Inter', size: 12 } } }
            }
        }
    });
}

// Polling Loop
function startPolling() {
    if (pollInterval) clearInterval(pollInterval);
    pollInterval = setInterval(fetchStatus, 1500);
    fetchStatus();
}

async function fetchStatus() {
    try {
        const res = await fetch('/api/status');
        if (!res.ok) return;
        const state = await res.json();
        updateStatusUI(state);
    } catch (err) {
        console.error('Failed to fetch status:', err);
    }
}

function updateStatusUI(state) {
    const isRunning = state.is_running;
    isSimulationActive = isRunning;

    // Header Pill
    const pill = document.getElementById('globalStatusPill');
    const pillText = document.getElementById('globalStatusText');
    const liveIndicator = document.getElementById('navLiveIndicator');

    pill.className = `status-pill ${isRunning ? 'running' : ''}`;
    pillText.textContent = isRunning ? (state.status_text || 'Training...') : 'Idle';
    liveIndicator.style.display = isRunning ? 'inline-block' : 'none';

    // Buttons
    document.getElementById('btnStartSimulation').style.display = isRunning ? 'none' : 'inline-flex';
    document.getElementById('btnStopSimulation').style.display = isRunning ? 'inline-flex' : 'none';
    document.getElementById('btnStopMonitor').style.display = isRunning ? 'inline-flex' : 'none';

    // Monitor Screen Updates
    document.getElementById('monitorPill').textContent = `Status: ${state.status_text}`;
    document.getElementById('liveModel').textContent = state.current_model || '--';
    document.getElementById('liveStrategy').textContent = state.current_strategy || '--';
    document.getElementById('liveRound').textContent = state.total_rounds > 0 ? `${state.current_round} / ${state.total_rounds}` : '--';

    const latestRound = state.live_rounds && state.live_rounds.length > 0 ? state.live_rounds[state.live_rounds.length - 1] : null;
    if (latestRound && latestRound.metrics) {
        const acc = latestRound.metrics.accuracy;
        const loss = latestRound.metrics.loss;
        document.getElementById('liveAcc').textContent = acc !== undefined ? `${acc.toFixed(2)}%` : '--';
        document.getElementById('liveLoss').textContent = loss !== undefined ? loss.toFixed(4) : '--';
    }

    // Progress Bar
    const percent = state.total_steps > 0 ? Math.min(100, Math.round((state.current_step / state.total_steps) * 100)) : (isRunning ? 20 : 0);
    document.getElementById('progressFill').style.width = `${percent}%`;
    document.getElementById('progressPercent').textContent = `${percent}%`;
    document.getElementById('progressPhaseText').textContent = state.status_text || 'Federated Simulation';

    // Elapsed
    const elapsedSec = state.elapsed_seconds || 0;
    const mins = Math.floor(elapsedSec / 60).toString().padStart(2, '0');
    const secs = (elapsedSec % 60).toString().padStart(2, '0');
    document.getElementById('liveElapsed').textContent = `${mins}:${secs}`;

    // Terminal Logs
    if (state.logs && state.logs.length > 0) {
        const terminal = document.getElementById('terminalLogs');
        terminal.innerHTML = state.logs.map(line => `<div class="terminal-line">${escapeHtml(line)}</div>`).join('');
        terminal.scrollTop = terminal.scrollHeight;
    }

    // Update Live Charts from live_rounds
    if (state.live_rounds && state.live_rounds.length > 0) {
        updateLiveCharts(state.live_rounds);
    }

    // If completed just now, refresh data
    if (!isRunning && state.latest_report && currentTab === 'overview') {
        loadDashboardData();
    }
}

function updateLiveCharts(liveRounds) {
    // Group series by "Model (Strategy)"
    const seriesMap = {};
    const roundLabelsSet = new Set();

    liveRounds.forEach(r => {
        const seriesKey = `${r.model} (${r.strategy})`;
        roundLabelsSet.add(r.round);
        if (!seriesMap[seriesKey]) {
            seriesMap[seriesKey] = { accs: [], losses: [] };
        }
        seriesMap[seriesKey].accs.push(r.metrics.accuracy || 0);
        seriesMap[seriesKey].losses.push(r.metrics.loss || 0);
    });

    const rounds = Array.from(roundLabelsSet).sort((a, b) => a - b).map(r => `Round ${r}`);
    const colorPalette = ['#06b6d4', '#10b981', '#8b5cf6', '#f59e0b', '#f43f5e', '#3b82f6'];

    // Update Accuracy Chart
    accuracyChart.data.labels = rounds;
    accuracyChart.data.datasets = Object.keys(seriesMap).map((key, i) => ({
        label: key,
        data: seriesMap[key].accs,
        borderColor: colorPalette[i % colorPalette.length],
        backgroundColor: colorPalette[i % colorPalette.length] + '33',
        tension: 0.25,
        fill: false,
        borderWidth: 2
    }));
    accuracyChart.update('none');

    // Update Loss Chart
    lossChart.data.labels = rounds;
    lossChart.data.datasets = Object.keys(seriesMap).map((key, i) => ({
        label: key,
        data: seriesMap[key].losses,
        borderColor: colorPalette[i % colorPalette.length],
        backgroundColor: colorPalette[i % colorPalette.length] + '33',
        tension: 0.25,
        fill: false,
        borderWidth: 2
    }));
    lossChart.update('none');
}

// Simulation Actions
async function handleStartSimulation(e) {
    e.preventDefault();

    const selectedModels = Array.from(document.querySelectorAll('input[name="cfgModels"]:checked')).map(el => el.value);
    const selectedStrategies = Array.from(document.querySelectorAll('input[name="cfgStrategies"]:checked')).map(el => el.value);

    if (selectedModels.length === 0 || selectedStrategies.length === 0) {
        alert('Please select at least one model and one aggregation strategy.');
        return;
    }

    const payload = {
        username: document.getElementById('cfgUsername').value.trim() || 'Manoj',
        dataset: document.getElementById('cfgDataset').value,
        target_col: document.getElementById('cfgTarget').value.trim(),
        task_type: document.getElementById('cfgTask').value,
        distribution: document.getElementById('cfgDistribution').value,
        num_clients: parseInt(document.getElementById('cfgClients').value, 10),
        num_rounds: parseInt(document.getElementById('cfgRounds').value, 10),
        batch_size: parseInt(document.getElementById('cfgBatchSize').value, 10),
        lr: parseFloat(document.getElementById('cfgLr').value),
        models: selectedModels,
        strategies: selectedStrategies
    };

    try {
        const res = await fetch('/api/simulation/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await res.json();
        if (res.ok) {
            switchTab('monitor');
            fetchStatus();
        } else {
            alert(data.error || 'Failed to start simulation.');
        }
    } catch (err) {
        alert('Error communicating with backend server.');
        console.error(err);
    }
}

async function handleStopSimulation() {
    if (!confirm('Are you sure you want to stop the ongoing federated simulation?')) return;
    try {
        const res = await fetch('/api/simulation/stop', { method: 'POST' });
        const data = await res.json();
        fetchStatus();
    } catch (err) {
        console.error('Stop error:', err);
    }
}

// Data Fetching
async function loadDashboardData() {
    try {
        const [simsRes, datasetsRes] = await Promise.all([
            fetch('/api/simulations'),
            fetch('/api/datasets')
        ]);

        if (simsRes.ok) {
            const data = await simsRes.json();
            allSimulations = data.simulations || [];
            renderRecentSimulations(allSimulations);
        }

        if (datasetsRes.ok) {
            const data = await datasetsRes.json();
            renderDatasets(data.datasets || []);
        }
    } catch (err) {
        console.error('Dashboard load error:', err);
    }
}

function renderRecentSimulations(simulations) {
    const tbody = document.querySelector('#recentSimulationsTable tbody');
    if (!simulations || simulations.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No simulation runs found yet. Run an experiment in Studio!</td></tr>';
        return;
    }

    let topAcc = 0;

    tbody.innerHTML = simulations.slice(0, 5).map(s => {
        const meta = s.metadata || {};
        const exps = s.experiments || {};

        const fedavgAcc = exps['DeepMLP_FedAvg']?.final_acc || exps['LogisticRegression_FedAvg']?.final_acc || null;
        const fedproxAcc = exps['DeepMLP_FedProx']?.final_acc || null;
        const fedadamAcc = exps['DeepMLP_FedAdam']?.final_acc || null;

        [fedavgAcc, fedproxAcc, fedadamAcc].forEach(acc => {
            if (acc && acc > topAcc) topAcc = acc;
        });

        return `
            <tr>
                <td><strong>${escapeHtml(s.modified_time || s.filename)}</strong></td>
                <td><span class="chip chip-cyan">${escapeHtml(meta.username || 'user')}</span></td>
                <td>${meta.num_rounds || 5} rounds (${meta.num_clients || 5} clients)</td>
                <td class="text-emerald font-semibold">${fedavgAcc ? `${fedavgAcc.toFixed(2)}%` : '--'}</td>
                <td class="text-cyan font-semibold">${fedproxAcc ? `${fedproxAcc.toFixed(2)}%` : '--'}</td>
                <td class="text-muted font-semibold">${fedadamAcc ? `${fedadamAcc.toFixed(2)}%` : '--'}</td>
            </tr>
        `;
    }).join('');

    if (topAcc > 0) {
        document.getElementById('statTopAcc').textContent = `${topAcc.toFixed(2)}%`;
    }
}

function renderDatasets(datasets) {
    const select = document.getElementById('cfgDataset');
    if (!datasets || datasets.length === 0) return;

    select.innerHTML = datasets.map(d => `<option value="${escapeHtml(d.filename)}">${escapeHtml(d.filename)} (${d.size_mb} MB)</option>`).join('');

    const first = datasets[0];
    if (first && first.total_rows) {
        document.getElementById('statSamples').textContent = `${first.total_rows.toLocaleString()} Records`;
    }
}

// Reports & Analytics Tab
async function loadReportsList() {
    try {
        const res = await fetch('/api/reports');
        if (!res.ok) return;
        const data = await res.json();
        allReports = data.reports || [];

        const select = document.getElementById('selectReportHistory');
        select.innerHTML = '<option value="">Select a previous simulation report...</option>' +
            allReports.map(r => `<option value="${escapeHtml(r.filename)}">${escapeHtml(r.content_name)} (${r.modified})</option>`).join('');

        if (allReports.length > 0 && !select.value) {
            select.selectedIndex = 1;
            loadReportDetail(allReports[0].filename);
        }
    } catch (err) {
        console.error('Failed to load reports:', err);
    }
}

async function loadReportDetail(filename) {
    try {
        const report = allReports.find(r => r.filename === filename);
        const res = await fetch(`/api/report/view?file=${encodeURIComponent(filename)}`);
        if (!res.ok) return;
        const data = await res.json();

        document.getElementById('reportMarkdownBody').textContent = data.content || 'Report empty.';
        document.getElementById('reportTitle').textContent = `Analytical Summary: ${filename}`;

        if (report) {
            if (report.model_plot) {
                document.getElementById('reportModelPlot').src = `${report.model_plot}?t=${Date.now()}`;
            }
            if (report.strategy_plot) {
                document.getElementById('reportStratPlot').src = `${report.strategy_plot}?t=${Date.now()}`;
            }
        }
    } catch (err) {
        console.error('Failed to load report detail:', err);
    }
}

function downloadCurrentReport() {
    const body = document.getElementById('reportMarkdownBody').textContent;
    const blob = new Blob([body], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'Federated_Learning_Clinical_Report.md';
    a.click();
    URL.revokeObjectURL(url);
}

// Dataset Explorer Tab
async function loadDatasetInfo() {
    try {
        const res = await fetch('/api/datasets');
        if (!res.ok) return;
        const data = await res.json();
        if (!data.datasets || data.datasets.length === 0) return;

        const ds = data.datasets[0];
        document.getElementById('datasetTitle').textContent = `Dataset: ${ds.filename}`;
        document.getElementById('datasetRowsChip').textContent = `${(ds.total_rows || 'N/A').toLocaleString()} Rows | ${ds.size_mb} MB`;

        const strip = document.getElementById('datasetStatsStrip');
        strip.innerHTML = `
            <div><strong>Total Records:</strong> ${ds.total_rows}</div>
            <div><strong>Features:</strong> ${ds.columns ? ds.columns.length : 0}</div>
            <div><strong>Storage:</strong> ${ds.size_mb} MB</div>
        `;

        // Render Features
        const featContainer = document.getElementById('featureTags');
        featContainer.innerHTML = (ds.columns || []).map(col => `
            <div class="feature-tag">
                <span>${escapeHtml(col)}</span>
                <span class="type-badge">${col === 'Predicted Disease' ? 'Target' : 'Feature'}</span>
            </div>
        `).join('');

        // Render Disease Chart
        if (ds.class_distribution && Object.keys(ds.class_distribution).length > 0) {
            diseaseChart.data.labels = Object.keys(ds.class_distribution);
            diseaseChart.data.datasets[0].data = Object.values(ds.class_distribution);
            diseaseChart.update();
        }

        // Render Preview Table
        if (ds.preview && ds.preview.length > 0) {
            const table = document.getElementById('datasetPreviewTable');
            const cols = Object.keys(ds.preview[0]);
            table.querySelector('thead').innerHTML = `<tr>${cols.map(c => `<th>${escapeHtml(c)}</th>`).join('')}</tr>`;
            table.querySelector('tbody').innerHTML = ds.preview.map(row => `
                <tr>${cols.map(c => `<td>${escapeHtml(String(row[c]))}</td>`).join('')}</tr>
            `).join('');
        }
    } catch (err) {
        console.error('Dataset info load error:', err);
    }
}

// Helpers
function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

// Global State
let currentThreshold = 0.50;
let lastPredictionData = null;

const presets = {
    healthy: {
        male: 0, age: 35, education: 3, currentSmoker: 0, cigsPerDay: 0,
        BPMeds: 0, prevalentStroke: 0, prevalentHyp: 0, diabetes: 0,
        sysBP: 115, diaBP: 75, totChol: 180, BMI: 22.5, heartRate: 68, glucose: 75
    },
    moderate: {
        male: 1, age: 52, education: 2, currentSmoker: 1, cigsPerDay: 15,
        BPMeds: 0, prevalentStroke: 0, prevalentHyp: 1, diabetes: 0,
        sysBP: 140, diaBP: 90, totChol: 240, BMI: 28.0, heartRate: 78, glucose: 92
    },
    high: {
        male: 1, age: 65, education: 1, currentSmoker: 1, cigsPerDay: 30,
        BPMeds: 1, prevalentStroke: 0, prevalentHyp: 1, diabetes: 1,
        sysBP: 175, diaBP: 105, totChol: 290, BMI: 33.2, heartRate: 88, glucose: 160
    }
};

document.addEventListener('DOMContentLoaded', () => {
    fetchDatasetInfo();
    triggerPredict();
    window.addEventListener('resize', drawSigmoidChart);
});

// Fetch Dataset Statistics & Model Performance Metrics
async function fetchDatasetInfo() {
    try {
        const response = await fetch('/api/dataset_info');
        const data = await response.json();
        
        if (data.metrics) {
            // Standard
            const std = data.metrics.standard;
            document.getElementById('stdAcc').innerText = (std.accuracy * 100).toFixed(1) + '%';
            document.getElementById('stdPrec').innerText = (std.precision * 100).toFixed(1) + '%';
            document.getElementById('stdRecall').innerText = (std.recall * 100).toFixed(1) + '%';
            document.getElementById('stdAuc').innerText = std.roc_auc.toFixed(3);

            // Balanced
            const bal = data.metrics.balanced;
            document.getElementById('balAcc').innerText = (bal.accuracy * 100).toFixed(1) + '%';
            document.getElementById('balPrec').innerText = (bal.precision * 100).toFixed(1) + '%';
            document.getElementById('balRecall').innerText = (bal.recall * 100).toFixed(1) + '%';
            document.getElementById('balAuc').innerText = bal.roc_auc.toFixed(3);
        }
    } catch (err) {
        console.error('Error fetching dataset info:', err);
    }
}

// Toggle Smoking input dependent state
function toggleSmoking() {
    const isSmoker = document.getElementById('input-currentSmoker').value === '1';
    const cigsInput = document.getElementById('input-cigsPerDay');
    if (!isSmoker) {
        cigsInput.value = 0;
        cigsInput.disabled = true;
    } else {
        cigsInput.disabled = false;
        if (cigsInput.value == 0) cigsInput.value = 15;
    }
    triggerPredict();
}

// Load Presets
function loadPreset(key) {
    const p = presets[key];
    if (!p) return;

    for (const [col, val] of Object.entries(p)) {
        const el = document.getElementById(`input-${col}`);
        if (el) {
            el.value = val;
        }
    }
    toggleSmoking();
    triggerPredict();
}

// Update Threshold Slider
function updateThreshold(val) {
    currentThreshold = parseFloat(val);
    document.getElementById('thresholdNum').innerText = currentThreshold.toFixed(2);
    document.getElementById('thresholdDisplayVal').innerText = currentThreshold.toFixed(2);
    triggerPredict();
}

// Main Prediction Fetch & Update
let predictDebounceTimer = null;
function triggerPredict() {
    clearTimeout(predictDebounceTimer);
    predictDebounceTimer = setTimeout(runPrediction, 120);
}

async function runPrediction() {
    const featureIds = [
        'male', 'age', 'education', 'currentSmoker', 'cigsPerDay',
        'BPMeds', 'prevalentStroke', 'prevalentHyp', 'diabetes',
        'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
    ];

    const inputs = {};
    featureIds.forEach(id => {
        const el = document.getElementById(`input-${id}`);
        if (el) inputs[id] = parseFloat(el.value) || 0;
    });

    const modelType = document.querySelector('input[name="model_type"]:checked').value;

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                inputs: inputs,
                model_type: modelType,
                threshold: currentThreshold
            })
        });

        const res = await response.json();
        if (res.success) {
            lastPredictionData = res;
            updateUI(res);
        }
    } catch (err) {
        console.error('Prediction API Error:', err);
    }
}

// Update UI elements
function updateUI(res) {
    // Strategy Badge
    document.getElementById('strategyBadge').innerText = res.model_type === 'balanced' ? 'Balanced Model' : 'Standard Model';
    document.getElementById('strategyBadge').className = res.model_type === 'balanced' ? 'badge tag-success' : 'badge';

    // Probability & Gauge
    const prob = res.probability_percentage;
    document.getElementById('probPercent').innerText = prob.toFixed(1) + '%';
    document.getElementById('zScoreVal').innerText = res.z_score >= 0 ? `+${res.z_score}` : `${res.z_score}`;
    document.getElementById('interceptVal').innerText = res.intercept;

    const gaugeCircle = document.getElementById('gaugeCircle');
    const predText = document.getElementById('predictionText');

    if (res.prediction === 1) {
        gaugeCircle.className = 'gauge-circle risk-high';
        predText.innerText = '🚨 High Risk of Heart Disease';
        predText.style.color = 'var(--danger-red)';
    } else {
        gaugeCircle.className = 'gauge-circle';
        predText.innerText = '✅ Low Risk of Heart Disease';
        predText.style.color = 'var(--success-green)';
    }

    // Render Contributions Table
    renderContributionsTable(res.contributions);

    // Draw Sigmoid Canvas
    drawSigmoidChart();

    // Re-render MathJax if present
    if (window.MathJax && MathJax.typesetPromise) {
        MathJax.typesetPromise();
    }
}

// Render Table
function renderContributionsTable(contributions) {
    const tbody = document.getElementById('contributionsTbody');
    tbody.innerHTML = '';

    contributions.forEach(c => {
        const tr = document.createElement('tr');
        const isPos = c.contribution >= 0;
        
        tr.innerHTML = `
            <td><strong>${c.display_name}</strong></td>
            <td class="num-cell">${c.raw_value}</td>
            <td class="num-cell">${c.mean}</td>
            <td class="num-cell">${c.scaled_value}</td>
            <td class="num-cell">${c.coefficient}</td>
            <td class="num-cell" style="color: ${isPos ? 'var(--danger-red)' : 'var(--success-green)'}">
                ${isPos ? '+' : ''}${c.contribution}
            </td>
            <td>
                <span class="badge-impact ${isPos ? 'badge-increase' : 'badge-decrease'}">
                    ${isPos ? '▲ Increases Risk' : '▼ Decreases Risk'}
                </span>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

// Canvas Sigmoid Chart
function drawSigmoidChart() {
    const canvas = document.getElementById('sigmoidCanvas');
    if (!canvas || !lastPredictionData) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    // Clear background
    ctx.clearRect(0, 0, width, height);

    const padLeft = 45;
    const padRight = 20;
    const padTop = 20;
    const padBottom = 35;

    const plotW = width - padLeft - padRight;
    const plotH = height - padTop - padBottom;

    // Range Z: -6 to +6
    const minZ = -6;
    const maxZ = 6;

    function zToX(z) {
        return padLeft + ((z - minZ) / (maxZ - MinZ_diff)) * plotW;
    }
    const MinZ_diff = maxZ - minZ;

    function probToY(p) {
        return padTop + (1 - p) * plotH;
    }

    // Grid lines & Axes
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;

    // Y Grid lines (0, 0.25, 0.5, 0.75, 1)
    [0, 0.25, 0.5, 0.75, 1].forEach(p => {
        const y = probToY(p);
        ctx.beginPath();
        ctx.moveTo(padLeft, y);
        ctx.lineTo(width - padRight, y);
        ctx.stroke();

        ctx.fillStyle = '#64748b';
        ctx.font = '10px "JetBrains Mono"';
        ctx.textAlign = 'right';
        ctx.fillText(p.toFixed(2), padLeft - 6, y + 3);
    });

    // Z Grid lines (-6, -3, 0, 3, 6)
    [-6, -3, 0, 3, 6].forEach(z => {
        const x = zToX(z);
        ctx.beginPath();
        ctx.moveTo(x, padTop);
        ctx.lineTo(x, height - padBottom);
        ctx.stroke();

        ctx.fillStyle = '#64748b';
        ctx.font = '10px "JetBrains Mono"';
        ctx.textAlign = 'center';
        ctx.fillText(`z=${z}`, x, height - padBottom + 16);
    });

    // Draw Threshold Line
    const threshY = probToY(currentThreshold);
    ctx.strokeStyle = 'rgba(245, 158, 11, 0.6)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(padLeft, threshY);
    ctx.lineTo(width - padRight, threshY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#f59e0b';
    ctx.font = '10px "Outfit"';
    ctx.textAlign = 'right';
    ctx.fillText(`Cutoff (${currentThreshold.toFixed(2)})`, width - padRight - 5, threshY - 5);

    // Draw Sigmoid Curve S(z) = 1 / (1 + e^-z)
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 3;
    ctx.beginPath();

    for (let px = 0; px <= plotW; px += 2) {
        const z = minZ + (px / plotW) * MinZ_diff;
        const p = 1 / (1 + Math.exp(-z));
        const x = padLeft + px;
        const y = probToY(p);

        if (px === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw Current Patient Point
    const pZ = Math.max(-5.9, Math.min(5.9, lastPredictionData.z_score));
    const pProb = lastPredictionData.probability;
    const ptX = zToX(pZ);
    const ptY = probToY(pProb);

    // Point Glow
    const isRisk = pProb >= currentThreshold;
    const ptColor = isRisk ? '#f43f5e' : '#10b981';

    ctx.shadowColor = ptColor;
    ctx.shadowBlur = 12;

    ctx.fillStyle = ptColor;
    ctx.beginPath();
    ctx.arc(ptX, ptY, 7, 0, Math.PI * 2);
    ctx.fill();

    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.stroke();

    ctx.shadowBlur = 0; // reset glow

    // Point Coordinates Text
    ctx.fillStyle = '#f1f5f9';
    ctx.font = 'bold 11px "JetBrains Mono"';
    ctx.textAlign = ptX > width - 100 ? 'right' : 'left';
    ctx.fillText(`Patient (z=${pZ.toFixed(2)}, p=${(pProb * 100).toFixed(1)}%)`, ptX + (ptX > width - 100 ? -12 : 12), ptY - 8);
}

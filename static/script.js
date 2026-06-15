document.addEventListener('DOMContentLoaded', () => {
    // ─── Constants ───────────────────────────────────────────────────
    const CANVAS_SIZE = 280;
    const GRID_SIZE   = 8;
    const BLOCK_SIZE  = CANVAS_SIZE / GRID_SIZE; // 35px

    // ─── DOM References ──────────────────────────────────────────────
    const canvas          = document.getElementById('drawing-canvas');
    const ctx             = canvas.getContext('2d');
    const downscaledCvs   = document.getElementById('downscaled-canvas');
    const dsCtx           = downscaledCvs.getContext('2d');
    const inverseCvs      = document.getElementById('inverse-canvas');
    const invCtx          = inverseCvs.getContext('2d');
    const canvasHint      = document.getElementById('canvas-hint');
    const canvasWrap      = canvas.closest('.canvas-wrap');

    const predDigitEl     = document.getElementById('pred-digit');
    const predStatusEl    = document.getElementById('pred-status');
    const predConfEl      = document.getElementById('pred-conf');
    const clearBtn        = document.getElementById('clear-btn');
    const predictBtn      = document.getElementById('predict-btn');
    const surjPill        = document.getElementById('surj-pill');

    // ─── State ───────────────────────────────────────────────────────
    let isDrawing = false;
    let gridData  = new Float32Array(GRID_SIZE * GRID_SIZE); // 64 features in [0,1]
    let hasDrawn  = false;

    // Chart instances — declared BEFORE any function that references them
    // (fixes the TDZ ReferenceError from before)
    let probChart  = null;
    let embedChart = null;
    let meanImages = {};

    // ─── Canvas Setup ────────────────────────────────────────────────
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth   = 22;
    ctx.lineCap     = 'round';
    ctx.lineJoin    = 'round';

    // Fill canvas white
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // ─── Drawing Event Listeners ─────────────────────────────────────
    canvas.addEventListener('mousedown',  onMouseDown);
    canvas.addEventListener('mousemove',  onMouseMove);
    canvas.addEventListener('mouseup',    onMouseUp);
    canvas.addEventListener('mouseleave', onMouseUp);

    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        const t = e.touches[0];
        simulateMouse('mousedown', canvas, t.clientX, t.clientY);
    }, { passive: false });

    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        const t = e.touches[0];
        simulateMouse('mousemove', canvas, t.clientX, t.clientY);
    }, { passive: false });

    canvas.addEventListener('touchend', (e) => {
        e.preventDefault();
        simulateMouse('mouseup', canvas, 0, 0);
    }, { passive: false });

    clearBtn.addEventListener('click',   clearCanvas);
    predictBtn.addEventListener('click', evaluateFunction);

    // ─── Drawing Functions ───────────────────────────────────────────
    function simulateMouse(type, target, clientX, clientY) {
        target.dispatchEvent(new MouseEvent(type, { clientX, clientY, bubbles: true }));
    }

    function getPos(evt) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width  / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (evt.clientX - rect.left) * scaleX,
            y: (evt.clientY - rect.top)  * scaleY
        };
    }

    function onMouseDown(e) {
        isDrawing = true;
        ctx.beginPath();
        const pos = getPos(e);
        ctx.moveTo(pos.x, pos.y);
        // Hide hint on first draw
        if (!hasDrawn) {
            hasDrawn = true;
            canvasHint.classList.add('hidden');
        }
        canvasWrap.classList.add('drawing');
    }

    function onMouseMove(e) {
        if (!isDrawing) return;
        const pos = getPos(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
        updateDownscaledMatrix();
    }

    function onMouseUp() {
        if (!isDrawing) return;
        isDrawing = false;
        ctx.beginPath(); // reset path to avoid stray lines
        canvasWrap.classList.remove('drawing');
        updateDownscaledMatrix();
    }

    function clearCanvas() {
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
        gridData.fill(0);
        hasDrawn = false;
        canvasHint.classList.remove('hidden');
        canvasWrap.classList.remove('drawing');

        // Reset result UI
        predDigitEl.textContent = '?';
        predDigitEl.classList.remove('updated');
        predStatusEl.textContent = 'Draw a digit and evaluate';
        predConfEl.textContent = '';

        // Reset preview canvases
        dsCtx.fillStyle = '#f8fafc';
        dsCtx.fillRect(0, 0, downscaledCvs.width, downscaledCvs.height);
        invCtx.fillStyle = '#f8fafc';
        invCtx.fillRect(0, 0, inverseCvs.width, inverseCvs.height);

        // Reset network node activations
        resetNetworkVisuals();

        // Clear the user's star point from the scatter plot (safe check)
        if (embedChart && embedChart.data.datasets.length > 10) {
            embedChart.data.datasets.splice(10); // remove all beyond the 10 digit datasets
            embedChart.update('none');
        }

        // Reset probability chart
        if (probChart) {
            probChart.data.datasets[0].data = new Array(10).fill(0);
            probChart.update('none');
        }
    }

    // ─── Downsampling ────────────────────────────────────────────────
    function updateDownscaledMatrix() {
        const imgData = ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE);
        const px = imgData.data;
        const newGrid = new Float32Array(GRID_SIZE * GRID_SIZE);

        for (let row = 0; row < GRID_SIZE; row++) {
            for (let col = 0; col < GRID_SIZE; col++) {
                const startX = col * BLOCK_SIZE;
                const startY = row * BLOCK_SIZE;
                let sum = 0, count = 0;

                for (let y = startY; y < startY + BLOCK_SIZE; y++) {
                    for (let x = startX; x < startX + BLOCK_SIZE; x++) {
                        const i = (Math.floor(y) * CANVAS_SIZE + Math.floor(x)) * 4;
                        // White=255 → ink=0. Black=0 → ink=255
                        const ink = 255 - (px[i] * 0.299 + px[i+1] * 0.587 + px[i+2] * 0.114);
                        sum += ink;
                        count++;
                    }
                }
                // normalise to [0, 1], boost slightly
                newGrid[row * GRID_SIZE + col] = Math.min((sum / (count * 255)) * 1.4, 1.0);
            }
        }

        gridData = newGrid;
        drawPreviewCanvas(dsCtx, downscaledCvs, newGrid);
        updateInputLayerVisuals(newGrid);
    }

    function drawPreviewCanvas(context, cvs, grid) {
        context.clearRect(0, 0, cvs.width, cvs.height);
        const scale = cvs.width / GRID_SIZE;
        for (let r = 0; r < GRID_SIZE; r++) {
            for (let c = 0; c < GRID_SIZE; c++) {
                const val  = grid[r * GRID_SIZE + c];
                const gray = Math.floor(255 - val * 255);
                context.fillStyle = `rgb(${gray},${gray},${gray})`;
                context.fillRect(c * scale, r * scale, scale, scale);
            }
        }
    }

    // ─── Network Setup ───────────────────────────────────────────────
    function setupNetworkDOM() {
        // Input: 64 nodes in a 16x4 grid
        const inputContainer = document.querySelector('.input-nodes');
        inputContainer.innerHTML = '';
        for (let i = 0; i < 64; i++) {
            const n = document.createElement('div');
            n.className = 'node';
            inputContainer.appendChild(n);
        }

        // H1: 32 circular nodes
        const h1Container = document.querySelector('.h1-nodes');
        h1Container.innerHTML = '';
        for (let i = 0; i < 32; i++) {
            const n = document.createElement('div');
            n.className = 'node';
            h1Container.appendChild(n);
        }

        // H2: 16 circular nodes
        const h2Container = document.querySelector('.h2-nodes');
        h2Container.innerHTML = '';
        for (let i = 0; i < 16; i++) {
            const n = document.createElement('div');
            n.className = 'node';
            h2Container.appendChild(n);
        }

        // Output: 10 labeled nodes
        const outContainer = document.querySelector('.output-nodes');
        outContainer.innerHTML = '';
        for (let i = 0; i < 10; i++) {
            const lbl = document.createElement('div');
            lbl.className = 'node-lbl';
            const n = document.createElement('div');
            n.className = 'node';
            const txt = document.createElement('span');
            txt.textContent = i;
            lbl.appendChild(n);
            lbl.appendChild(txt);
            outContainer.appendChild(lbl);
        }
    }

    function updateInputLayerVisuals(grid) {
        const nodes = document.querySelectorAll('.input-nodes .node');
        grid.forEach((val, idx) => {
            if (nodes[idx]) {
                const alpha = 0.05 + val * 0.9;
                nodes[idx].style.backgroundColor = `rgba(59,130,246,${alpha.toFixed(2)})`;
            }
        });
    }

    function animateNetwork(predictedDigit, probs) {
        resetNetworkVisuals();
        updateInputLayerVisuals(gridData);

        const h1Nodes = [...document.querySelectorAll('.h1-nodes .node')];
        const h2Nodes = [...document.querySelectorAll('.h2-nodes .node')];
        const outNodes = [...document.querySelectorAll('.output-nodes .node')];

        // Stagger H1 activation
        h1Nodes.forEach((node, i) => {
            setTimeout(() => {
                if (Math.random() > 0.35) {
                    node.style.backgroundColor = '#60a5fa';
                    node.style.transform = 'scale(1.25)';
                    node.classList.add('fire');
                }
            }, i * 8);
        });

        // H2 activation
        setTimeout(() => {
            h2Nodes.forEach((node, i) => {
                setTimeout(() => {
                    if (Math.random() > 0.25) {
                        node.style.backgroundColor = '#818cf8';
                        node.style.transform = 'scale(1.3)';
                    }
                }, i * 12);
            });
        }, 280);

        // Output
        setTimeout(() => {
            outNodes.forEach((node, idx) => {
                const p = probs[idx] || 0;
                if (idx === predictedDigit) {
                    node.style.backgroundColor = '#10b981';
                    node.style.transform = 'scale(1.6)';
                    node.style.boxShadow = '0 0 8px rgba(16,185,129,0.5)';
                } else {
                    const alpha = 0.1 + p * 3;
                    node.style.backgroundColor = `rgba(59,130,246,${Math.min(alpha,0.7).toFixed(2)})`;
                    node.style.transform = 'scale(1)';
                }
            });
        }, 550);
    }

    function resetNetworkVisuals() {
        document.querySelectorAll('.h1-nodes .node, .h2-nodes .node, .output-nodes .node')
            .forEach(n => {
                n.style.backgroundColor = '';
                n.style.transform = '';
                n.style.boxShadow = '';
            });
    }

    // ─── API Calls ───────────────────────────────────────────────────
    async function initWorkspace() {
        try {
            const [meanRes, embedRes] = await Promise.all([
                fetch('/inverse-data'),
                fetch('/embedding-data')
            ]);

            const meanData  = await meanRes.json();
            const embedData = await embedRes.json();

            if (meanData.success)  meanImages = meanData.mean_images;
            if (embedData.success) renderEmbeddingScatter(embedData.embeddings);

            renderSurjectiveChart(new Array(10).fill(0));

            if (surjPill) {
                surjPill.textContent = '✓ Confirmed';
                surjPill.style.background = '#d1fae5';
                surjPill.style.color = '#059669';
            }
        } catch (err) {
            console.error('Failed to load workspace data:', err);
        }
    }

    async function evaluateFunction() {
        if (!hasDrawn) {
            predStatusEl.textContent = 'Please draw a digit first!';
            return;
        }

        predictBtn.disabled = true;
        predictBtn.textContent = 'Computing…';
        predStatusEl.textContent = 'Running f(x) through the network…';

        try {
            const res  = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ features: Array.from(gridData) })
            });
            const data = await res.json();

            if (data.success) {
                const digit = data.prediction;
                const probs  = data.probabilities;
                const conf   = (probs[digit] * 100).toFixed(1);

                // ── Result display ──────────────────────────────────
                predDigitEl.textContent = digit;
                predDigitEl.classList.remove('updated');
                void predDigitEl.offsetWidth; // reflow to re-trigger animation
                predDigitEl.classList.add('updated');

                predStatusEl.textContent = `f(input) = ${digit}  ·  Mapping complete`;
                predConfEl.textContent   = `Confidence: ${conf}%`;

                // ── Animate network ──────────────────────────────────
                animateNetwork(digit, probs);

                // ── Prob chart ──────────────────────────────────────
                if (probChart) {
                    probChart.data.datasets[0].data = probs.map(p => (p * 100).toFixed(2));
                    probChart.data.datasets[0].backgroundColor = probs.map((_, i) =>
                        i === digit ? 'rgba(16,185,129,0.75)' : 'rgba(59,130,246,0.5)'
                    );
                    probChart.update();
                }

                // ── Scatter plot: show user's point ─────────────────
                if (embedChart) {
                    if (embedChart.data.datasets.length > 10) {
                        embedChart.data.datasets.splice(10);
                    }
                    embedChart.data.datasets.push({
                        label: `Your Input → ${digit}`,
                        data: [{ x: data.pca_projection[0], y: data.pca_projection[1] }],
                        backgroundColor: '#f43f5e',
                        borderColor: '#fff',
                        borderWidth: 2,
                        pointRadius: 11,
                        pointHoverRadius: 14,
                        pointStyle: 'star'
                    });
                    embedChart.update();
                }

                // ── Inverse canvas ──────────────────────────────────
                const key = String(digit);
                if (meanImages[key]) {
                    drawPreviewCanvas(invCtx, inverseCvs, meanImages[key]);
                }

                // ── Highlight active concept chip ────────────────────
                document.querySelectorAll('.concept-chip').forEach((chip, i) => {
                    chip.classList.toggle('active', i === 0);
                });
            } else {
                predStatusEl.textContent = `Error: ${data.error}`;
            }
        } catch (err) {
            predStatusEl.textContent = 'Connection error – is the server running?';
            console.error(err);
        } finally {
            predictBtn.disabled = false;
            predictBtn.innerHTML = `
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                Evaluate f(x)
            `;
        }
    }

    // ─── Charts ──────────────────────────────────────────────────────
    function renderSurjectiveChart(initProbs) {
        const el = document.getElementById('prob-chart');
        if (!el) return;

        probChart = new Chart(el.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['0','1','2','3','4','5','6','7','8','9'],
                datasets: [{
                    label: 'Probability (%)',
                    data: initProbs,
                    backgroundColor: 'rgba(59,130,246,0.55)',
                    borderColor: 'rgba(59,130,246,0.9)',
                    borderWidth: 1.5,
                    borderRadius: 5,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 400 },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        ticks: { font: { size: 9 }, color: '#94a3b8' },
                        grid: { color: 'rgba(0,0,0,0.04)' }
                    },
                    x: {
                        ticks: { font: { size: 9, weight: '700' }, color: '#475569' },
                        grid: { display: false }
                    }
                },
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: ctx => ` ${parseFloat(ctx.raw).toFixed(1)}%`
                        }
                    }
                }
            }
        });
    }

    function renderEmbeddingScatter(embeddings) {
        const el = document.getElementById('embed-chart');
        if (!el) return;

        // Group by class
        const classes = Array.from({ length: 10 }, () => []);
        embeddings.forEach(pt => classes[pt.label].push({ x: pt.x, y: pt.y }));

        const palette = [
            '#ef4444','#f97316','#eab308','#84cc16',
            '#10b981','#06b6d4','#3b82f6','#6366f1',
            '#8b5cf6','#d946ef'
        ];

        const datasets = classes.map((pts, idx) => ({
            label: `${idx}`,
            data: pts,
            backgroundColor: palette[idx],
            pointRadius: 4,
            pointHoverRadius: 6,
            pointStyle: 'circle'
        }));

        embedChart = new Chart(el.getContext('2d'), {
            type: 'scatter',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 600 },
                scales: {
                    x: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { display: false } },
                    y: { grid: { color: 'rgba(0,0,0,0.03)' }, ticks: { display: false } }
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { boxWidth: 8, font: { size: 9 }, padding: 10, usePointStyle: true }
                    },
                    tooltip: {
                        callbacks: {
                            label: ctx => `Digit ${ctx.dataset.label}`
                        }
                    }
                }
            }
        });
    }

    // ─── Init ─────────────────────────────────────────────────────────
    setupNetworkDOM();
    initWorkspace();
    // Fill initial downscaled canvas
    dsCtx.fillStyle = '#f8fafc';
    dsCtx.fillRect(0, 0, downscaledCvs.width, downscaledCvs.height);
    invCtx.fillStyle = '#f8fafc';
    invCtx.fillRect(0, 0, inverseCvs.width, inverseCvs.height);
});

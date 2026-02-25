import { state, IMG_SIZE, CATEGORIES, EMOJIS, MAX_EPOCHS } from './config.js';

export function switchStage(stageNum) {
    state.currentStage = stageNum;
    document.querySelectorAll('.stage-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.nav-item').forEach(el => el.classList.remove('active'));

    document.getElementById(`stage${stageNum}`).classList.add('active');
    document.getElementById(`nav-s${stageNum}`).classList.add('active');
}

export function goToStage2() {
    document.getElementById('nav-s1').classList.add('completed');
    switchStage(2);
}

export function goToStage3() {
    switchStage(3);
}

export function startStage1Previews() {
    if (state.currentStage !== 1) return;

    CATEGORIES.forEach(cat => {
        const items = state.trainData.filter(d => d.c === cat);
        if (items.length === 0) return;

        let canvasId = "";
        if (cat === "apple") canvasId = "cvsApple";
        else if (cat === "cat") canvasId = "cvsCat";
        else if (cat === "clock") canvasId = "cvsClock";

        function runLoop() {
            if (state.currentStage !== 1) return; // Stop if moved to stage 2
            const sample = items[Math.floor(Math.random() * items.length)];
            let frames = 0;
            const totalFrames = 45; // Slower animation

            // Clear canvas before starting new animation
            drawDoodleOnCanvas(sample.d, canvasId, 0);

            function anim() {
                if (state.currentStage !== 1) return;
                frames++;
                drawDoodleOnCanvas(sample.d, canvasId, frames / totalFrames);
                if (frames < totalFrames) {
                    requestAnimationFrame(anim);
                } else {
                    setTimeout(runLoop, 600);
                }
            }
            requestAnimationFrame(anim);
        }

        setTimeout(runLoop, Math.random() * 500);
    });
}

export function drawDoodleOnCanvas(strokes, canvasId, progress) {
    const cvs = document.getElementById(canvasId);
    if (!cvs) return;
    const ctx = cvs.getContext('2d');

    if (progress <= 0) {
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(0, 0, cvs.width, cvs.height);
        return; // don't draw anything
    }

    let minX = 9999, minY = 9999, maxX = 0, maxY = 0;
    strokes.forEach(s => {
        s[0].forEach(x => { minX = Math.min(minX, x); maxX = Math.max(maxX, x); });
        s[1].forEach(y => { minY = Math.min(minY, y); maxY = Math.max(maxY, y); });
    });

    const rawW = maxX - minX || 1;
    const rawH = maxY - minY || 1;
    const size = Math.max(rawW, rawH);
    const scale = (cvs.width * 0.8) / size;
    const offsetX = (cvs.width - rawW * scale) / 2 - minX * scale;
    const offsetY = (cvs.height - rawH * scale) / 2 - minY * scale;

    ctx.strokeStyle = '#000';
    ctx.lineWidth = 14;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    const points = [];
    strokes.forEach(s => {
        points.push({ x: s[0][0], y: s[1][0], isStart: true });
        for (let i = 1; i < s[0].length; i++) {
            points.push({ x: s[0][i], y: s[1][i], isStart: false });
        }
    });

    const numPointsToDraw = Math.max(1, Math.floor(points.length * progress));

    ctx.beginPath();
    for (let i = 0; i < numPointsToDraw; i++) {
        const p = points[i];
        const px = p.x * scale + offsetX;
        const py = p.y * scale + offsetY;

        if (p.isStart) {
            if (i > 0) ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(px, py);
        } else {
            ctx.lineTo(px, py);
        }
    }
    ctx.stroke();
}

export function updateLiveFeed() {
    const feed = document.getElementById('liveFeed');
    feed.innerHTML = '';

    // Pick 3 random test samples
    const idxs = [
        Math.floor(Math.random() * state.testData.length),
        Math.floor(Math.random() * state.testData.length),
        Math.floor(Math.random() * state.testData.length)
    ];

    // Extract single images
    const xsToPredict = [];
    for (let i of idxs) {
        xsToPredict.push(state.tensors.testXs.slice([i, 0, 0, 0], [1, IMG_SIZE, IMG_SIZE, 1]));
    }
    const batchImages = tf.concat(xsToPredict);

    tf.tidy(() => {
        const preds = state.model.predict(batchImages);
        const scores = preds.arraySync();

        for (let i = 0; i < 3; i++) {
            const thumb = state.testThumbnails[idxs[i]];
            const sampleScores = scores[i];

            let barsHtml = '';
            CATEGORIES.forEach((cat, cIdx) => {
                const prob = Math.round(sampleScores[cIdx] * 100);
                const isMax = prob === Math.max(...sampleScores.map(v => Math.round(v * 100)));
                barsHtml += `
                    <div class="prob-bar-container">
                        <span style="width:20px;">${EMOJIS[cat]}</span>
                        <div class="prob-bar">
                            <div class="prob-fill" style="width: ${prob}%; background: ${isMax ? 'var(--accent-green)' : 'var(--accent-purple)'}"></div>
                        </div>
                    </div>
                `;
            });

            feed.innerHTML += `
                <div class="feed-item">
                    <div class="feed-img"><img src="${thumb}"></div>
                    <div class="feed-bars">${barsHtml}</div>
                </div>
            `;
        }
    });

    batchImages.dispose();
    xsToPredict.forEach(t => t.dispose());
}

// Chart Initializer
export function initChart() {
    const ctxChart = document.getElementById('trainingChart').getContext('2d');
    state.myChart = new Chart(ctxChart, {
        type: 'line',
        data: state.chartData,
        options: {
            responsive: true, maintainAspectRatio: false,
            animation: { duration: 0 },
            plugins: { legend: { display: true, labels: { color: '#e2e8f0' } } },
            elements: { point: { radius: 0 }, line: { borderWidth: 3, tension: 0.3 } },
            scales: {
                x: {
                    display: true,
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#64748b' }
                },
                y: {
                    type: 'linear', display: true, position: 'left',
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#fb923c', font: { family: 'JetBrains Mono' } },
                    title: { display: true, text: 'Loss', color: '#64748b' }
                },
                y1: {
                    type: 'linear', display: true, position: 'right', min: 0, max: 1,
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#34d399', font: { family: 'JetBrains Mono' }, callback: v => (v * 100).toFixed(0) + '%' },
                    title: { display: true, text: 'Accuracy', color: '#64748b' }
                }
            }
        }
    });
}

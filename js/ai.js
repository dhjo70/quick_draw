import { state, CATEGORIES, IMG_SIZE, MAX_EPOCHS } from './config.js?v=6';
import { updateLiveFeed } from './ui.js?v=6';
import { prepareAllTensors } from './data.js?v=6';

export function buildModel() {
    state.model = tf.sequential();
    state.model.add(tf.layers.conv2d({
        inputShape: [IMG_SIZE, IMG_SIZE, 1],
        kernelSize: 3, filters: 16, activation: 'relu'
    }));
    state.model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
    state.model.add(tf.layers.conv2d({ kernelSize: 3, filters: 32, activation: 'relu' }));
    state.model.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));
    state.model.add(tf.layers.flatten());
    state.model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
    state.model.add(tf.layers.dense({ units: CATEGORIES.length, activation: 'softmax' }));

    state.model.compile({
        optimizer: tf.train.adam(0.002),
        loss: 'sparseCategoricalCrossentropy',
        metrics: ['accuracy']
    });
}

export async function startTraining() {
    if (state.isTraining) return;

    const btn = document.getElementById('btnTrain');
    btn.disabled = true;
    btn.innerText = "⏳ Preparing Hardware...";

    // On first click, generate tensors
    if (!state.tensors.trainXs) {
        await prepareAllTensors();
    }

    state.isTraining = true;
    btn.innerText = "🚨 Training in Progress...";
    document.getElementById('gpuBadge').classList.replace('idle', 'active');

    await state.model.fit(state.tensors.trainXs, state.tensors.trainYs, {
        batchSize: 128,
        epochs: MAX_EPOCHS,
        validationData: [state.tensors.testXs, state.tensors.testYs],
        shuffle: true,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                // Update loss quickly
                document.getElementById('valLoss').innerText = logs.loss.toFixed(3);
                await tf.nextFrame();
            },
            onEpochEnd: async (epoch, logs) => {
                document.getElementById('valEpoch').innerHTML = `${epoch + 1}<span style="font-size:1rem;color:var(--text-muted)">/${MAX_EPOCHS}</span>`;
                document.getElementById('valAcc').innerText = (logs.val_acc * 100).toFixed(1) + '%';

                state.chartData.labels.push(epoch + 1);
                state.chartData.datasets[0].data.push(logs.loss);
                state.chartData.datasets[1].data.push(logs.val_acc);
                if (state.myChart) state.myChart.update();

                updateLiveFeed();
            }
        }
    });

    state.isTraining = false;
    btn.style.display = 'none';
    document.getElementById('gpuBadge').classList.replace('active', 'idle');
    document.getElementById('btnGoStage3').style.display = 'inline-block';
    document.getElementById('nav-s2').classList.add('completed');
}

export async function runBulkEvaluation() {
    const btn = document.getElementById('btnEval');
    btn.disabled = true;
    btn.innerText = "Running GPU Evaluation...";

    const grid = document.getElementById('evalGrid');
    grid.innerHTML = ''; // clear

    document.getElementById('gpuBadge').classList.replace('idle', 'active');
    await tf.nextFrame();

    // 1. Run full batch evaluation on test set (lightning fast on GPU)
    const predsTensor = state.model.predict(state.tensors.testXs);
    const predsArgMax = predsTensor.argMax(1);
    const predsArray = predsArgMax.dataSync();

    // 2. Compute accuracy sequentially
    const truthArray = state.tensors.testYs.dataSync();
    let correctCount = 0;
    let currentIdx = 0;

    // A delay that scales based on data length (~300 items = ~8ms per item, total ~2.5 seconds)
    const msDelay = Math.max(1, Math.floor(2500 / state.testData.length));

    function addNextResult() {
        if (currentIdx >= state.testData.length) {
            // Update Final Stats
            const accPct = (correctCount / state.testData.length * 100).toFixed(1);
            document.getElementById('evalCorrect').innerText = correctCount;
            document.getElementById('evalWrong').innerText = state.testData.length - correctCount;
            document.getElementById('evalAcc').innerText = accPct + "%";

            document.getElementById('finalResults').classList.add('show');
            document.getElementById('gpuBadge').classList.replace('active', 'idle');
            btn.innerText = "Evaluation Complete";
            document.getElementById('nav-s3').classList.add('completed');

            predsTensor.dispose();
            predsArgMax.dispose();
            return;
        }

        const predCatId = predsArray[currentIdx];
        const truCatId = truthArray[currentIdx];
        const isCorrect = (predCatId === truCatId);
        if (isCorrect) correctCount++;

        const div = document.createElement('div');
        div.className = `eval-item ${isCorrect ? 'correct' : 'wrong'}`;

        const icon = isCorrect ? '✔' : '✘';
        div.innerHTML = `
            <img src="${state.testThumbnails[currentIdx]}">
            <div class="eval-overlay">${icon}</div>
        `;
        grid.appendChild(div);
        grid.scrollTop = grid.scrollHeight; // Auto-scroll down

        currentIdx++;
        setTimeout(addNextResult, msDelay);
    }

    // Start the sequencer
    addNextResult();
}

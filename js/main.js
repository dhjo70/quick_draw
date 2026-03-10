import { state } from './config.js?v=6';
import { initChart, startStage1Previews, switchStage, goToStage2, goToStage3 } from './ui.js?v=6';
import { buildModel, startTraining, runBulkEvaluation } from './ai.js?v=6';

// Initialization run on load
async function init() {
    initChart();
    await tf.ready();

    const backend = tf.getBackend();
    const gpuBadge = document.getElementById('gpuBadge');
    if (backend === 'webgl' || backend === 'webgpu') {
        gpuBadge.classList.replace('idle', 'active');
        document.getElementById('gpuText').innerText = `Hardware Accel: ${backend.toUpperCase()}`;
        document.getElementById('valBackend').innerText = backend.toUpperCase() + " GPU";
    } else {
        document.getElementById('gpuText').innerText = `CPU Only Mode (${backend})`;
        document.getElementById('valBackend').innerText = "CPU";
    }

    try {
        const res = await fetch('./dataset.json');
        const dataObj = await res.json();
        state.trainData = dataObj.train;
        state.testData = dataObj.test;

        document.getElementById('s1-train-count').innerText = state.trainData.length;
        document.getElementById('s1-test-count').innerText = state.testData.length;

        document.getElementById('btnGoStage2').disabled = false;
        document.getElementById('s1-status').innerText = "Data loaded successfully. Ready to proceed.";

        // Start preview animations for Stage 1
        startStage1Previews();

        // Build Model architecture eagerly
        buildModel();

    } catch (e) {
        console.error(e);
        document.getElementById('s1-status').innerText = "Error Loading Data. Ensure prepare_data.py was run.";
        document.getElementById('s1-status').style.color = "var(--accent-red)";
    }
}

// Bind Global click handlers using element IDs
document.getElementById('nav-s1').onclick = () => switchStage(1);

document.getElementById('nav-s2').onclick = () => {
    if (state.tensors.trainXs || document.getElementById('nav-s1').classList.contains('completed')) {
        switchStage(2);
    }
}

document.getElementById('nav-s3').onclick = () => {
    if (document.getElementById('nav-s2').classList.contains('completed')) {
        switchStage(3);
    }
}

document.getElementById('btnGoStage2').onclick = goToStage2;
document.getElementById('btnTrain').onclick = startTraining;
document.getElementById('btnGoStage3').onclick = goToStage3;
document.getElementById('btnEval').onclick = runBulkEvaluation;

// Initialize when module loads
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

import { state, IMG_SIZE, CATEGORIES } from './config.js?v=6';

export function rasterizeToImageData(strokes, canvasSize = 256) {
    let minX = 9999, minY = 9999, maxX = 0, maxY = 0;
    strokes.forEach(s => {
        s[0].forEach(x => { minX = Math.min(minX, x); maxX = Math.max(maxX, x); });
        s[1].forEach(y => { minY = Math.min(minY, y); maxY = Math.max(maxY, y); });
    });

    const w = maxX - minX || 1;
    const h = maxY - minY || 1;
    const size = Math.max(w, h);
    const padding = size * 0.1;
    const scale = canvasSize / (size + padding * 2);

    const offsetX = (canvasSize - w * scale) / 2 - minX * scale;
    const offsetY = (canvasSize - h * scale) / 2 - minY * scale;

    const offCanvas = document.createElement('canvas');
    offCanvas.width = canvasSize;
    offCanvas.height = canvasSize;
    const octx = offCanvas.getContext('2d');

    octx.fillStyle = 'black';
    octx.fillRect(0, 0, canvasSize, canvasSize);
    octx.strokeStyle = 'white';
    octx.lineWidth = canvasSize * 0.08;
    octx.lineCap = 'round';
    octx.lineJoin = 'round';

    strokes.forEach(stroke => {
        const xs = stroke[0];
        const ys = stroke[1];
        if (xs.length < 2) return;
        octx.beginPath();
        octx.moveTo(xs[0] * scale + offsetX, ys[0] * scale + offsetY);
        for (let i = 1; i < xs.length; i++) {
            octx.lineTo(xs[i] * scale + offsetX, ys[i] * scale + offsetY);
        }
        octx.stroke();
    });

    const thumbCanvas = document.createElement('canvas');
    thumbCanvas.width = IMG_SIZE;
    thumbCanvas.height = IMG_SIZE;
    const tctx = thumbCanvas.getContext('2d');
    tctx.imageSmoothingEnabled = true;
    tctx.drawImage(offCanvas, 0, 0, IMG_SIZE, IMG_SIZE);
    return tctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
}

export async function prepareAllTensors() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    const loadText = document.getElementById('loadingText');
    loadingOverlay.style.display = 'flex';

    async function convert(dataList, typeName) {
        const images = [];
        const labels = [];
        for (let i = 0; i < dataList.length; i++) {
            const item = dataList[i];
            const imgData = rasterizeToImageData(item.d);
            const data = new Float32Array(IMG_SIZE * IMG_SIZE);
            for (let j = 0; j < data.length; j++) {
                data[j] = imgData.data[j * 4] / 255.0;
            }
            images.push(data);
            labels.push(CATEGORIES.indexOf(item.c));

            if (i % 200 === 0) {
                loadText.innerText = `Rasterizing ${typeName}... ${i}/${dataList.length}`;
                await tf.nextFrame(); // flush UI
            }
        }
        return {
            xs: tf.tensor2d(images).reshape([dataList.length, IMG_SIZE, IMG_SIZE, 1]),
            ys: tf.tensor1d(labels, 'float32'),
            imgs: images
        };
    }

    const trainSet = await convert(state.trainData, "Train Data");
    state.tensors.trainXs = trainSet.xs;
    state.tensors.trainYs = trainSet.ys;

    const testSet = await convert(state.testData, "Test Data");
    state.tensors.testXs = testSet.xs;
    state.tensors.testYs = testSet.ys;

    // Render test images to DataURLs in advance for Stage 3 speed
    loadText.innerText = `Generating UI thumbnails...`;
    await tf.nextFrame();

    state.testThumbnails = [];
    const tCanv = document.createElement('canvas');
    tCanv.width = 28; tCanv.height = 28;
    const tCtx = tCanv.getContext('2d');
    const tImgData = tCtx.createImageData(28, 28);

    for (let i = 0; i < state.testData.length; i++) {
        const floats = testSet.imgs[i];
        for (let j = 0; j < floats.length; j++) {
            const val = floats[j] * 255;
            tImgData.data[j * 4] = val; // r
            tImgData.data[j * 4 + 1] = val; // g
            tImgData.data[j * 4 + 2] = val; // b
            tImgData.data[j * 4 + 3] = 255; // a
        }
        tCtx.putImageData(tImgData, 0, 0);
        state.testThumbnails.push(tCanv.toDataURL());
        if (i % 50 === 0) await tf.nextFrame();
    }

    loadingOverlay.style.display = 'none';
}

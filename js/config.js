export const CATEGORIES = ["apple", "cat", "clock"];
export const EMOJIS = { "apple": "🍎", "cat": "🐱", "clock": "⏰" };
export const IMG_SIZE = 28;
export const MAX_EPOCHS = 50;

export const state = {
    trainData: [],
    testData: [],
    tensors: { trainXs: null, trainYs: null, testXs: null, testYs: null },
    testThumbnails: [],
    model: null,
    isTraining: false,
    myChart: null,
    chartData: {
        labels: [],
        datasets: [
            { label: 'Train Loss', data: [], borderColor: '#fb923c', yAxisID: 'y' },
            { label: 'Val Accuracy', data: [], borderColor: '#34d399', yAxisID: 'y1' }
        ]
    },
    currentStage: 1
};

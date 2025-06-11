/* attempt at creating [B12, B11, B9] visualisation on SentinelHub */

let minVal = 0.0;
let maxVal = 0.4;

let viz = new HighlightCompressVisualizer(minVal, maxVal);

function setup() {
  return {
    input: ["B12", "B11", "B09"],
    output: { bands: 4 }
  };
}

function evaluatePixel(samples) {
  let val = [samples.B12, samples.B11, samples.B09];
  return viz.processList(val);
}

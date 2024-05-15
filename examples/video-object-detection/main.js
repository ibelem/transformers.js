import {
  env,
  AutoModel,
  AutoProcessor,
  RawImage,
} from "./dist/transformers.js";

// Since we will download the model from the Hugging Face Hub, we can skip the local model check
// env.allowLocalModels = true;
// env.localModelPath = './static/models/yolo-v3-opset-12-fp16_fp16.onnx';
env.allowRemoteModels = true;

env.backends.onnx.wasm.proxy = false;
env.backends.onnx.wasm.simd = true;
env.backends.onnx.wasm.numThreads = 1;
// env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0-esmtest.20240411-1abb64e894/dist/';

const getQueryValue = (name) => {
  const urlParams = new URLSearchParams(window.location.search);
  return urlParams.get(name);
};

// Reference the elements that we will need
const status = document.getElementById("status");
const container = document.getElementById("container");
const overlay = document.getElementById("overlay");
const canvas = document.getElementById("canvas");
const video = document.getElementById("video");
const thresholdSlider = document.getElementById("threshold");
const thresholdLabel = document.getElementById("threshold-value");
const sizeSlider = document.getElementById("size");
const sizeLabel = document.getElementById("size-value");
const scaleSlider = document.getElementById("scale");
const scaleLabel = document.getElementById("scale-value");

function setStreamSize(width, height) {
  video.width = canvas.width = Math.round(width);
  video.height = canvas.height = Math.round(height);
}

status.textContent = "Loading model...";

// Load model and processor
const model_id = "webnn/ssd-mobilenet-v1";
// const model = await AutoModel.from_pretrained(model_id, { device: 'wasm',
// dtype: 'q8' });
let model;
let options = {}
try {
  let provider = document.querySelector('#provider');
  if (getQueryValue("provider") && getQueryValue("provider").toLowerCase() === "webgpu") {
    provider.innerHTML = 'WebGPU';
    options = {
      device: "webgpu",
      dtype: "fp16",
    }
    
  } else {
    provider.innerHTML = 'WebNN';
    options = {
      device: "webnn",
      dtype: "fp16",
      session_options: {
        executionProviders: [
          {
            name: "webnn",
            deviceType: "gpu",
            powerPreference: "default",
            preferredLayout: "NHWC",
          },
        ],
        freeDimensionOverrides: { "unk__6578": 1, "unk__6579": 224, "unk__6580": 224 },
        // freeDimensionOverrides: { batch_size: 1, height: 256, width: 320 },
        logSeverityLevel: 0,
      },
    }
  }

  model = await AutoModel.from_pretrained(model_id, options);
} catch (err) {
  console.log(err.message);
}

const processor = await AutoProcessor.from_pretrained(model_id);

// Set up controls
let scale = 0.5;
scaleSlider.addEventListener("input", () => {
  scale = Number(scaleSlider.value);
  setStreamSize(video.videoWidth * scale, video.videoHeight * scale);
  scaleLabel.textContent = scale;
});
scaleSlider.disabled = false;

let threshold = 0.25;
thresholdSlider.addEventListener("input", () => {
  threshold = Number(thresholdSlider.value);
  thresholdLabel.textContent = threshold.toFixed(2);
});
thresholdSlider.disabled = false;

let size = 128;
processor.feature_extractor.size = { shortest_edge: size };
sizeSlider.addEventListener("input", () => {
  size = Number(sizeSlider.value);
  processor.feature_extractor.size = { shortest_edge: size };
  sizeLabel.textContent = size;
});
sizeSlider.disabled = false;

status.textContent = "Ready";

const COLOURS = [
  "#EF4444",
  "#4299E1",
  "#059669",
  "#FBBF24",
  "#4B52B1",
  "#7B3AC2",
  "#ED507A",
  "#1DD1A1",
  "#F3873A",
  "#4B5563",
  "#DC2626",
  "#1852B4",
  "#18A35D",
  "#F59E0B",
  "#4059BE",
  "#6027A5",
  "#D63D60",
  "#00AC9B",
  "#E64A19",
  "#272A34",
];

// Render a bounding box and label on the image
function renderBox([xmin, ymin, xmax, ymax, score, id], [w, h]) {
  if (score < threshold) return; // Skip boxes with low confidence

  // Generate a random color for the box
  const color = COLOURS[id % COLOURS.length];

  // Draw the box
  const boxElement = document.createElement("div");
  boxElement.className = "bounding-box";
  Object.assign(boxElement.style, {
    borderColor: color,
    left: (100 * xmin) / w + "%",
    top: (100 * ymin) / h + "%",
    width: (100 * (xmax - xmin)) / w + "%",
    height: (100 * (ymax - ymin)) / h + "%",
  });

  // Draw label
  const labelElement = document.createElement("span");
  labelElement.textContent = `${model.config.id2label[id]} (${(
    100 * score
  ).toFixed(2)}%)`;
  labelElement.className = "bounding-box-label";
  labelElement.style.backgroundColor = color;

  boxElement.appendChild(labelElement);
  overlay.appendChild(boxElement);
}

let isProcessing = false;
let previousTime;
const context = canvas.getContext("2d", { willReadFrequently: true });
function updateCanvas() {
  const { width, height } = canvas;
  context.drawImage(video, 0, 0, width, height);

  if (!isProcessing) {
    isProcessing = true;
    (async function () {
      // Read the current frame from the video
      const pixelData = context.getImageData(0, 0, width, height).data;
      const image = new RawImage(pixelData, width, height, 4);

      // Process the image and run the model
      const inputs = await processor(image);
      const { outputs } = await model(inputs);

      // Update UI
      overlay.innerHTML = "";

      const sizes = inputs.reshaped_input_sizes[0].reverse();
      console.log(outputs);
      console.log(outputs.tolist());

      // renderBox([0.375, 14.28125, 136.375, 127.75, 0.9404296875, 0], sizes);
      // renderBox([xmin, ymin, xmax, ymax, score, id], [w, h]) 
      outputs.tolist().forEach((x) => renderBox(x, sizes));

      if (previousTime !== undefined) {
        const fps = 1000 / (performance.now() - previousTime);
        status.textContent = `FPS: ${fps.toFixed(2)}`;
      }
      previousTime = performance.now();
      isProcessing = false;
    })();
  }

  window.requestAnimationFrame(updateCanvas);
}

// Start the video stream
navigator.mediaDevices
  .getUserMedia(
    { video: true } // Ask for video
  )
  .then((stream) => {
    // Set up the video and canvas elements.
    video.srcObject = stream;
    video.play();

    const videoTrack = stream.getVideoTracks()[0];
    const { width, height } = videoTrack.getSettings();

    setStreamSize(width * scale, height * scale);

    // Set container width and height depending on the image aspect ratio
    const ar = width / height;
    const [cw, ch] = ar > 720 / 405 ? [720, 720 / ar] : [405 * ar, 405];
    container.style.width = `${cw}px`;
    container.style.height = `${ch}px`;

    // Start the animation loop
    window.requestAnimationFrame(updateCanvas);
  })
  .catch((error) => {
    alert(error);
  });

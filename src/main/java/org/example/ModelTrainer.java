package org.example;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.types.TFloat32;

import java.nio.file.*;
import org.tensorflow.Result;
import java.util.HashMap;
import java.util.Map;

public class ModelTrainer {

    private static final Logger logger = LoggerFactory.getLogger(ModelTrainer.class);

    private final int inputSize;
    private final int numClasses = 2; // 0 = real, 1 = ai_generated
    private SavedModelBundle model;

    public ModelTrainer(int inputSize) {
        this.inputSize = inputSize;
    }

    public void loadModel(String modelPath) {
        Path path = Paths.get(modelPath);
        validateModelDirectory(path);

        logger.info("Loading SavedModel from: {}", modelPath);
        try {
            if (model != null) {
                model.close();
            }
            model = SavedModelBundle.load(modelPath, "serve");
            logger.info("Model loaded successfully.");
            logModelSignatures();
        } catch (Exception e) {
            throw new RuntimeException("Failed to load SavedModel from: " + modelPath, e);
        }
    }

    public boolean isModelLoaded() {
        return model != null;
    }

    public SavedModelBundle getModel() {
        if (!isModelLoaded()) {
            throw new IllegalStateException("No model loaded. Call loadModel() first.");
        }
        return model;
    }

    public void close() {
        if (model != null) {
            model.close();
            model = null;
            logger.info("Model closed and resources released.");
        }
    }

    public float smokeTest(String inputNodeName, String outputNodeName) {
        if (!isModelLoaded()) {
            throw new IllegalStateException("No model loaded. Call loadModel() first.");
        }

        logger.info("Running smoke test with a blank {}x{}x3 image...", inputSize, inputSize);

        float[] blankPixels = new float[1 * inputSize * inputSize * 3];
        Shape inputShape = Shape.of(1, inputSize, inputSize, 3);

        try (TFloat32 inputTensor = TFloat32.tensorOf(inputShape, DataBuffers.of(blankPixels))) {

            Map<String, org.tensorflow.Tensor> feeds = new HashMap<>();
            feeds.put(inputNodeName, inputTensor);

            org.tensorflow.Result fetches =
                    model.function("serving_default").call(feeds);

            try (TFloat32 outputTensor = (TFloat32) fetches.get(outputNodeName).get()) {
                float confidence = extractConfidence(outputTensor);
                logger.info("Smoke test passed — confidence for blank image: {}", confidence);
                return confidence;
            }

        } catch (Exception e) {
            throw new RuntimeException(
                    "Smoke test failed. Check that inputNodeName/outputNodeName match the "
                            + "model's serving signature. "
                            + "Run: saved_model_cli show --dir <path> --all", e);
        }
    }

    public float evaluate(float[][][][] images, int[] labels,
                          String inputName, String outputName) {
        if (!isModelLoaded()) {
            throw new IllegalStateException("No model loaded. Call loadModel() first.");
        }
        if (images.length != labels.length) {
            throw new IllegalArgumentException("images and labels must have the same length.");
        }

        int correct = 0;
        int total   = images.length;

        for (int i = 0; i < total; i++) {
            float confidence = runSingleInference(images[i], inputName, outputName);
            int predicted    = confidence > 0.5f ? 1 : 0;
            if (predicted == labels[i]) correct++;
        }

        float accuracy = (float) correct / total;
        logger.info("Evaluation complete — {}/{} correct, accuracy = {}%",
                correct, total, String.format("%.2f", accuracy * 100));
        return accuracy;
    }

    private float runSingleInference(float[][][] image, String inputName, String outputName) {
        int h = image.length;
        int w = image[0].length;
        int c = image[0][0].length;

        float[] pixelData = new float[h * w * c];
        int idx = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                for (int ch = 0; ch < c; ch++) {
                    pixelData[idx++] = image[y][x][ch];
                }
            }
        }

        Shape shape = Shape.of(1, h, w, c);

        try (TFloat32 tensor = TFloat32.tensorOf(shape, DataBuffers.of(pixelData))) {
            Map<String, org.tensorflow.Tensor> feeds = new HashMap<>();
            feeds.put(inputName, tensor);

            org.tensorflow.Result fetches =
                    model.function("serving_default").call(feeds);

            try (TFloat32 outputTensor = (TFloat32) fetches.get(outputName).get()) {
                return extractConfidence(outputTensor);
            }
        }
    }

    private float extractConfidence(TFloat32 t) {
        long rank = t.rank();
        long size = t.size();

        logger.debug("Output tensor — rank: {}, size: {}", rank, size);

        // Scalar — rank 0
        if (rank == 0) {
            return t.getFloat();
        }

        // Shape [1] — rank 1, single element
        if (rank == 1) {
            return t.getFloat(0);
        }

        // Shape [1, 1] — rank 2, single sigmoid output (your model hits this)
        if (rank == 2) {
            return t.getFloat(0, 0);
        }

        // Fallback — read all values and return the last one
        logger.warn("Unexpected output tensor rank {}; using fallback.", rank);
        float[] values = new float[(int) size];
        t.asRawTensor().data().asFloats().read(values);
        return values[0];
    }

    private void validateModelDirectory(Path modelPath) {
        if (!Files.isDirectory(modelPath)) {
            throw new IllegalArgumentException(
                    "Model path does not exist or is not a directory: " + modelPath);
        }
        Path pbFile = modelPath.resolve("saved_model.pb");
        if (!Files.exists(pbFile)) {
            throw new IllegalArgumentException(
                    "saved_model.pb not found in: " + modelPath
                            + ". Export from Python with: model.save('<path>')");
        }
    }

    private void logModelSignatures() {
        try {
            var signatures = model.metaGraphDef().getSignatureDefMap();
            logger.info("Available serving signatures:");
            signatures.forEach((key, sig) -> {
                logger.info("  Signature key: '{}'", key);
                sig.getInputsMap().forEach((k, v) ->
                        logger.info("    INPUT  '{}' -> tensor name: '{}'", k, v.getName()));
                sig.getOutputsMap().forEach((k, v) ->
                        logger.info("    OUTPUT '{}' -> tensor name: '{}'", k, v.getName()));
            });
        } catch (Exception e) {
            logger.warn("Could not read model signatures: {}", e.getMessage());
        }
    }
}
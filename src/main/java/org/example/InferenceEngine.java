package org.example;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.DataBuffers;
import org.tensorflow.types.TFloat32;

import java.nio.file.Path;
import org.tensorflow.Result;
import java.util.HashMap;
import java.util.Map;

public class InferenceEngine {
    private static final Logger logger = LoggerFactory.getLogger(InferenceEngine.class);

    private final ImageProcessor imageProcessor;
    private SavedModelBundle model;
    private boolean modelLoaded = false;
    private final int inputSize = 224;

    private static final String INPUT_NODE  = "input_image";
    private static final String OUTPUT_NODE = "output_0";

    public InferenceEngine(ImageProcessor imageProcessor) {
        this.imageProcessor = imageProcessor;
    }

    public void loadModel(String modelPath) {
        try {
            model = SavedModelBundle.load(modelPath, "serve");
            modelLoaded = true;
            logger.info("Model loaded successfully from {}", modelPath);
        } catch (Exception e) {
            logger.error("Failed to load model from {}", modelPath, e);
            modelLoaded = false;
        }
    }

    public boolean isModelLoaded() {
        return modelLoaded;
    }

    public void close() {
        if (model != null) {
            model.close();
            model = null;
            modelLoaded = false;
            logger.info("Model closed and resources released.");
        }
    }

    public DetectionResult detect(Path imagePath) {
        if (!modelLoaded) {
            throw new IllegalStateException("Model not loaded — call loadModel() first.");
        }

        try {
            Mat image = imageProcessor.loadAndPreprocess(imagePath);

            try (TFloat32 inputTensor = convertMatToTensor(image)) {

                Map<String, org.tensorflow.Tensor> feeds = new HashMap<>();
                feeds.put(INPUT_NODE, inputTensor);

                org.tensorflow.Result fetches =
                        model.function("serving_default").call(feeds);

                try (TFloat32 outputTensor = (TFloat32) fetches.get(OUTPUT_NODE).get()) {
                    float confidence      = extractConfidence(outputTensor);
                    boolean isAIGenerated = confidence > 0.5f;
                    logger.debug("Detection for {}: aiGenerated={}, confidence={}",
                            imagePath.getFileName(), isAIGenerated, String.format("%.3f", confidence));
                    return new DetectionResult(isAIGenerated, confidence);
                }
            }

        } catch (Exception e) {
            logger.error("Detection failed for {}", imagePath, e);
            return new DetectionResult(false, 0.0f);
        }
    }

    private float extractConfidence(TFloat32 t) {
        long rank = t.rank();
        long size = t.size();

        logger.debug("Output tensor — rank: {}, size: {}", rank, size);

        if (rank == 0) {
            return t.getFloat();
        }

        if (rank == 1) {
            return t.getFloat(0);
        }

        if (rank == 2) {
            return t.getFloat(0, 0);
        }

        logger.warn("Unexpected output tensor rank {}; using fallback.", rank);
        float[] values = new float[(int) size];
        t.asRawTensor().data().asFloats().read(values);
        return values[0];
    }

    private TFloat32 convertMatToTensor(Mat image) {
        Mat floatImage;
        if (image.type() == CvType.CV_32FC3) {
            floatImage = image;
        } else {
            floatImage = new Mat();
            image.convertTo(floatImage, CvType.CV_32FC3);
            logger.warn("convertMatToTensor: Mat was not CV_32FC3 — type-cast only, no rescaling. "
                    + "Ensure ImageProcessor normalises to [0,1] before this call.");
        }

        int height   = floatImage.rows();
        int width    = floatImage.cols();
        int channels = floatImage.channels();

        float[] pixelData = new float[height * width * channels];
        int idx = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double[] pixel = floatImage.get(y, x);
                for (int c = 0; c < channels; c++) {
                    pixelData[idx++] = (float) pixel[c];
                }
            }
        }

        Shape shape = Shape.of(1, height, width, channels);
        return TFloat32.tensorOf(shape, DataBuffers.of(pixelData));
    }

    public static class DetectionResult {
        private final boolean isAIGenerated;
        private final float confidence;

        public DetectionResult(boolean isAIGenerated, float confidence) {
            this.isAIGenerated = isAIGenerated;
            this.confidence    = confidence;
        }

        public boolean isAIGenerated() { return isAIGenerated; }
        public float   getConfidence() { return confidence; }

        @Override
        public String toString() {
            return String.format("DetectionResult{aiGenerated=%s, confidence=%.3f}",
                    isAIGenerated, confidence);
        }
    }
}
package org.example;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

public class ImageProcessor {
    private static final Logger logger = LoggerFactory.getLogger(ImageProcessor.class);
    private final int targetWidth;
    private final int targetHeight;

    static {
        nu.pattern.OpenCV.loadLocally();
    }

    public ImageProcessor(int targetWidth, int targetHeight) {
        this.targetWidth = targetWidth;
        this.targetHeight = targetHeight;
    }

    public Mat loadAndPreprocess(Path imagePath) {
        Mat original = Imgcodecs.imread(imagePath.toString());
        if (original.empty()) {
            throw new RuntimeException("Failed to load image: " + imagePath);
        }

        Mat resized = new Mat();
        Imgproc.resize(original, resized, new Size(targetWidth, targetHeight));

        Mat normalized = new Mat();
        resized.convertTo(normalized, CvType.CV_32FC3, 1.0/255.0);

        return normalized;
    }

    public float[] extractFeatures(Mat image) {
        float[] features = new float[9];

        Mat[] channels = new Mat[3];
        Core.split(image, Arrays.asList(channels));

        for (int c = 0; c < 3; c++) {
            Mat channel = channels[c];

            Scalar mean = Core.mean(channel);
            features[c * 3] = (float) mean.val[0];

            MatOfDouble stdDev = new MatOfDouble();
            Core.meanStdDev(channel, new MatOfDouble(), stdDev);
            features[c * 3 + 1] = (float) stdDev.toArray()[0];

            features[c * 3 + 2] = calculateEntropy(channel);
        }

        return features;
    }

    private float calculateEntropy(Mat channel) {
        Mat hist = new Mat();
        Imgproc.calcHist(List.of(channel), new MatOfInt(0), new Mat(),
                hist, new MatOfInt(256), new MatOfFloat(0f, 1f));
        Core.normalize(hist, hist, 1, 0, Core.NORM_L1);
        float entropy = 0f;
        for (int i = 0; i < 256; i++) {
            float p = (float) hist.get(i, 0)[0];
            if (p > 0) entropy -= p * (float)(Math.log(p) / Math.log(2));
        }
        return entropy;
    }

    public void saveProcessedImage(Mat image, Path outputPath) {
        Mat output = new Mat();
        image.convertTo(output, CvType.CV_8UC3, 255.0);
        Imgcodecs.imwrite(outputPath.toString(), output);
    }
}

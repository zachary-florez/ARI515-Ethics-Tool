package org.example;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class Main {
    private static final Logger logger = LoggerFactory.getLogger(Main.class);

    public static void main(String[] args) {
        logger.info("Starting Satellite AI Detection Platform...");
        System.out.println("Working dir: " + System.getProperty("user.dir"));
        String dataPath = args.length > 0 ? args[0] : "data";
        String metadataPath = dataPath + "/metadata.csv";

        try {
            DataLoader dataLoader = new DataLoader(dataPath);
            dataLoader.loadMetadata(metadataPath);
            logger.info("Loaded {} images", dataLoader.getAllMetadata().size());

            ImageProcessor imageProcessor = new ImageProcessor(224, 224);
            InferenceEngine inferenceEngine = new InferenceEngine(imageProcessor);

            String modelPath = args.length > 1 ? args[1] : "./models/satellite_detector";
            inferenceEngine.loadModel(modelPath);

            ModelTrainer modelTrainer = new ModelTrainer(224);
            modelTrainer.loadModel(modelPath);

            float smokeConfidence = modelTrainer.smokeTest("input_image", "output_0");
            logger.info("Smoke test confidence: {}", smokeConfidence);

            // Integration part, never got it to it
            // OPENAI_API_KEY --
            LLMIntegrator llmIntegrator = new LLMIntegrator(
                    System.getenv("OPENAI_API_KEY"),
                    ""
            );


            DetectionTracker tracker = new DetectionTracker();
            List<ImageMetadata> images = dataLoader.getAllMetadata();
            int processed = 0;

            for (ImageMetadata image : images) {
                Path imagePath = dataLoader.getImagePath(image);

                InferenceEngine.DetectionResult result = inferenceEngine.detect(imagePath);
                tracker.recordDetection(image.getImageId(), result);

                // if it was between 40% & 80%
                if (result.getConfidence() > 0.4 && result.getConfidence() < 0.8) {
                    String llmAnalysis = llmIntegrator.analyzeWithLLM(imagePath.toString(), result);
                    logger.info("LLM Analysis for {}: {}", image.getImageId(), llmAnalysis);
                }

                processed++;
                if (processed % 100 == 0) {
                    logger.info("Processed {}/{} images", processed, images.size());
                }
            }
            long highConfidence  = images.stream()
                    .map(img -> tracker.getDetectionHistory(img.getImageId()))
                    .filter(h -> !h.isEmpty())
                    .map(h -> h.get(h.size() - 1))
                    .filter(r -> r.getConfidence() > 0.8f || r.getConfidence() < 0.2f)
                    .count();
            long uncertain = images.size() - highConfidence;
            logger.info("High confidence detections (>0.8 or <0.2): {}/{}", highConfidence, images.size());
            logger.info("Uncertain detections (0.2-0.8): {}/{}", uncertain, images.size());
            DetectionTracker.DetectionStatistics stats = tracker.getStatistics();
            logger.info("Detection complete: {}", stats);

            long correct = 0, total = 0;
            for (ImageMetadata image : images) {
                if (image.getLabel().equals("unknown")) continue;
                List<DetectionTracker.DetectionRecord> history =
                        tracker.getDetectionHistory(image.getImageId());
                if (history.isEmpty()) continue;

                boolean predictedAI = history.get(history.size()-1).isAIGenerated();
                boolean actualAI    = image.getLabel().equals("ai_generated");
                if (predictedAI == actualAI) correct++;
                total++;
            }
            if (total > 0) {
                logger.info("Accuracy on labeled images: {}/{} = {}%",
                        correct, total, String.format("%.1f", correct * 100.0 / total));
            }

            SatelliteImageFetcher fetcher = new SatelliteImageFetcher(
                    inferenceEngine,
                    llmIntegrator,
                    "./data/realworld_cache"
            );

//            // NASA Epic
//            SatelliteImageFetcher.RealWorldDetectionResult nasaResult =
//                    fetcher.analyzeNASAEpicImage();
//            logger.info(nasaResult.getSummary());

            // Mapbox
            String mapboxToken = System.getenv("MAPBOX_TOKEN"); // set in environment
            if (mapboxToken != null) {

                SatelliteImageFetcher.RealWorldDetectionResult nycResult =
                        fetcher.analyzeMapboxTile(20, 30, 16, mapboxToken);
                logger.info(nycResult.getSummary());

                SatelliteImageFetcher.RealWorldDetectionResult areaResult =
                        fetcher.analyzeMapboxTile(51, -2, 15, mapboxToken); // London
                logger.info(areaResult.getSummary());
            } else {
                logger.info("ERROR, mapbox token was null...");
            }

            // I just picked random AI generated images from my original data
            String[] aiImages = {
                    "./data/raw/000000001.jpg",
                    "./data/raw/000000012.jpg",
                    "./data/raw/000000023.jpg",
                    "./data/raw/000000034.jpg",
                    "./data/raw/000000045.jpg"
            };

            for (String aiImagePath : aiImages) {
                SatelliteImageFetcher.RealWorldDetectionResult result =
                        fetcher.analyzeLocalFile(Paths.get(aiImagePath));
                logger.info(result.getSummary());
            }

            // I just picked random real images from my original data
            String[] realImages = {
                    "./data/raw/000000051-r.jpg",
                    "./data/raw/000000062-r.jpg",
                    "./data/raw/000000073-r.jpg",
                    "./data/raw/000000084-r.jpg",
                    "./data/raw/000000095-r.jpg"
            };

            for (String realImagePath : realImages) {
                SatelliteImageFetcher.RealWorldDetectionResult result =
                        fetcher.analyzeLocalFile(Paths.get(realImagePath));
                logger.info(result.getSummary());
            }

            ReportGenerator reporter = new ReportGenerator("./reports");
            Path reportPath = reporter.generateReport(images, tracker);
            logger.info("Report saved — open in browser: file://{}", reportPath.toAbsolutePath());

        } catch (IOException e) {
            logger.error("Failed to load data", e);
        } catch (Exception e) {
            logger.error("Application error", e);
        }

    }
}
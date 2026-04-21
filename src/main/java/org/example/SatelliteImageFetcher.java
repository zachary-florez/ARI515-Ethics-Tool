package org.example;

import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.Response;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;

public class SatelliteImageFetcher {
    private static final Logger logger = LoggerFactory.getLogger(SatelliteImageFetcher.class);

    private final OkHttpClient httpClient;
    private final InferenceEngine inferenceEngine;
    private final LLMIntegrator llmIntegrator;
    private final Path cacheDir;

    public SatelliteImageFetcher(InferenceEngine inferenceEngine,
                                 LLMIntegrator llmIntegrator,
                                 String cacheDirPath) throws IOException {
        this.inferenceEngine = inferenceEngine;
        this.llmIntegrator   = llmIntegrator;
        this.cacheDir        = Paths.get(cacheDirPath);
        Files.createDirectories(this.cacheDir);

        this.httpClient = new OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)
                .build();
    }

    public RealWorldDetectionResult analyzeNASAEpicImage() throws IOException {
        logger.info("Fetching latest NASA EPIC satellite image...");

        String metadataUrl = "https://epic.gsfc.nasa.gov/api/natural";
        String metadataJson = fetchText(metadataUrl);

        String filename = extractJsonField(metadataJson, "image");
        String date     = extractJsonField(metadataJson, "date"); 
        String datePath = date.substring(0, 10).replace("-", "/"); 

        String imageUrl = String.format(
                "https://epic.gsfc.nasa.gov/archive/natural/%s/png/%s.png",
                datePath, filename);

        logger.info("Downloading NASA EPIC image: {}", filename);
        return analyzeFromUrl(imageUrl, filename + ".png", "NASA EPIC");
    }

    public RealWorldDetectionResult analyzeMapboxTile(double latitude, double longitude,
                                                      int zoom, String mapboxToken) throws IOException {
        logger.info("Fetching Mapbox satellite tile for ({}, {}) zoom {}...",
                latitude, longitude, zoom);

        String imageUrl = String.format(
                "https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/%f,%f,%d,0/512x512"
                        + "?access_token=%s",
                longitude, latitude, zoom, mapboxToken);

        String cacheName = String.format("mapbox_%.4f_%.4f_z%d.jpg", latitude, longitude, zoom);
        return analyzeFromUrl(imageUrl, cacheName, "Mapbox");
    }

    public RealWorldDetectionResult analyzeFromUrl(String imageUrl,
                                                   String cacheName,
                                                   String sourceName) throws IOException {
        Path localPath = downloadImage(imageUrl, cacheName);

        logger.info("Running detection on {} image: {}", sourceName, cacheName);
        InferenceEngine.DetectionResult mlResult = inferenceEngine.detect(localPath);

        String llmAnalysis = null;
        if (mlResult.getConfidence() > 0.4f && mlResult.getConfidence() < 0.8f) {
            logger.info("  Confidence uncertain ({}), running LLM analysis...",
                    String.format("%.3f", mlResult.getConfidence()));
            llmAnalysis = llmIntegrator.analyzeWithLLM(localPath.toString(), mlResult);
        }

        RealWorldDetectionResult result = new RealWorldDetectionResult(
                cacheName, sourceName, imageUrl, mlResult, llmAnalysis);

        logResult(result);
        return result;
    }

    public RealWorldDetectionResult analyzeLocalFile(Path imagePath) throws IOException {
        logger.info("Analyzing local file: {}", imagePath);
        InferenceEngine.DetectionResult mlResult = inferenceEngine.detect(imagePath);

        String llmAnalysis = null;
        if (mlResult.getConfidence() > 0.4f && mlResult.getConfidence() < 0.8f) {
            llmAnalysis = llmIntegrator.analyzeWithLLM(imagePath.toString(), mlResult);
        }

        RealWorldDetectionResult result = new RealWorldDetectionResult(
                imagePath.getFileName().toString(), "local",
                imagePath.toString(), mlResult, llmAnalysis);

        logResult(result);
        return result;
    }

    private Path downloadImage(String url, String cacheName) throws IOException {
        Path cachedPath = cacheDir.resolve(cacheName);

        if (Files.exists(cachedPath)) {
            logger.info("Using cached image: {}", cachedPath);
            return cachedPath;
        }

        logger.info("Downloading: {}", url);
        Request request = new Request.Builder().url(url).build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful() || response.body() == null) {
                throw new IOException("Failed to download image from: " + url
                        + " — HTTP " + response.code());
            }

            try (InputStream inputStream = response.body().byteStream()) {
                Files.copy(inputStream, cachedPath);
            }
        }

        logger.info("Saved image to: {}", cachedPath);
        return cachedPath;
    }

    private String fetchText(String url) throws IOException {
        Request request = new Request.Builder().url(url).build();
        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful() || response.body() == null) {
                throw new IOException("Failed to fetch: " + url + " — HTTP " + response.code());
            }
            return response.body().string();
        }
    }

    private String extractJsonField(String json, String key) {
        String search = "\"" + key + "\":\"";
        int start = json.indexOf(search);
        if (start == -1) throw new RuntimeException("Field '" + key + "' not found in JSON response");
        start += search.length();
        int end = json.indexOf("\"", start);
        return json.substring(start, end);
    }

    private void logResult(RealWorldDetectionResult result) {
        logger.info("=== DETECTION RESULT ===");
        logger.info("  Image:       {}", result.getImageName());
        logger.info("  Source:      {}", result.getSource());
        logger.info("  Verdict:     {}", result.getMlResult().isAIGenerated()
                ? "AI-GENERATED" : "REAL");
        logger.info("  Confidence:  {}%", String.format("%.1f", result.getMlResult().getConfidence() * 100));
        if (result.getLlmAnalysis() != null) {
            logger.info("  LLM Analysis: {}", result.getLlmAnalysis());
        }
        logger.info("========================");
    }

    public static class RealWorldDetectionResult {
        private final String imageName;
        private final String source;
        private final String imageUrl;
        private final InferenceEngine.DetectionResult mlResult;
        private final String llmAnalysis; // null if confidence was high

        public RealWorldDetectionResult(String imageName, String source, String imageUrl,
                                        InferenceEngine.DetectionResult mlResult,
                                        String llmAnalysis) {
            this.imageName   = imageName;
            this.source      = source;
            this.imageUrl    = imageUrl;
            this.mlResult    = mlResult;
            this.llmAnalysis = llmAnalysis;
        }

        public String getImageName()   { return imageName; }
        public String getSource()      { return source; }
        public String getImageUrl()    { return imageUrl; }
        public InferenceEngine.DetectionResult getMlResult() { return mlResult; }
        public String getLlmAnalysis() { return llmAnalysis; }

        public String getSummary() {
            return String.format("[%s] %s — %s (%.1f%% confidence)%s",
                    source, imageName,
                    mlResult.isAIGenerated() ? "AI-GENERATED" : "REAL",
                    mlResult.getConfidence() * 100,
                    llmAnalysis != null ? "\n  LLM: " + llmAnalysis : "");
        }
    }
}

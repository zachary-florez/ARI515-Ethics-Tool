// src/main/java/com/satellite/detector/tracking/DetectionTracker.java
package org.example;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

public class DetectionTracker {
    private static final Logger logger = LoggerFactory.getLogger(DetectionTracker.class);
    private final Map<String, List<DetectionRecord>> detectionHistory;

    public DetectionTracker() {
        this.detectionHistory = new ConcurrentHashMap<>();
    }

    public void recordDetection(String imageId, InferenceEngine.DetectionResult result) {
        DetectionRecord record = new DetectionRecord(imageId, result, LocalDateTime.now());
        detectionHistory.computeIfAbsent(imageId, k -> new ArrayList<>()).add(record);
        logger.info("Recorded detection for {}: {}", imageId, result);
    }

    public DetectionStatistics getStatistics() {
        int totalDetections = 0;
        int aiGeneratedCount = 0;

        for (List<DetectionRecord> records : detectionHistory.values()) {
            if (!records.isEmpty()) {
                totalDetections++;
                if (records.get(records.size() - 1).isAIGenerated()) {
                    aiGeneratedCount++;
                }
            }
        }

        return new DetectionStatistics(totalDetections, aiGeneratedCount);
    }

    public List<DetectionRecord> getDetectionHistory(String imageId) {
        return detectionHistory.getOrDefault(imageId, Collections.emptyList());
    }

    public static class DetectionRecord {
        private final String imageId;
        private final boolean isAIGenerated;
        private final float confidence;
        private final LocalDateTime timestamp;

        public DetectionRecord(String imageId, InferenceEngine.DetectionResult result, LocalDateTime timestamp) {
            this.imageId = imageId;
            this.isAIGenerated = result.isAIGenerated();
            this.confidence = result.getConfidence();
            this.timestamp = timestamp;
        }

        public boolean isAIGenerated() { return isAIGenerated; }
        public float getConfidence() { return confidence; }
        public LocalDateTime getTimestamp() { return timestamp; }
    }

    public static class DetectionStatistics {
        private final int totalImagesAnalyzed;
        private final int aiGeneratedCount;

        public DetectionStatistics(int total, int aiCount) {
            this.totalImagesAnalyzed = total;
            this.aiGeneratedCount = aiCount;
        }

        public double getAiPercentage() {
            return totalImagesAnalyzed > 0 ? (aiGeneratedCount * 100.0 / totalImagesAnalyzed) : 0.0;
        }

        @Override
        public String toString() {
            return String.format("Statistics: %d images, %.1f%% AI-generated",
                    totalImagesAnalyzed, getAiPercentage());
        }
    }
}
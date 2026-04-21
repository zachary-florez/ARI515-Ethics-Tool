package org.example;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

public class ReportGenerator {
    private static final Logger logger = LoggerFactory.getLogger(ReportGenerator.class);
    private static final DateTimeFormatter TIMESTAMP_FMT =
            DateTimeFormatter.ofPattern("yyyy-MM-dd_HH-mm-ss");
    private static final DateTimeFormatter DISPLAY_FMT =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    private static final int BAR_WIDTH = 40; 

    private final Path reportsDir;

    public ReportGenerator(String reportsDirPath) throws IOException {
        this.reportsDir = Paths.get(reportsDirPath);
        Files.createDirectories(reportsDir);
    }

    public Path generateReport(List<ImageMetadata> images,
                               DetectionTracker tracker) throws IOException {
        String timestamp   = LocalDateTime.now().format(TIMESTAMP_FMT);
        String displayTime = LocalDateTime.now().format(DISPLAY_FMT);
        Path reportPath    = reportsDir.resolve("report_" + timestamp + ".txt");
        int total     = 0;
        int aiCount   = 0;
        int realCount = 0;
        long correct  = 0;
        long labeled  = 0;
        int[] buckets = new int[10];

        for (ImageMetadata image : images) {
            List<DetectionTracker.DetectionRecord> history =
                    tracker.getDetectionHistory(image.getImageId());
            if (history.isEmpty()) continue;
            DetectionTracker.DetectionRecord latest = history.get(history.size() - 1);
            total++;
            if (latest.isAIGenerated()) aiCount++;
            else realCount++;
            int bucket = Math.min((int)(latest.getConfidence() * 10), 9);
            buckets[bucket]++;
            if (image.getLabel() != null && !image.getLabel().equals("unknown")) {
                labeled++;
                boolean actualAI    = image.getLabel().equals("ai_generated");
                boolean predictedAI = latest.isAIGenerated();
                if (actualAI == predictedAI) correct++;
            }
        }

        double aiPct       = total > 0 ? (aiCount   * 100.0 / total)   : 0;
        double realPct     = total > 0 ? (realCount  * 100.0 / total)   : 0;
        double accuracy    = labeled > 0 ? (correct  * 100.0 / labeled) : -1;
        long   highConf    = 0;
        long   uncertain   = 0;
        for (ImageMetadata image : images) {
            List<DetectionTracker.DetectionRecord> history =
                    tracker.getDetectionHistory(image.getImageId());
            if (history.isEmpty()) continue;
            float conf = history.get(history.size() - 1).getConfidence();
            if (conf > 0.8f || conf < 0.2f) highConf++;
            else uncertain++;
        }

        try (PrintWriter w = new PrintWriter(new FileWriter(reportPath.toFile()))) {
            line(w, "=", 60);
            w.println("  ETHICSAITOOL — DETECTION REPORT");
            w.println("  Generated: " + displayTime);
            line(w, "=", 60);
            w.println();

            w.println("SUMMARY");
            line(w, "-", 60);
            w.printf("  Total Processed  : %d%n", total);
            w.printf("  AI-Generated     : %d  (%.1f%%)%n", aiCount, aiPct);
            w.printf("  Real             : %d  (%.1f%%)%n", realCount, realPct);
            w.printf("  High Confidence  : %d  (>80%% or <20%%)%n", highConf);
            w.printf("  Uncertain        : %d  (20%%–80%%)%n", uncertain);
            if (accuracy >= 0) {
                w.printf("  Accuracy         : %.1f%%  (%d/%d labeled images)%n",
                        accuracy, correct, labeled);
            } else {
                w.println("  Accuracy         : N/A (no labeled images)");
            }
            w.println();

            w.println("VERDICT DISTRIBUTION");
            line(w, "-", 60);
            printBar(w, "AI-Generated", aiCount,  total);
            printBar(w, "Real        ", realCount, total);
            w.println();

            w.println("CONFIDENCE SCORE DISTRIBUTION");
            line(w, "-", 60);
            String[] labels = {
                    " 0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
                    "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"
            };
            int max = 1;
            for (int b : buckets) max = Math.max(max, b);
            for (int i = 0; i < 10; i++) {
                printBar(w, labels[i], buckets[i], max);
            }
            w.println();

            if (labeled > 0) {
                w.println("ACCURACY ON LABELED IMAGES");
                line(w, "-", 60);
                printBar(w, "Correct  ", (int) correct,           (int) labeled);
                printBar(w, "Incorrect", (int)(labeled - correct), (int) labeled);
                w.println();
            }
            line(w, "=", 60);
            w.println("  END OF REPORT");
            line(w, "=", 60);
        }
        logger.info("Report saved to: {}", reportPath.toAbsolutePath());
        return reportPath;
    }

    private void printBar(PrintWriter w, String label, int value, int max) {
        int filled  = max > 0 ? (int)((value * (double) BAR_WIDTH) / max) : 0;
        int empty   = BAR_WIDTH - filled;
        double pct  = max > 0 ? (value * 100.0 / max) : 0;
        String bar = "█".repeat(filled) + "░".repeat(empty);
        w.printf("  %-10s |%s  %d (%.1f%%)%n", label, bar, value, pct);
    }

    private void line(PrintWriter w, String ch, int width) {
        w.println(ch.repeat(width));
    }
}

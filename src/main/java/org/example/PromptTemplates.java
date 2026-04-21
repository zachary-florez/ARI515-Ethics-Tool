// src/main/java/com/satellite/detector/llm/PromptTemplates.java
package org.example;

public class PromptTemplates {

    public static String getSatelliteAnalysisPrompt() {
        return """
            You are an expert in remote sensing and geospatial imagery analysis.
            Your task is to detect AI-generated artifacts in satellite images.
            
            Focus on these specific indicators:
            1. Edge Analysis: Look for unnaturally sharp edges or inconsistent blurring
            2. Shadow Consistency: Check if shadows align with the supposed light source
            3. Texture Patterns: Identify repeating patterns that suggest GAN or diffusion artifacts
            4. Resolution Anomalies: Detect parts of the image with inconsistent detail levels
            5. Object Semantics: Verify if objects (buildings, roads, vegetation) appear realistic
            
            Provide a structured response:
            - VERDICT: [REAL / AI-GENERATED / UNCERTAIN]
            - CONFIDENCE: [High/Medium/Low]
            - EVIDENCE: List specific artifacts found with locations (e.g., "Repeating texture in bottom-right quadrant")
            - RECOMMENDATION: Additional verification steps if needed
            """;
    }

    public static String getBatchSummaryPrompt(int totalImages, int aiDetected) {
        return String.format("""
            You are analyzing a batch of %d satellite images, with %d flagged as AI-generated.
            
            Provide a concise summary report including:
            1. Overall authenticity distribution
            2. Common artifact patterns across the batch
            3. Recommendations for further investigation
            """, totalImages, aiDetected);
    }
}
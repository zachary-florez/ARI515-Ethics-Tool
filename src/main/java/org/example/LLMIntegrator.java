package org.example;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import okhttp3.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

public class LLMIntegrator {
    private static final Logger logger = LoggerFactory.getLogger(LLMIntegrator.class);

    private static final String OPENAI_API_URL = "";
    private static final String OPENAI_VERSION  = "";
    private static final String MODEL              = "";
    private static final int    MAX_TOKENS         = 1024;

    private final OkHttpClient httpClient;
    private final Gson gson;
    private final String apiKey;

    public LLMIntegrator(String apiKey) {
        this.apiKey = apiKey;
        this.httpClient = new OkHttpClient.Builder()
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(60, TimeUnit.SECONDS)
                .build();
        this.gson = new Gson();
    }

    public LLMIntegrator(String apiKey, String ignoredUrl) {
        this(apiKey);
    }

    public String analyzeWithLLM(String imagePath, InferenceEngine.DetectionResult mlResult) {
        if (apiKey == null || apiKey.isBlank()) {
            logger.error("Key not set");
            return "LLM analysis skipped — Key not set";
        }

        String systemPrompt = PromptTemplates.getSatelliteAnalysisPrompt();
        String userMessage  = buildUserMessage(imagePath, mlResult);

        return callAPI(systemPrompt, userMessage);
    }

    private String buildUserMessage(String imagePath, InferenceEngine.DetectionResult mlResult) {
        return String.format(
                "Image: %s\n\n"
                        + "ML Model Result: %s\n"
                        + "ML Confidence:   %.1f%%\n\n"
                        + "The ML model confidence is in the uncertain range (40-80%%), "
                        + "meaning it could not make a high-confidence determination. "
                        + "Please analyze this satellite image for AI-generated artifacts including:\n"
                        + "- Unnatural edge sharpness or blurring\n"
                        + "- Inconsistent shadow casting\n"
                        + "- Repetitive or glitched texture patterns\n"
                        + "- Anomalies in spatial resolution\n"
                        + "- Unrealistic object placement or geometry",
                imagePath,
                mlResult.isAIGenerated() ? "AI-GENERATED" : "REAL",
                mlResult.getConfidence() * 100
        );
    }

    private String callAPI(String systemPrompt, String userContent) {
        JsonObject requestBody = new JsonObject();
        requestBody.addProperty("model",      MODEL);
        requestBody.addProperty("max_tokens", MAX_TOKENS);
        requestBody.addProperty("system",     systemPrompt);

        JsonArray messages = new JsonArray();
        JsonObject userMessage = new JsonObject();
        userMessage.addProperty("role",    "user");
        userMessage.addProperty("content", userContent);
        messages.add(userMessage);
        requestBody.add("messages", messages);

        Request request = new Request.Builder()
                .url(OPENAI_API_URL)
                .addHeader("x-api-key",         apiKey)
                .addHeader("openai-version",  OPENAI_VERSION)
                .addHeader("content-type",       "application/json")
                .post(RequestBody.create(
                        gson.toJson(requestBody),
                        MediaType.parse("application/json")))
                .build();

        try (Response response = httpClient.newCall(request).execute()) {
            if (!response.isSuccessful() || response.body() == null) {
                logger.error("API error: HTTP {}", response.code());
                if (response.code() == 401) {
                    logger.error("401 Unauthorized — check your Key is correct");
                } else if (response.code() == 429) {
                    logger.error("429 Rate limited — too many requests, wait and retry");
                }
                return "LLM analysis failed — HTTP " + response.code();
            }

            String responseBody = response.body().string();
            return parseResponse(responseBody);

        } catch (IOException e) {
            logger.error("Failed to call API", e);
            return "LLM analysis error: " + e.getMessage();
        }
    }

    private String parseResponse(String responseBody) {
        try {
            JsonObject json    = gson.fromJson(responseBody, JsonObject.class);
            JsonArray  content = json.getAsJsonArray("content");

            if (content == null || content.isEmpty()) {
                logger.error("Empty content in response: {}", responseBody);
                return "LLM analysis failed — empty response";
            }

            JsonObject firstBlock = content.get(0).getAsJsonObject();
            if (!"text".equals(firstBlock.get("type").getAsString())) {
                return "LLM analysis failed — unexpected content type";
            }

            return firstBlock.get("text").getAsString();

        } catch (Exception e) {
            logger.error("Failed to parse response: {}", responseBody, e);
            return "LLM analysis failed — could not parse response";
        }
    }
}
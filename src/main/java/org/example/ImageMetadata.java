package org.example;

import com.opencsv.bean.CsvBindByName;

public class ImageMetadata {

    @CsvBindByName(column = "image_id")
    private String imageId;

    @CsvBindByName(column = "filename")
    private String filename;

    @CsvBindByName(column = "source")
    private String source;

    @CsvBindByName(column = "label")
    private String label; // "real" or "ai_generated"

    @CsvBindByName(column = "resolution")
    private String resolution;

    @CsvBindByName(column = "date_collected")
    private String dateCollected;

    @CsvBindByName(column = "llm_analyzed")
    private boolean llmAnalyzed;

    public String getImageId() { return imageId; }
    public void setImageId(String imageId) { this.imageId = imageId; }

    public String getFilename() { return filename; }
    public void setFilename(String filename) { this.filename = filename; }

    public String getSource() { return source; }
    public void setSource(String source) { this.source = source; }

    public String getLabel() { return label; }
    public void setLabel(String label) { this.label = label; }

    public String getResolution() { return resolution; }
    public void setResolution(String resolution) { this.resolution = resolution; }

    public String getDateCollected() { return dateCollected; }
    public void setDateCollected(String dateCollected) { this.dateCollected = dateCollected; }

    public boolean isLlmAnalyzed() { return llmAnalyzed; }
    public void setLlmAnalyzed(boolean llmAnalyzed) { this.llmAnalyzed = llmAnalyzed; }

    @Override
    public String toString() {
        return String.format("ImageMetadata{id='%s', filename='%s', label='%s'}",
                imageId, filename, label);
    }
}
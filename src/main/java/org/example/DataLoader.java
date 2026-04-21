package org.example;

import com.opencsv.bean.CsvToBeanBuilder;
import com.opencsv.bean.StatefulBeanToCsv;
import com.opencsv.bean.StatefulBeanToCsvBuilder;
import com.opencsv.exceptions.CsvDataTypeMismatchException;
import com.opencsv.exceptions.CsvRequiredFieldEmptyException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Writer;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class DataLoader {
    private static final Logger logger = LoggerFactory.getLogger(DataLoader.class);
    private final Path dataRoot;
    private List<ImageMetadata> metadataList;

    public DataLoader(String dataRootPath) {
        this.dataRoot = Paths.get(dataRootPath);
        this.metadataList = new ArrayList<>();
    }

    public void loadMetadata(String csvFilePath) throws IOException {
        try (FileReader reader = new FileReader(csvFilePath)) {
            metadataList = new CsvToBeanBuilder<ImageMetadata>(reader)
                    .withType(ImageMetadata.class)
                    .build()
                    .parse();
            logger.info("Loaded {} image metadata entries", metadataList.size());
        }
    }

    public List<ImageMetadata> getImagesByLabel(String label) {
        return metadataList.stream()
                .filter(img -> label.equals(img.getLabel()))
                .collect(Collectors.toList());
    }

    public List<ImageMetadata> getUnanalyzedImages() {
        return metadataList.stream()
                .filter(img -> !img.isLlmAnalyzed())
                .collect(Collectors.toList());
    }

    public Path getImagePath(ImageMetadata metadata) {
        return dataRoot.resolve("raw").resolve(metadata.getFilename());
    }

    public void updateMetadata(ImageMetadata updatedMetadata) {
        // Find and replace in list
        for (int i = 0; i < metadataList.size(); i++) {
            if (metadataList.get(i).getImageId().equals(updatedMetadata.getImageId())) {
                metadataList.set(i, updatedMetadata);
                break;
            }
        }
    }

    public void saveUpdatedMetadata(String csvFilePath) throws IOException, CsvDataTypeMismatchException, CsvRequiredFieldEmptyException {
        try (Writer writer = new FileWriter(csvFilePath)) {
            StatefulBeanToCsv<ImageMetadata> beanWriter = new StatefulBeanToCsvBuilder<ImageMetadata>(writer).build();
            beanWriter.write(metadataList);
        }
    }

    public List<ImageMetadata> getAllMetadata() {
        return new ArrayList<>(metadataList);
    }
}
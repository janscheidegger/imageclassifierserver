package ch.mobi.kis.imageclassifierserver;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Result {

    private Map<String, Float> classificationResults;

    public Result(Map<String, Float> predictionMap) {
        this.classificationResults = predictionMap;
    }

    public Map<String, Float> getClassificationResults() {
        return classificationResults;
    }

    public void setClassificationResults(Map<String, Float> classificationResults) {
        this.classificationResults = classificationResults;
    }

    public String getResults(int numberOfResults) {
        if (classificationResults == null) throw new RuntimeException("Need Classification Results");
        return classificationResults.entrySet().stream()
                .sorted(Map.Entry.comparingByValue())
                .limit(numberOfResults)
                .map(e -> e.getKey()+": "+e.getValue())
                .collect(Collectors.joining("\n"));
    }
}

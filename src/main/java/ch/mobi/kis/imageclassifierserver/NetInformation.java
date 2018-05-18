package ch.mobi.kis.imageclassifierserver;

import java.io.Serializable;
import java.util.List;

public class NetInformation implements Serializable {

    private List<String> labels;

    public List<String> getLabels() {
        return labels;
    }

    public void setLabels(List<String> labels) {
        this.labels = labels;
    }
}

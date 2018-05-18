package ch.mobi.kis.imageclassifierserver;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Objects;
import java.util.Random;

public class ImageDataSetIteratorCreator {

    private static final Logger LOG = LoggerFactory.getLogger(ImageDataSetIteratorCreator.class);

    private int width;
    private int height;
    private int channels;
    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private InputSplit trainData,testData;
    private static final Random rng  = new Random(13);
    private int numClasses;


    public ImageDataSetIteratorCreator(int width, int height, int channels, int batchSize) {
        this.width = width;
        this.height = height;
        this.channels = channels;
    }

    public ImageDataSetIteratorCreator setup(File path, int trainPercentage) {
        this.numClasses = Objects.requireNonNull(path.listFiles(File::isDirectory)).length;
        FileSplit filesInDir = new FileSplit(path, allowedExtensions, rng);
        BalancedPathFilter pathFilter = new BalancedPathFilter(rng, allowedExtensions, labelMaker);
        if (trainPercentage >= 100) {
            throw new IllegalArgumentException("Percentage of data set aside for training has to be less than 100%. Test percentage = 100 - training percentage, has to be greater than 0");
        }
        InputSplit[] filesInDirSplit = filesInDir.sample(pathFilter, trainPercentage, 100-trainPercentage);
        this.trainData = filesInDirSplit[0];
        this.testData = filesInDirSplit[1];
        return this;
    }

    public DataSetIterator getTrainData(int batchSize) throws IOException {
        return makeIterator(trainData, batchSize);
    }

    public DataSetIterator getTestData(int batchSize) throws IOException {
        return makeIterator(testData, batchSize);
    }

    private DataSetIterator makeIterator(InputSplit split, int batchSize) throws IOException {
        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        recordReader.initialize(split);
        DataSetIterator iter = new RecordReaderDataSetIterator(recordReader, batchSize, 1, this.numClasses);
        iter.setPreProcessor( new VGG16ImagePreProcessor());
        return iter;
    }

    public int getNumClasses() {
        return numClasses;
    }

}

package ch.mobi.kis.imageclassifierserver;

import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.FrameFilter;
import org.datavec.api.util.ClassPathResource;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Classifier {

    private static final Logger LOGGER = LoggerFactory.getLogger(Classifier.class);

    private ComputationGraph computationGraph;
    private String name;


    private static Result decodePredictions(INDArray predictions, List<String> labels) {

        Map<String, Float> predictionMap = new HashMap<>();
        StringBuilder predictionDescription = new StringBuilder();
        int[] top5 = new int[labels.size()];
        float[] top5Prob = new float[labels.size()];
        int i = 0;

        for (int batch = 0; batch < predictions.size(0); ++batch) {
            predictionDescription.append("Predictions for batch ");
            if (predictions.size(0) > 1) {
                predictionDescription.append(String.valueOf(batch));
            }

            predictionDescription.append(" :");

            for (INDArray currentBatch = predictions.getRow(batch).dup(); i < labels.size(); ++i) {
                top5[i] = Nd4j.argMax(currentBatch, 1).getInt(0, 0);
                top5Prob[i] = currentBatch.getFloat(batch, top5[i]);
                currentBatch.putScalar(0, top5[i], 0.0D);

                predictionMap.put(labels.get(top5[i]), top5Prob[i] * 100.0f);


                predictionDescription.append("\n\t").append(String.format("%3f", top5Prob[i] * 100.0F)).append("%, ").append(labels.get(top5[i]));
            }
        }

        return new Result(predictionMap);
    }

    public void trainAndEvaluate(File trainFolder) throws IOException {
        this.name = trainFolder.getName();
        int seed = 1234;
        final int trainPerc = 80;
        final int batchSize = 10;
        NetInformation netInformation = new NetInformation();

        ZooModel zooModel = new VGG16();
        ComputationGraph pretrainedNet = (ComputationGraph) zooModel.initPretrained();

        LOGGER.info("Creating Datasets");
        ImageDataSetIteratorCreator dataSetIteratorCreator = new ImageDataSetIteratorCreator(244, 244, 3, 100)
                .setup(trainFolder, trainPerc);

        DataSetIterator trainIter = dataSetIteratorCreator.getTrainData(batchSize);
        DataSetIterator testIter = dataSetIteratorCreator.getTestData(batchSize);


        LOGGER.info(pretrainedNet.summary());

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .learningRate(5e-5)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS)
                .seed(seed)
                .build();

        this.computationGraph = new TransferLearning.GraphBuilder(pretrainedNet)
                .fineTuneConfiguration(fineTuneConf)
                .setFeatureExtractor("fc2")
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                                .nIn(4096).nOut(dataSetIteratorCreator.getNumClasses())
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX).build(), "fc2")
                .build();

        attachUIServerotNet(this.computationGraph);

        LOGGER.info(this.computationGraph.summary());

        System.out.println(trainIter.getLabels());
        netInformation.setLabels(trainIter.getLabels());
        Evaluation eval;
        eval = computationGraph.evaluate(testIter);
        LOGGER.info("Eval stats BEFORE fit.....");
        LOGGER.info(eval.stats() + "\n");
        testIter.reset();

        int iter = 0;
        while (trainIter.hasNext()) {
            computationGraph.fit(trainIter.next());
            if (iter % 10 == 0) {
                LOGGER.info("Evaluate model at iter " + iter + " ....");
                eval = computationGraph.evaluate(testIter);
                LOGGER.info(eval.stats());
                testIter.reset();
            }
            iter++;
        }

        LOGGER.info("Model build complete");
        LOGGER.info("Saving Model configuration");
        boolean mkdir = new File(trainFolder.getName()).mkdir();

        if(mkdir) {
            try (ObjectOutputStream oos = new ObjectOutputStream(
                    new FileOutputStream(new File(this.name+"/netconfig.ser")))) {
                oos.writeObject(netInformation);
            }
            File locationToSave = new File(this.name + "/model.zip");
            ModelSerializer.writeModel(computationGraph, locationToSave, true);
        }
    }

    public Result test(File fileToTest) throws Exception {

        NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
        INDArray image = loader.asMatrix(fileToTest);

        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(image);


        INDArray[] output = computationGraph.output(false, image);

        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(new File(this.name+"/netconfig.ser")))) {
            NetInformation info = (NetInformation) ois.readObject();
            return decodePredictions(output[0], info.getLabels());
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
            throw e;
        }


    }

    public ComputationGraph loadModel(File file) throws IOException {
        this.name = file.getName();
        computationGraph = ModelSerializer.restoreComputationGraph(file+"/model.zip");
        return computationGraph;
    }

    public void attachUIServerotNet(ComputationGraph model) {
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, score vs. time etc) is to be stored. Here: store in memory.
        StatsStorage statsStorage = new InMemoryStatsStorage();         //Alternative: new FileStatsStorage(File), for saving and loading later

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        //Then add the StatsListener to collect this information from the network, as it trains
        model.setListeners(new StatsListener(statsStorage));
        System.out.println("started UI server");
    }


}

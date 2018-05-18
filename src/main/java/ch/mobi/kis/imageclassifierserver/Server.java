package ch.mobi.kis.imageclassifierserver;

import com.fasterxml.jackson.databind.ObjectMapper;
import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;
import org.datavec.api.util.ClassPathResource;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.net.ServerSocket;
import java.net.Socket;

public class Server {

    private ServerSocket serverSocket;
    private Socket client = new Socket();
    private BufferedReader in;
    private PrintWriter out;
    boolean clientConnected;
    boolean runServer = true;
    Classifier classifier = new Classifier();


    private Server(String[] args) throws IOException {

        ArgumentParser parser = ArgumentParsers.newFor("Server").build()
                .defaultHelp(true)
                .description("Control the Server");

        parser.addArgument("-p", "--port")
                .setDefault(4321)
                .type(Integer.class)
                .help("Specify which Port to use for the server");

        parser.addArgument("-m", "--model")
                .help("Load earlier Model");

        parser.addArgument("-d", "--directory")
        .type(String.class)
        .help("Directory where the training data is located. Must have sub directory for classes");


        Namespace ns = null;
        try {
            ns = parser.parseArgs(args);
        } catch (ArgumentParserException e) {
            parser.handleError(e);
            System.exit(1);
        }
        int port = ns.getInt("port");


        String loadConfig = ns.getString("model");
        if (loadConfig != null) {
            classifier.loadModel(new File(loadConfig));
        }
        String trainDirectory = ns.getString("directory");
        if(trainDirectory != null) {
            classifier.trainAndEvaluate(new ClassPathResource(trainDirectory).getFile());
        }



        try {
            serverSocket = new ServerSocket(port);
            listen();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    // start without parameter
    // -p specify Port
    // -l load Model
    // -d load with Directory to train from

    public static void main(String[] args) throws IOException {
        new Server(args);
    }

    private void listen() throws Exception {
        try {
            System.out.println("Server Ready");
            while (runServer) {
                client = serverSocket.accept();
                if (client.isConnected()) {
                    clientConnected = true;
                    System.out.println("Client Connected");
                    in = new BufferedReader(new InputStreamReader(client.getInputStream()));
                    out = new PrintWriter(new OutputStreamWriter(client.getOutputStream()));
                        String data;
                        while (clientConnected && (data = in.readLine()) != null) {
                            System.out.println("\r\nMessage from Client : " + data);
                            if (data.equals("end")) {
                                client.close();
                                out.close();
                                clientConnected = false;
                            } else if (data.equals("shutdown")) {
                                client.close();
                                serverSocket.close();
                                clientConnected = false;
                                runServer = false;
                            } else if (data.contains("test")) {
                                String fileName = data.substring(data.indexOf(" ")+1);
                                Result result = classifier.test(new ClassPathResource(fileName).getFile());
                                System.out.println(result.getResults(2));
                                ObjectMapper objectMapper = new ObjectMapper();
                                String outputResult = objectMapper.writeValueAsString(result);
                                out.println(outputResult);
                                out.flush();

                            }
                        }


                    System.out.println("Client Disconnected");
                }
            }

        } catch (IOException e) {
            e.printStackTrace();
        }
    }


}

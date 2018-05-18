package ch.mobi.kis.imageclassifierserver;

import net.sourceforge.argparse4j.ArgumentParsers;
import net.sourceforge.argparse4j.inf.ArgumentParser;
import net.sourceforge.argparse4j.inf.ArgumentParserException;
import net.sourceforge.argparse4j.inf.Namespace;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.Socket;
import java.util.Collections;
import java.util.List;

public class Client {

    // -p Port for Server (default 4321)
    // -h hostname (default 127.0.0.1)
    // -t Test against file

    public static void main(String[] args) {

        ArgumentParser parser = ArgumentParsers.newFor("Server").build()
                .defaultHelp(true)
                .description("Control the Server");

        parser.addArgument("-p", "--port")
                .setDefault(4321)
                .type(Integer.class)
                .help("Specify which Port to use for the server");

        parser.addArgument("--host")
                .type(String.class)
                .help("Set the hostname to connect the socket to")
                .setDefault("127.0.0.1");

        parser.addArgument("file").nargs("*")
                .help("Files to predict");



        Namespace ns = null;
        try {
            ns = parser.parseArgs(args);
        } catch (ArgumentParserException e) {
            parser.handleError(e);
            System.exit(1);
        }
        int port = ns.getInt("port") != null ? ns.getInt("port") : 4321;

        List<String> files = ns.getList("file") != null ? ns.getList("file") : Collections.EMPTY_LIST;
        String host = ns.getString("host") != null ? ns.getString("host") : "127.0.0.1";



        try {
            Socket socket = new Socket(host ,port);
            PrintWriter out = new PrintWriter(socket.getOutputStream(),
                    true);
            for(String file : files) {
                out.println("test " + file);
            }
            out.println("end");
            BufferedReader reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            String result;
            while((result = reader.readLine()) != null) {
                System.out.println(result);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }


}

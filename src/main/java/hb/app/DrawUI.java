package hb.app;

import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.stage.Stage;

public class DrawUI extends Application {
    @Override
    public void start(Stage stage) throws Exception {
        FXMLLoader fxmlLoader = new FXMLLoader(DrawUI.class.getResource("drawui.fxml"));
        Scene scene = new Scene(fxmlLoader.load());
        stage.setTitle("Digit Recognizer");
        stage.setScene(scene);
        stage.show();
    }
}

module NN.main {
    requires progressbar;

    requires javafx.graphics;
    requires javafx.fxml;

    requires org.controlsfx.controls;

    opens hb.app to javafx.fxml;
    exports hb.app;
}
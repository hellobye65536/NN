package hb.app;

import javafx.scene.canvas.Canvas;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;

public class DrawUIController {
    public Canvas draw;

    private float[] canvasPixels = new float[Model.IMAGE_SIZE];

    public void initialize() {
        draw.widthProperty().addListener(_observable -> redrawCanvas());
        draw.heightProperty().addListener(_observable -> redrawCanvas());
    }

    public void drawDrag(MouseEvent mouseEvent) {
        System.out.println(mouseEvent);

        redrawCanvas();
    }

    private void redrawCanvas() {
        var gc = draw.getGraphicsContext2D();

        gc.setFill(Color.BLACK);
        gc.fillRect(0, 0, 100, 100);
    }
}

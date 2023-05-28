package hb.app;

import hb.matrix.Matrix;
import javafx.scene.canvas.Canvas;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;
import javafx.scene.transform.Affine;

import java.util.Random;

public class DrawUIController {
    public Canvas draw;

    private Matrix drawPixels = Matrix.zeros(28, 28);

    public void initialize() {
        for (int row = 0; row < drawPixels.rows(); row++) {
            for (int col = 0; col < drawPixels.cols(); col++) {
                drawPixels.set(row, col, 1.0f);
            }
        }

        draw.widthProperty().addListener(_observable -> redrawCanvas());
        draw.heightProperty().addListener(_observable -> redrawCanvas());
    }

    public void drawDrag(MouseEvent mouseEvent) {
        System.out.println(mouseEvent);

        redrawCanvas();
    }

    private void redrawCanvas() {
        final double squareWidth = Math.min(draw.getWidth(), draw.getHeight());

        var gc = draw.getGraphicsContext2D();
        gc.setTransform(new Affine());

        gc.clearRect(0, 0, draw.getWidth(), draw.getHeight());

        gc.translate(
            (draw.getWidth() - squareWidth) / 2,
            (draw.getHeight() - squareWidth) / 2
        );

        gc.scale(squareWidth / 28, squareWidth / 28);

        for (int x = 0; x < 28; x++) {
            for (int y = 0; y < 28; y++) {
                gc.setFill(Color.gray(drawPixels.get(y, x)));
                gc.fillRect(x, y, 1, 1);
            }
        }
    }
}

package hb.app;

import hb.matrix.Matrix;
import javafx.fxml.FXML;
import javafx.geometry.Point2D;
import javafx.scene.canvas.Canvas;
import javafx.scene.input.MouseEvent;
import javafx.scene.paint.Color;
import javafx.scene.transform.Affine;

public class DrawUIController {
    @FXML
    private Canvas draw;

    private Matrix drawPixels = Matrix.zeros(28, 28);
    // used when dragging
    private double prevMouseX, prevMouseY;

    @FXML
    private void initialize() {
        for (int row = 0; row < drawPixels.rows(); row++) {
            for (int col = 0; col < drawPixels.cols(); col++) {
                drawPixels.set(row, col, 1.0f);
            }
        }

        draw.widthProperty().addListener(_observable -> redrawCanvas());
        draw.heightProperty().addListener(_observable -> redrawCanvas());
    }

//    private Point2D transformMouse(Point2D pos) {
//        return transformMouse(pos.getX(), pos.getY());
//    }

    private Point2D transformMouse(double x, double y) {
        final double square_width = Math.min(draw.getWidth(), draw.getHeight());

        final double x_shift = (draw.getWidth() - square_width) / 2;
        final double y_shift = (draw.getHeight() - square_width) / 2;

        return new Point2D((x - x_shift) * 28 / square_width, (y - y_shift) * 28 / square_width);
    }

    @FXML
    private void drawMousePressed(MouseEvent mouseEvent) {
        final Point2D curMouse = transformMouse(mouseEvent.getX(), mouseEvent.getY());
        final double curMouseX = curMouse.getX();
        final double curMouseY = curMouse.getY();

        prevMouseX = curMouseX;
        prevMouseY = curMouseY;

        
    }

    @FXML
    private void drawMouseDragged(MouseEvent mouseEvent) {
        final Point2D curMouse = transformMouse(mouseEvent.getX(), mouseEvent.getY());
        final double curMouseX = curMouse.getX();
        final double curMouseY = curMouse.getY();

        System.out.printf("(%f, %f) -> (%f, %f)\n", prevMouseX, prevMouseY, curMouseX, curMouseY);

        prevMouseX = curMouseX;
        prevMouseY = curMouseY;

//        redrawCanvas();
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

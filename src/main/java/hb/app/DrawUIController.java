package hb.app;

import hb.layers.Layer;
import hb.matrix.Matrix;
import hb.network.Network;
import javafx.fxml.FXML;
import javafx.geometry.Point2D;
import javafx.geometry.Pos;
import javafx.scene.canvas.Canvas;
import javafx.scene.control.Label;
import javafx.scene.input.MouseEvent;
import javafx.scene.layout.GridPane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Rectangle;
import javafx.scene.text.TextAlignment;
import javafx.scene.transform.Affine;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;

import static hb.app.Model.IMAGE_SIZE;
import static hb.app.Model.IMAGE_WIDTH;

public class DrawUIController {
    // The size of the brush when drawing
    private static final double DRAW_BRUSH_SIZE = 2;
    // The size of the digit bar at its maximum width.
    private static final double DIGIT_BAR_MAX_WIDTH = 100;
    // The pixels drawn by the user
    private final Matrix drawPixels = Matrix.zeros(IMAGE_WIDTH, IMAGE_WIDTH);

    // The network to predict digits
    private final Layer[] network = Model.buildNetwork();
    private final Rectangle[] digitBars = new Rectangle[10];
    @FXML
    private Canvas draw;
    @FXML
    private GridPane predictionsPane;
    // Used when dragging, keep track of previous mouse position
    private double prevMouseX, prevMouseY;

    @FXML
    private void initialize() {
        // build predictions chart
        for (int i = 0; i < 10; i++) {
            Label digitLabel = new Label(i + " - ");
            digitLabel.setPrefWidth(40);
            digitLabel.setAlignment(Pos.CENTER_RIGHT);
            digitLabel.setTextAlignment(TextAlignment.RIGHT);

            Rectangle digitBar = new Rectangle(0, 18, Color.BLACK);

            predictionsPane.add(digitLabel, 0, i);
            predictionsPane.add(digitBar, 1, i);
            digitBars[i] = digitBar;
        }

        resetCanvas();
        loadWeights();
        calculatePrediction();

        // make canvas automatically resize
        draw.widthProperty().addListener(_observable -> redrawCanvas());
        draw.heightProperty().addListener(_observable -> redrawCanvas());
    }

    /**
     * Loads weights for the neural network from the file <code>./trained_weights</code>
     */
    private void loadWeights() {
        try (InputStream weightIn = new BufferedInputStream(new FileInputStream("trained_weights"))) {
            Model.loadWeights(network, weightIn);
        } catch (IOException e) {
            System.err.println("Error loading trained weights");
            e.printStackTrace();
        }
    }

    @FXML
    private void resetCanvas() {
        for (int row = 0; row < drawPixels.rows(); row++) {
            for (int col = 0; col < drawPixels.cols(); col++) {
                drawPixels.set(row, col, 0.0f);
            }
        }

        redrawCanvas();
        calculatePrediction();
    }

    /**
     * Transforms a mouse coordinate in the draw canvas to pixel space (such that top left is <code>(0, 0)</code> and
     * bottom right is <code>(28, 28)</code>
     */
    private Point2D transformMouse(double x, double y) {
        final double square_width = Math.min(draw.getWidth(), draw.getHeight());

        final double x_shift = (draw.getWidth() - square_width) / 2;
        final double y_shift = (draw.getHeight() - square_width) / 2;
        final double scale = IMAGE_WIDTH / square_width;

        return new Point2D(
            (x - x_shift) * scale,
            (y - y_shift) * scale
        );
    }

    /**
     * Draw when a mouse clicks
     */
    @FXML
    private void drawMousePressed(MouseEvent mouseEvent) {
        final Point2D curMouse = transformMouse(mouseEvent.getX(), mouseEvent.getY());
        final double curMouseX = curMouse.getX();
        final double curMouseY = curMouse.getY();

        drawPoint(curMouseX, curMouseY);

        prevMouseX = curMouseX;
        prevMouseY = curMouseY;
    }

    /**
     * Draw when a mouse drags
     */
    @FXML
    private void drawMouseDragged(MouseEvent mouseEvent) {
        final Point2D curMouse = transformMouse(mouseEvent.getX(), mouseEvent.getY());
        final double curMouseX = curMouse.getX();
        final double curMouseY = curMouse.getY();

        drawLine(prevMouseX, prevMouseY, curMouseX, curMouseY);

        prevMouseX = curMouseX;
        prevMouseY = curMouseY;
    }

    /**
     * Draw a brush point
     * <p>
     * Checks nearby pixels, and calculates the squared distance from the center of each pixel and the position of the
     * mouse
     */
    private void drawPoint(final double mouseX, final double mouseY) {
        final int mousePixelX = (int) mouseX;
        final int mousePixelY = (int) mouseY;

        // check 9 pixels around mouse position
        for (int dx = -1; dx <= 1; dx++) {
            final int pixelX = mousePixelX + dx;
            if (!(0 <= pixelX && pixelX < IMAGE_WIDTH)) continue;

            for (int dy = -1; dy <= 1; dy++) {
                final int pixelY = mousePixelY + dy;
                if (!(0 <= pixelY && pixelY < IMAGE_WIDTH)) continue;

                final double pixelCenterX = pixelX + 0.5;
                final double pixelCenterY = pixelY + 0.5;

                final double squaredDist = squaredDistance(mouseX, mouseY, pixelCenterX, pixelCenterY);

                if (squaredDist >= DRAW_BRUSH_SIZE)
                    continue;

                drawPixel(pixelX, pixelY, squaredDist / DRAW_BRUSH_SIZE);
            }
        }

        redrawCanvas();
        calculatePrediction();
    }

    /**
     * Draw a brush line
     * <p>
     * Checks every pixel, and calculates the squared distance from the center of each pixel and the line segment drawn
     */
    private void drawLine(
        final double prevMouseX,
        final double prevMouseY,
        final double curMouseX,
        final double curMouseY
    ) {
        for (int pixelX = 0; pixelX < IMAGE_WIDTH; pixelX++) {
            for (int pixelY = 0; pixelY < IMAGE_WIDTH; pixelY++) {
                final double pixelCenterX = pixelX + 0.5;
                final double pixelCenterY = pixelY + 0.5;

                final double squaredDist = squaredDistanceLineSegment(prevMouseX, prevMouseY, curMouseX, curMouseY,
                    pixelCenterX, pixelCenterY);

                if (squaredDist >= DRAW_BRUSH_SIZE)
                    continue;

                drawPixel(pixelX, pixelY, squaredDist / DRAW_BRUSH_SIZE);
            }
        }

        redrawCanvas();
        calculatePrediction();
    }

    /**
     * Finds the shortest squared distance from point <code>(px, py)</code> to line segment <code>((ax, ay), (bx,
     * by))</code>
     */
    private double squaredDistanceLineSegment(
        final double ax,
        final double ay,
        final double bx,
        final double by,
        final double px,
        final double py
    ) {
        // taken from: https://stackoverflow.com/a/1501725

        final Point2D a = new Point2D(ax, ay),
            b = new Point2D(bx, by),
            p = new Point2D(px, py);

        final double segmentLengthSquared = squaredDistance(ax, ay, bx, by);
        if (segmentLengthSquared == 0)
            return squaredDistance(ax, ay, px, py);

        final double t = Math.min(Math.max(p.subtract(a).dotProduct(b.subtract(a)) / segmentLengthSquared, 0), 1);
        final Point2D proj = a.add(b.subtract(a).multiply(t));
        return squaredDistance(px, py, proj.getX(), proj.getY());
    }

    /**
     * Finds the squared distance between <code>(ax, ay)</code> and <code>(bx, by)</code>
     */
    private double squaredDistance(final double ax, final double ay, final double bx, final double by) {
        final double dx = ax - bx;
        final double dy = ay - by;

        return dx * dx + dy * dy;
    }

    /**
     * Fill a pixel <code>(x, y)</code> with black using some alpha
     */
    private void drawPixel(int x, int y, double alpha) {
        final double prevPixel = drawPixels.get(y, x);
        drawPixels.set(y, x, (float) (prevPixel * alpha + (1 - alpha)));
    }

    /**
     * Redraw the canvas using <code>drawPixels</code>
     */
    private void redrawCanvas() {
        final double squareWidth = Math.min(draw.getWidth(), draw.getHeight());

        final var gc = draw.getGraphicsContext2D();
        gc.setTransform(new Affine());

        gc.clearRect(0, 0, draw.getWidth(), draw.getHeight());

        gc.translate(
            (draw.getWidth() - squareWidth) / 2,
            (draw.getHeight() - squareWidth) / 2
        );

        gc.scale(squareWidth / IMAGE_WIDTH, squareWidth / IMAGE_WIDTH);

        for (int x = 0; x < IMAGE_WIDTH; x++) {
            for (int y = 0; y < IMAGE_WIDTH; y++) {
                gc.setFill(Color.gray(1 - drawPixels.get(y, x)));
                gc.fillRect(x, y, 1, 1);
            }
        }
    }

    /**
     * Predict the current digit using the network, and redraw the chart
     */
    private void calculatePrediction() {
        Matrix prediction = Network.runNetwork(network, drawPixels.reshape(IMAGE_SIZE, 1));

        int mostLikely = 0;
        float mostLikelyProbability = 0;
        for (int i = 0; i < 10; i++) {
            Rectangle digitBar = digitBars[i];
            float probability = prediction.get(i, 0);
            digitBar.setWidth(DIGIT_BAR_MAX_WIDTH * probability);
            digitBar.setFill(Color.BLACK);

            if (probability > mostLikelyProbability) {
                mostLikelyProbability = probability;
                mostLikely = i;
            }
        }

        // highlight highest probability in red
        digitBars[mostLikely].setFill(Color.RED);
    }
}

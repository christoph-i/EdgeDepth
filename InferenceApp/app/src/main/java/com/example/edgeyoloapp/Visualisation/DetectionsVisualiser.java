package com.example.edgeyoloapp.Visualisation;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import com.example.edgeyoloapp.Networks.Recognition;

import java.util.List;

public class DetectionsVisualiser {
    public static Bitmap visDetections(Bitmap image, List<Recognition> recognitions, boolean showConf) {
        Bitmap resultImage = image.copy(image.getConfig(), true);
        Canvas canvas = new Canvas(resultImage);
        Paint paint = new Paint();

        for (Recognition recognition : recognitions) {
            int classId = recognition.getType();

            int leftAbs = (int) (recognition.getLeft() * image.getWidth());
            int topAbs = (int) (recognition.getTop() * image.getHeight());
            int rightAbs = (int) (recognition.getRight() * image.getWidth());
            int bottomAbs = (int) (recognition.getBottom() * image.getHeight());

            int color = Color.rgb(
                    (int) (BboxColorLookup.getColor(classId)[0] * 255),
                    (int) (BboxColorLookup.getColor(classId)[1] * 255),
                    (int) (BboxColorLookup.getColor(classId)[2] * 255)
            );
            //int color = Color.rgb(150,0,120);

            paint.setColor(color);
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(5);
            canvas.drawRect(leftAbs, topAbs, rightAbs, bottomAbs, paint);

            paint.setStyle(Paint.Style.FILL);
//            int textBackgroundColor = Color.argb(179,
//                    (int) (Recognition.COLORS[classId][0] * 255),
//                    (int) (Recognition.COLORS[classId][1] * 255),
//                    (int) (Recognition.COLORS[classId][2] * 255)
//            );
            String text = showConf ?
                    recognition.getTitle() + ":" + String.format("%.1f%%", recognition.getConfidence() * 100) :
                    recognition.getTitle();
            text = text + " | Depth: " + recognition.getDepthPrediction();

            int textColor = Color.WHITE;
//            if (recognition.getColorMean() < 0.5) {
//                textColor = Color.BLACK;
//            }

            int textBackgroundColor = Color.BLACK;
            paint.setColor(textBackgroundColor);
//            Rect backgroundRect = new Rect(leftAbs, topAbs + 1, leftAbs + (int) (1.5 * textRect.width()) + 1, topAbs + textRect.height());
//            canvas.drawRect(backgroundRect, paint);

            paint.setColor(textColor);
            paint.setTextSize(0.03f * image.getWidth());
            //canvas.drawText(text, leftAbs, topAbs + textRect.height(), paint);
            canvas.drawText(text, leftAbs, topAbs , paint);
        }

        return resultImage;
    }
}

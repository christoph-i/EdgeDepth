package com.example.edgeyoloapp;

import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import com.example.edgeyoloapp.Visualisation.BboxColorLookup;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

public class ImageUtils {
    static public class CropAttributes {

        public float top;
        public float bottom;

        public CropAttributes(float top, float bottom) {
            this.top = top;
            this.bottom = bottom;
        }
    }

    static public class ResizeAttributes {

        public int width;
        public int height;

        public ResizeAttributes(int width, int height) {
            this.width = width;
            this.height = height;
        }
    }

    static public class NormalizationAttributes {

        public float mean;
        public float std;

        public NormalizationAttributes(float mean, float std) {
            this.mean = mean;
            this.std = std;
        }
    }

    static public class ImageBuffer {

        public int[] intValues;
        public ByteBuffer imgData;

        public ImageBuffer(int width, int height, int depth, int bytesPerChannel) {

            int square = width * height;
            intValues = new int[square];
            imgData = ByteBuffer.allocateDirect(square * depth * bytesPerChannel);
            imgData.order(ByteOrder.nativeOrder());
        }
    }

    public static int getCroppedImageHeight(int height, CropAttributes crop) {

        return (int)(height - (height * crop.top) - (height * crop.bottom));
    }

    public static Bitmap crop(Bitmap image, CropAttributes crop) {

        int width = image.getWidth();
        int height = image.getHeight();

        return Bitmap.createBitmap(
                image, 0, (int) (height * crop.top), width, getCroppedImageHeight(height, crop)
        );
    }

    public static Bitmap resize(Bitmap image, ResizeAttributes resize) {

        return Bitmap.createScaledBitmap(
                image, resize.width, resize.height, false
        );
    }

    public static void normalizeAndFillBuffer(ImageBuffer buffer, NormalizationAttributes norm) {

        buffer.imgData.rewind();

        for (int pixel : buffer.intValues) {
            buffer.imgData.putFloat((((pixel >> 16) & 0xFF) - norm.mean) / norm.std); // R
            buffer.imgData.putFloat((((pixel >>  8) & 0xFF) - norm.mean) / norm.std); // G
            buffer.imgData.putFloat((((pixel      ) & 0xFF) - norm.mean) / norm.std); // B
        }
    }

    public static void fillBuffer(ImageBuffer buffer) {
        buffer.imgData.rewind();

        for (int pixel : buffer.intValues) {
            buffer.imgData.put((byte) ((pixel >> 16) & 0xFF)); // R
            buffer.imgData.put((byte) ((pixel >>  8) & 0xFF)); // G
            buffer.imgData.put((byte) ((pixel      ) & 0xFF)); // B
        }
    }

    public static void fillBufferByte(int[] imageData, ByteBuffer outBuf) {
        outBuf.rewind();
        for (int pixel : imageData) {
            outBuf.put((byte)((pixel >> 16) & 0xFF)); // R
            outBuf.put((byte)((pixel >>  8) & 0xFF)); // G
            outBuf.put((byte)((pixel      ) & 0xFF)); // B
        }
    }

    public static void normalize(Bitmap image, float normValue, float shiftValue) {

    }


}

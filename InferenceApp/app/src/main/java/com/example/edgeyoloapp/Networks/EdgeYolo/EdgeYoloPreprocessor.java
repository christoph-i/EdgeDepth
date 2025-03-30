package com.example.edgeyoloapp.Networks.EdgeYolo;

import android.graphics.Bitmap;
import android.provider.ContactsContract;

import com.example.edgeyoloapp.ImageUtils;

import org.apache.commons.math3.geometry.spherical.twod.Edge;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class EdgeYoloPreprocessor {


    public FloatBuffer preprocessImage(Bitmap image, ImageUtils.ResizeAttributes resizeAttributes) {
        long t0 = System.currentTimeMillis();
        // image = ImageUtils.crop(image, new ImageUtils.CropAttributes(1.0f/3.0f, 0.0f));
        image = ImageUtils.resize(image, resizeAttributes);
        int square = EdgeYolo.INPUT_SIZE * EdgeYolo.INPUT_SIZE;
        int[] intValues = new int[square];
        ByteBuffer buffer = ByteBuffer.allocateDirect(4 * EdgeYolo.INPUT_SIZE * EdgeYolo.INPUT_SIZE * 3);
        buffer.order(ByteOrder.nativeOrder());

        // fill int buffer
        image.getPixels(
                intValues, 0, image.getWidth(),
                0, 0, image.getWidth(), image.getHeight()
        );

        buffer.rewind();

        long t1 = System.currentTimeMillis();

        long t00 = System.currentTimeMillis();
        for (int pixel : intValues) {
            buffer.putFloat((float) ((pixel      ) & 0xFF)); // B
//            buffer.putFloat((float) ((pixel >>  8) & 0xFF)); // G
//            buffer.putFloat((float) ((pixel >> 16) & 0xFF)); // R
        }
        for (int pixel : intValues) {
            buffer.putFloat((float) ((pixel >>  8) & 0xFF)); // G
        }
        for (int pixel : intValues) {
            buffer.putFloat((float) ((pixel >> 16) & 0xFF)); // R
        }
        long t11 = System.currentTimeMillis();
        System.out.println("Part 2: " + Math.abs(t11-t00));

        long t000 = System.currentTimeMillis();
        buffer.rewind();
        FloatBuffer inputBuffer = buffer.asFloatBuffer();
        inputBuffer.rewind();
        long t111 = System.currentTimeMillis();
        System.out.println("Part 3: " + Math.abs(t111-t000));

        return inputBuffer;
    }

}

package com.example.edgeyoloapp.Networks.EdgeYolo;

import android.content.Context;
import android.graphics.Bitmap;

import com.example.edgeyoloapp.ImageUtils;
import com.example.edgeyoloapp.Networks.Recognition;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.OptionalDouble;
import java.util.concurrent.TimeUnit;

public class EdgeYoloDepth extends EdgeYolo{

    public static final int INPUT_SIZE = 384;
    public static final int NUM_DETECTIONS = 3024;
    public static final String MODEL_FILE = "003_depth_addition_400_epochs_best_384x384_batch1.tflite";

    public static final boolean useGPU = true; // IF false NNAPI will be used

    private Context context;
    private ImageProcessor imageProcessor;

    private EdgeYoloPreprocessor preprocessor;
    private EdgeYoloPostprocessor postprocessor;

    public static final float MINIMUM_CONFIDENCE = 0.5f;
    public static final float NMS_IOU_THRESHOLD = 0.5f;

    private final ImageUtils.ResizeAttributes resizeAttributes = new ImageUtils.ResizeAttributes(INPUT_SIZE, INPUT_SIZE);
    private final ImageUtils.NormalizationAttributes normalizationAttributes = new ImageUtils.NormalizationAttributes(0.f, 255.f);

    public static final List<String> LABELS = Arrays.asList(
            "traffic_sign",
            "vehicle"
    );
    public static final int valuesPerDetectin = 5 + LABELS.size() + 1; // +1 for extra depth output
    public EdgeYoloDepth(Context context) {
        super(context);
        this.context=context;

        loadModelInterpreter(MODEL_FILE);

        this.outputProbabilityBuffer = TensorBuffer.createFixedSize(new int[]{1, NUM_DETECTIONS, valuesPerDetectin}, DataType.FLOAT32);

        this.preprocessor = new EdgeYoloPreprocessor();
        this.postprocessor = new EdgeYoloPostprocessor();

    }


    private  List<Recognition> postprocess(float[] output) {
        List<Recognition> processedResults = this.postprocessor.toRecognitions(output, valuesPerDetectin, true);
        return this.postprocessor.nms(processedResults);
    }



    public List<Recognition> runInference(Bitmap image) {

        long t0Pre = System.currentTimeMillis();
        FloatBuffer input = preprocessor.preprocessImage(image, this.resizeAttributes);
        long t1Pre = System.currentTimeMillis();

        long t0Infer = System.currentTimeMillis();
        this.interpreter.run(input, this.outputProbabilityBuffer.getBuffer());
        long t1Infer = System.currentTimeMillis();

        long t0Post = System.currentTimeMillis();
        List<Recognition> results = postprocess(this.outputProbabilityBuffer.getFloatArray());
        long t1Post = System.currentTimeMillis();

        this.execTimesPreNanos.add(Math.abs(t1Pre - t0Pre));
        this.execTimesRawModelNanos.add(interpreter.getLastNativeInferenceDurationNanoseconds());
        this.execTimesPostNanos.add(Math.abs(t1Post - t0Post));

        System.out.println("PREEE: " + Math.abs(t0Pre-t1Pre));
        System.out.println("INFERRR: " + Math.abs(t0Infer-t1Infer));
        System.out.println("POSTT: " + Math.abs(t0Post-t1Post));

        return results;
    }

}

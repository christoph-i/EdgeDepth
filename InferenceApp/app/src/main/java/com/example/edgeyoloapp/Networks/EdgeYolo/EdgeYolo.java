package com.example.edgeyoloapp.Networks.EdgeYolo;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.media.tv.TableRequest;

import com.example.edgeyoloapp.ImageUtils;
import com.example.edgeyoloapp.Networks.Recognition;

import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Delegate;
import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegateImpl;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;
import java.util.concurrent.TimeUnit;

public class EdgeYolo {

    public static final int INPUT_SIZE = 384;
    public static final int NUM_DETECTIONS = 3024;
    //TODO provide converted kitti models  public repo
    public static final String MODEL_FILE = "YOUR_CONVERTED_MODEL.tflite";

    public static final boolean useGPU = true; // IF false NNAPI will be used

    private Context context;
    protected InterpreterApi interpreter;
    private ImageProcessor imageProcessor;
    private TensorImage inputTensorImage;
    protected TensorBuffer outputProbabilityBuffer;
    private EdgeYoloPreprocessor preprocessor;
    private EdgeYoloPostprocessor postprocessor;

    protected List<Long> execTimesPreNanos;
    protected List<Long> execTimesPostNanos;
    protected List<Long> execTimesRawModelNanos;

    protected final ImageUtils.ImageBuffer imageBuffer;
    public static final float MINIMUM_CONFIDENCE = 0.5f;
    public static final float NMS_IOU_THRESHOLD = 0.5f;

    public static final boolean REDUCE_CLASSES_TO_VEHICLE_SIGN = true;

    private final ImageUtils.ResizeAttributes resizeAttributes = new ImageUtils.ResizeAttributes(INPUT_SIZE, INPUT_SIZE);
    private final ImageUtils.NormalizationAttributes normalizationAttributes = new ImageUtils.NormalizationAttributes(0.f, 255.f);

    public static final List<String> LABELS = Arrays.asList(
            "Sign",
            "Beacon",
            "Road Arrow",
            "Traffic Light",
            "Vehicle front",
            "Vehicle rear",
            "Person",
            "Bicycle",
            "Vehicle side",
            "Roadwork Fence"
    );
    public static final int valuesPerDetectin = 5 + LABELS.size();
    public EdgeYolo(Context context) {
        this.context=context;
        loadModelInterpreter(MODEL_FILE);

        this.imageProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(INPUT_SIZE, INPUT_SIZE, ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(0.0f, 255.0f))
                        .build();
        this.inputTensorImage = new TensorImage(DataType.FLOAT32);

        imageBuffer = new ImageUtils.ImageBuffer(INPUT_SIZE, INPUT_SIZE, 3, 4);


        this.outputProbabilityBuffer = TensorBuffer.createFixedSize(new int[]{1, NUM_DETECTIONS, valuesPerDetectin}, DataType.FLOAT32);

        this.preprocessor = new EdgeYoloPreprocessor();
        this.postprocessor = new EdgeYoloPostprocessor();

        this.execTimesPreNanos = new ArrayList<>();
        this.execTimesRawModelNanos = new ArrayList<>();
        this.execTimesPostNanos = new ArrayList<>();

    }

    void loadModelInterpreter(String modelFile) {
        MappedByteBuffer tfliteModel;
        try {
            tfliteModel = FileUtil.loadMappedFile(this.context, modelFile);
        } catch (IOException e) {
            //TODO Show error on app screen
            throw new RuntimeException(e);
        }
        InterpreterApi.Options opt = new InterpreterApi.Options();

        Delegate delegate;
        if (this.useGPU) {
            delegate = new GpuDelegate();
        } else {
            delegate = new NnApiDelegate();
        }

        opt.addDelegate(delegate);
        this.interpreter = InterpreterApi.create(tfliteModel, opt);
    }



    private  List<Recognition> postprocess(float[] output) {
        List<Recognition> processedResults = this.postprocessor.toRecognitions(output, valuesPerDetectin, false);
        return this.postprocessor.nms(processedResults);
    }

//    private void preprocess(Bitmap image) {
//        image = ImageUtils.resize(image, resizeAttributes);
//
//        // fill int buffer
//        image.getPixels(
//                imageBuffer.intValues, 0, image.getWidth(),
//                0, 0, image.getWidth(), image.getHeight()
//        );
//
//        // normalize & fill byte buffer
//        ImageUtils.normalizeAndFillBuffer(imageBuffer, normalizationAttributes);
//
//
//        // set image data for prediction
//        //inputs.setImage(imageBuffer.imgData);
//    }



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


    public long getLastExecTimePreMillis() {
        return TimeUnit.NANOSECONDS.toMillis(this.execTimesPreNanos.get(this.execTimesPreNanos.size() - 1));
    }

    public long getLastExecTimeRawModelMillis() {
        return TimeUnit.NANOSECONDS.toMillis(this.execTimesRawModelNanos.get(this.execTimesRawModelNanos.size() - 1));
    }

    public long getLastExecTimePostMillis() {
        return TimeUnit.NANOSECONDS.toMillis(this.execTimesPostNanos.get(this.execTimesPostNanos.size() - 1));
    }

    public long getAvgExecTimePreMillis() {
        OptionalDouble average = this.execTimesPreNanos
                .stream()
                .mapToDouble(a -> a)
                .average();
        return TimeUnit.NANOSECONDS.toMillis(average.isPresent() ? (long) average.getAsDouble() : 0);
    }

    public long getAvgExecTimeRawModelMillis() {
        OptionalDouble average = this.execTimesRawModelNanos
                .stream()
                .mapToDouble(a -> a)
                .average();
        return TimeUnit.NANOSECONDS.toMillis(average.isPresent() ? (long) average.getAsDouble() : 0);
    }

    public long getAvgExecTimePostMillis() {
        OptionalDouble average = this.execTimesPostNanos
                .stream()
                .mapToDouble(a -> a)
                .average();
        return TimeUnit.NANOSECONDS.toMillis(average.isPresent() ? (long) average.getAsDouble() : 0);
    }
}

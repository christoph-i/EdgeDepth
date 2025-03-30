package com.example.edgeyoloapp.ui

import android.graphics.Bitmap
import androidx.camera.core.ExperimentalGetImage
import com.example.edgeyoloapp.Networks.Recognition
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

@ExperimentalGetImage
class DisNetFragment : InferenceFragment() {


    val MODEL_FILE = "DisNet_class_id_xywh.tflite"
    private lateinit var tflite: Interpreter

    companion object {
        fun newInstance() = DisNetFragment()
    }

    override fun loadDepthModel() {
        val assetFileDescriptor = context?.assets?.openFd(MODEL_FILE)
        val fileInputStream = assetFileDescriptor?.createInputStream()
        val fileChannel = fileInputStream?.channel
        val startOffset = assetFileDescriptor?.startOffset ?: 0
        val declaredLength = assetFileDescriptor?.declaredLength ?: 0
        val modelBuffer: MappedByteBuffer = fileChannel!!.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        tflite = Interpreter(modelBuffer)
    }

    override fun inferDepths(results: List<Recognition>, nextImage: Bitmap?) {
        results.forEach {
            // DisNet input vector is ordered like: [h_norm, w_norm, diagonal_norm, x_norm, y_norm, classes]
            val inputBuffer = ByteBuffer.allocateDirect(6 * 4).order(ByteOrder.nativeOrder())
            var xywh = it.getXYWH()
            inputBuffer.putFloat(xywh[3])
            inputBuffer.putFloat(xywh[2])
            inputBuffer.putFloat(it.getDiagonalSize())
            inputBuffer.putFloat(xywh[0])
            inputBuffer.putFloat(xywh[1])
            inputBuffer.putFloat(it.getType().toFloat())

            val outputBuffer = ByteBuffer.allocateDirect(4).order(ByteOrder.nativeOrder())
            tflite.run(inputBuffer, outputBuffer)

            outputBuffer.rewind()
            val outputValue = outputBuffer.float
            it.setDepthPrediction(outputValue.toInt())
        }
    }
}


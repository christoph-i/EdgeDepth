package com.example.edgeyoloapp.ui

import android.graphics.Bitmap
import androidx.camera.core.ExperimentalGetImage
import com.example.edgeyoloapp.Networks.Recognition
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.roundToInt

@ExperimentalGetImage
class MidasFragment : InferenceFragment() {

    private val MODEL_FILE = "midas.tflite"
    private lateinit var tflite: Interpreter
    private lateinit var gpuDelegate: GpuDelegate

    companion object {
        fun newInstance() = MidasFragment()
        private const val INPUT_SIZE = 256
    }

    override fun loadDepthModel() {
        val assetFileDescriptor = context?.assets?.openFd(MODEL_FILE)
        val fileInputStream = assetFileDescriptor?.createInputStream()
        val fileChannel = fileInputStream?.channel
        val startOffset = assetFileDescriptor?.startOffset ?: 0
        val declaredLength = assetFileDescriptor?.declaredLength ?: 0
        val modelBuffer: MappedByteBuffer = fileChannel!!.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

        val compatList = CompatibilityList()
        val options = Interpreter.Options().apply {
            if (compatList.isDelegateSupportedOnThisDevice) {
                // if the device has a supported GPU, add the GPU delegate
                gpuDelegate = GpuDelegate()
                addDelegate(gpuDelegate)
            } else {
                // if the GPU is not supported, run on 4 threads
                setNumThreads(4)
            }
        }

        tflite = Interpreter(modelBuffer, options)
    }

    override fun inferDepths(results: List<Recognition>, nextImage: Bitmap?) {
        val inputBuffer = nextImage?.let { preprocessImage(it) }
        val outputBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE).order(ByteOrder.nativeOrder())

        tflite.run(inputBuffer, outputBuffer)
        outputBuffer.rewind()

        // Convert ByteBuffer to FloatArray for easier manipulation
        val depthMap = FloatArray(INPUT_SIZE * INPUT_SIZE)
        for (i in 0 until INPUT_SIZE * INPUT_SIZE) {
            depthMap[i] = outputBuffer.float
        }

        results.forEach { recognition ->
            val left_abs = (recognition.left * INPUT_SIZE).roundToInt().coerceIn(0, INPUT_SIZE - 1)
            val top_abs = (recognition.top * INPUT_SIZE).roundToInt().coerceIn(0, INPUT_SIZE - 1)
            val right_abs = (recognition.right * INPUT_SIZE).roundToInt().coerceIn(0, INPUT_SIZE - 1)
            val bottom_abs = (recognition.bottom * INPUT_SIZE).roundToInt().coerceIn(0, INPUT_SIZE - 1)

            val depthValues = mutableListOf<Float>()

            if (recognition.title.contains("vehicle", ignoreCase = true)) {
                // For vehicles, use depth close to bbox bottom
                val bottomArea = maxOf(bottom_abs - 3, 0)
                for (y in bottomArea until bottom_abs) {
                    for (x in left_abs until right_abs) {
                        depthValues.add(depthMap[y * INPUT_SIZE + x])
                    }
                }
            } else {
                // For static objects, use median of whole bbox
                for (y in top_abs until bottom_abs) {
                    for (x in left_abs until right_abs) {
                        depthValues.add(depthMap[y * INPUT_SIZE + x])
                    }
                }
            }

            val medianDepth = depthValues.sorted()[depthValues.size / 2]

            recognition.setDepthPrediction(medianDepth.roundToInt())
        }
    }


    private fun preprocessImage(bitmap: Bitmap): ByteBuffer {
        //Using the squeeze approach
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        val inputBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3).order(ByteOrder.nativeOrder())

        val intValues = IntArray(INPUT_SIZE * INPUT_SIZE)
        resizedBitmap.getPixels(intValues, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        var pixel = 0
        for (i in 0 until INPUT_SIZE * INPUT_SIZE) {
            val value = intValues[i]
            inputBuffer.putFloat(((value shr 16 and 0xFF) / 255f))
            inputBuffer.putFloat(((value shr 8 and 0xFF) / 255f))
            inputBuffer.putFloat((value and 0xFF) / 255f)
        }

        inputBuffer.rewind()
        return inputBuffer
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::gpuDelegate.isInitialized) {
            gpuDelegate.close()
        }
        if (::tflite.isInitialized) {
            tflite.close()
        }
    }
}

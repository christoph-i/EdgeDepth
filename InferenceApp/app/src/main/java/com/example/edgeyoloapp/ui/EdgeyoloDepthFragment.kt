package com.example.edgeyoloapp.ui

import android.graphics.Bitmap
import androidx.camera.core.ExperimentalGetImage
import com.example.edgeyoloapp.Networks.EdgeYolo.EdgeYolo
import com.example.edgeyoloapp.Networks.EdgeYolo.EdgeYoloDepth
import com.example.edgeyoloapp.Networks.Recognition
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

@ExperimentalGetImage
class EdgeyoloDepthFragment : InferenceFragment() {


    private lateinit var tflite: Interpreter

    companion object {
        fun newInstance() = EdgeyoloDepthFragment()
    }

    override fun loadDepthModel() {
        return
    }

    override fun getEdgeYoloVariantInstance(): EdgeYolo {
        return EdgeYoloDepth(requireContext())
    }

    override fun inferDepths(results: List<Recognition>, nextImage: Bitmap?) {
        return
    }


}


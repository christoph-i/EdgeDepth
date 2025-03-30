package com.example.edgeyoloapp.ui

import android.graphics.Bitmap
import androidx.camera.core.*
import com.example.edgeyoloapp.Networks.Recognition

@ExperimentalGetImage
class ODOnlyFragment : InferenceFragment() {

    companion object {
        fun newInstance() = ODOnlyFragment()
    }

    override fun loadDepthModel() {
        return
    }

    override fun inferDepths(results: List<Recognition>, nextImage: Bitmap?) {
        results.forEach {
            it.setDepthPrediction(0)
        }
    }
}


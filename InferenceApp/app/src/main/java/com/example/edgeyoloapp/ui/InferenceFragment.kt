package com.example.edgeyoloapp.ui

import android.content.res.AssetManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.lifecycle.ViewModelProvider
import android.os.Bundle
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.camera.core.*
import com.example.edgeyoloapp.viewmodels.MainViewModel
import com.example.edgeyoloapp.viewmodels.SharedViewModel
import com.example.edgeyoloapp.Networks.EdgeYolo.EdgeYolo
import com.example.edgeyoloapp.Networks.Recognition
import com.example.edgeyoloapp.R
import com.example.edgeyoloapp.Visualisation.DetectionsVisualiser
import com.example.edgeyoloapp.databinding.FragmentInferenceViewBinding
import java.io.IOException
import java.util.concurrent.TimeUnit
import kotlin.time.Duration
import kotlin.time.DurationUnit
import kotlin.time.ExperimentalTime

@ExperimentalGetImage
abstract class InferenceFragment : Fragment() {

    private val SLEEP_TIME = 0L

    companion object {
        const val EXAMPLE_IMGS_DIR = "kitti_example_imgskitti_example_imgs"
    }

    private var execTimesTotalODNanos: MutableList<Long> = mutableListOf()
    private var execTimesTotalDepthNanos: MutableList<Long> = mutableListOf()
    private lateinit var assetManager: AssetManager
    private lateinit var exampleImgFiles: Array<String>

    private var nextImage: Bitmap? = null
    private var imageWithBoxes: Bitmap? = null

    private var nrProcessedImages: Int = 0

    private lateinit var viewModel: MainViewModel
    private lateinit var sharedViewModel: SharedViewModel
    private lateinit var _binding: FragmentInferenceViewBinding

    private lateinit var edgeYolo: EdgeYolo

    @Volatile
    private var runInference: Boolean = true

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewModel = ViewModelProvider(this).get(MainViewModel::class.java)
        activity?.let {
            sharedViewModel = ViewModelProvider(it).get(SharedViewModel::class.java)
            assetManager = it.assets // Use the activity's context to get the assets

            try {
                exampleImgFiles = assetManager.list(EXAMPLE_IMGS_DIR) ?: arrayOf()
            } catch (e: IOException) {
                throw RuntimeException(e)
            }
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentInferenceViewBinding.inflate(inflater, container, false)
        return _binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // Setup detection model
        this.edgeYolo = getEdgeYoloVariantInstance()

        // Set up the listeners for take photo and video capture buttons
        _binding.startButton.setOnClickListener { startInferenceSpeedTest() }
        _binding.stopButton.setOnClickListener { stopInference() }

        // Set up the listener for the back button
        _binding.backButton.setOnClickListener {
            stopInference()
            parentFragmentManager.beginTransaction()
                .replace(R.id.container, EntryFragment.newInstance())
                .commit()
        }
    }

    protected open fun getEdgeYoloVariantInstance(): EdgeYolo {
        return EdgeYolo(requireContext())
    }

    abstract fun loadDepthModel()

    abstract fun inferDepths(results: List<Recognition>, nextImage: Bitmap?)

    private fun stopInference() {
        this.runInference = false
    }

    private fun startInferenceSpeedTest() {
        this.runInference = true
        this.nrProcessedImages = 0
        this.execTimesTotalODNanos = mutableListOf()
        this.execTimesTotalDepthNanos = mutableListOf()
        Thread {
            // New object so the measurements from last run are not affecting new results
            this.edgeYolo = getEdgeYoloVariantInstance()
            loadDepthModel()
            while (runInference) {

                loadNextImage(object : DataLoadListener {
                    override fun onDataLoaded(data: Bitmap) {
                        nextImage = data
                    }
                })

                val t0TotalOD = System.nanoTime()
                val results = edgeYolo.runInference(nextImage)
                val t1TotalOD = System.nanoTime()
                execTimesTotalODNanos.add(Math.abs(t1TotalOD - t0TotalOD))

                val t0TotalDepth = System.nanoTime()
                inferDepths(results, nextImage)
                val t1TotalDepth = System.nanoTime()
                execTimesTotalDepthNanos.add(Math.abs(t1TotalDepth - t0TotalDepth))

                this.imageWithBoxes = DetectionsVisualiser.visDetections(nextImage, results, true)

                nrProcessedImages++

                activity?.runOnUiThread {
                    updateUI()
                }

                Thread.sleep(SLEEP_TIME)
            }
        }.start()
    }

    @OptIn(ExperimentalTime::class)
    private fun updateUI() {
        _binding.imageViewResult.setImageBitmap(this.imageWithBoxes)
        _binding.nrIterationsView.text = "Iteration: $nrProcessedImages"
        _binding.lastExecTimeODTotalView.text = "Last total (OD): ${String.format("%.2f", Duration.convert(execTimesTotalODNanos.last().toDouble(), DurationUnit.NANOSECONDS, DurationUnit.MILLISECONDS))}"
        _binding.avgExecTimeODTotalView.text = "Avg total (OD): ${String.format("%.2f", Duration.convert(execTimesTotalODNanos.average(), DurationUnit.NANOSECONDS, DurationUnit.MILLISECONDS))}"

        val lastExecTimeDepthTotalMillis = Duration.convert(execTimesTotalDepthNanos.last().toDouble(), DurationUnit.NANOSECONDS, DurationUnit.MILLISECONDS)
        val avgExecTimeDepthTotalMillis = Duration.convert(execTimesTotalDepthNanos.average(), DurationUnit.NANOSECONDS, DurationUnit.MILLISECONDS)

        val lastExecTimeDepthTotalFormatted = String.format("%.2f", lastExecTimeDepthTotalMillis)
        val avgExecTimeDepthTotalFormatted = String.format("%.2f", avgExecTimeDepthTotalMillis)

        _binding.lastExecTimeDepthTotalView.text = "Last total (Depth): $lastExecTimeDepthTotalFormatted ms"
        _binding.avgExecTimeDepthTotalView.text = "Avg total (Depth): $avgExecTimeDepthTotalFormatted ms"
    }

    interface DataLoadListener {
        fun onDataLoaded(data: Bitmap)
    }

    private fun loadNextImage(listener: DataLoadListener) {
        try {
            val imgFilePath = "$EXAMPLE_IMGS_DIR/${exampleImgFiles[nrProcessedImages % exampleImgFiles.size]}"
            assetManager.open(imgFilePath).use { inputStream ->
                val image = BitmapFactory.decodeStream(inputStream)
                nextImage = image
                listener.onDataLoaded(image)
            }
        } catch (e: IOException) {
            e.printStackTrace()
        }
    }
}
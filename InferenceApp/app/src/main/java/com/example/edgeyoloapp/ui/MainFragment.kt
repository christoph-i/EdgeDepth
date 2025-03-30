package com.example.edgeyoloapp.ui

import android.annotation.SuppressLint
import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Build
import androidx.lifecycle.ViewModelProvider
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.*
import androidx.core.content.ContextCompat
import androidx.lifecycle.MutableLiveData
import com.example.edgeyoloapp.databinding.FragmentMainBinding
import com.example.edgeyoloapp.utils.toBitmap
import com.example.edgeyoloapp.viewmodels.MainViewModel
import com.example.edgeyoloapp.viewmodels.SharedViewModel
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import androidx.lifecycle.lifecycleScope

import com.example.edgeyoloapp.Networks.EdgeYolo.EdgeYolo
import com.example.edgeyoloapp.Visualisation.DetectionsVisualiser
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.TimeUnit
import kotlin.concurrent.thread

@ExperimentalGetImage class MainFragment : Fragment() {

    companion object {
        fun newInstance() = MainFragment()
    }

    private var execTimesTotalNanos: MutableList<Long> = mutableListOf()

    private lateinit var viewModel: MainViewModel
    private lateinit var sharedViewModel: SharedViewModel
    private lateinit var _binding: FragmentMainBinding

    private var imageCapture: ImageCapture? = null

    private lateinit var cameraExecutor: ExecutorService

    private lateinit var edgeYolo: EdgeYolo
    @Volatile
    private var nextImage: Bitmap? = null
    @Volatile
    private var imageWithBoxes: Bitmap? = null

    private var nrProcessedImages: Int = 0

    private val capturedImageLiveData = MutableLiveData<ImageProxy>()

    @Volatile
    private var runInference: Boolean = true

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewModel = ViewModelProvider(this).get(MainViewModel::class.java)
        activity?.let {
            sharedViewModel = ViewModelProvider(it).get(SharedViewModel::class.java)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentMainBinding.inflate(inflater, container, false)
        return _binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        sharedViewModel.isPermissionGranted.observe(viewLifecycleOwner) {
            if (it) startCamera()
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        // Set up the listeners for take photo and video capture buttons
        _binding.startButton.setOnClickListener { startContinuousInference() }
        _binding.stopButton.setOnClickListener { stopInference() }

        startObserverInference()

    }

    private fun stopInference() {
        this.runInference = false

    }

    private fun startContinuousInference() {
        this.nrProcessedImages = 0
        this.runInference = true
        Thread {
            this.edgeYolo = EdgeYolo(requireContext())
            while (runInference) {
                capturePhoto()
                Thread.sleep(50)
            }
        }.start()
    }

    private fun updateUI() {

        _binding.lastExecTimePreView.setText("Last Pre: ${this.edgeYolo.lastExecTimePreMillis}")
        _binding.avgExecTimePreView.setText("Avg Pre: ${this.edgeYolo.avgExecTimePreMillis}")

        _binding.lastExecTimeRawModelView.setText("Last Model: ${this.edgeYolo.lastExecTimeRawModelMillis}")
        _binding.avgExecTimeRawModelView.setText("Avg Model: ${this.edgeYolo.avgExecTimeRawModelMillis}")

        _binding.lastExecTimePostView.setText("Last Post: ${this.edgeYolo.lastExecTimePostMillis}")
        _binding.avgExecTimePostView.setText("Avg Post: ${this.edgeYolo.avgExecTimePostMillis}")

        _binding.lastExecTimeTotalView.setText("Last TOTAL: ${TimeUnit.NANOSECONDS.toMillis(this.execTimesTotalNanos.last())}")
        _binding.avgExecTimeTotalView.setText("Avg TOTAL: ${
            TimeUnit.NANOSECONDS.toMillis(this.execTimesTotalNanos.average().let {
                if (it.isNaN()) 0 else it.toLong()
            })
        }")

        //_binding.nrIterationsView.setText("Iteration: " + this.nrProcessedImages)

        _binding.imageViewResult.setImageBitmap(this.imageWithBoxes)
    }

    private fun startObserverInference(){
        capturedImageLiveData.observe(viewLifecycleOwner) { image ->
            // Process the image in a background thread
            lifecycleScope.launch(Dispatchers.IO) {
                val rotationDegrees = image.imageInfo.rotationDegrees
                var imgBitmap = image.image!!.toBitmap()

                // Rotate the bitmap if necessary
                if (rotationDegrees != 0) {
                    val matrix = Matrix().apply { postRotate(rotationDegrees.toFloat()) }
                    imgBitmap = Bitmap.createBitmap(imgBitmap, 0, 0, imgBitmap.width, imgBitmap.height, matrix, true)
                }

                // Calculate the target aspect ratio (16:9) dimensions
                val targetAspect = 16.0 / 9.0
                val currentAspect = imgBitmap.width.toDouble() / imgBitmap.height

                if (currentAspect > targetAspect) {
                    // Crop the sides
                    val newWidth = (imgBitmap.height * targetAspect).toInt()
                    val xOffset = (imgBitmap.width - newWidth) / 2
                    imgBitmap = Bitmap.createBitmap(imgBitmap, xOffset, 0, newWidth, imgBitmap.height)
                } else {
                    // Crop the top and bottom
                    val newHeight = (imgBitmap.width / targetAspect).toInt()
                    val yOffset = (imgBitmap.height - newHeight) / 2
                    imgBitmap = Bitmap.createBitmap(imgBitmap, 0, yOffset, imgBitmap.width, newHeight)
                }

                val t0Total = System.nanoTime()
                val results = this@MainFragment.edgeYolo.runInference(imgBitmap)
                val t1Total = System.nanoTime()
                this@MainFragment.execTimesTotalNanos.add(Math.abs(t1Total - t0Total))

               // this@MainFragment.imageWithBoxes = DetectionsVisualiser.visDetections(imgBitmap, results, true)
                this@MainFragment.imageWithBoxes = DetectionsVisualiser.visDetections(imgBitmap, results, true)

                this@MainFragment.nrProcessedImages += 1

                activity?.runOnUiThread() {
                    updateUI()
                }

                image.close()
            }
        }
    }

    private fun capturePhoto() {
        val imageCapture = imageCapture ?: return

        /** https://developer.android.com/training/camerax/take-photo
         * takePicture(Executor, OnImageCapturedCallback): This method provides an in-memory buffer of the captured image.
         * takePicture(OutputFileOptions, Executor, OnImageSavedCallback): This method saves the captured image to the provided file location.
         */

        imageCapture.takePicture(
            ContextCompat.getMainExecutor(requireContext()),
            object : ImageCapture.OnImageCapturedCallback() {
                @SuppressLint("UnsafeOptInUsageError")
                override fun onCaptureSuccess(image: ImageProxy) {
                    super.onCaptureSuccess(image)
                    capturedImageLiveData.postValue(image)
                }
            }
        )
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())

        cameraProviderFuture.addListener({
            // Used to bind the lifecycle of cameras to the lifecycle owner
            val cameraProvider: ProcessCameraProvider = cameraProviderFuture.get()

//            // Preview
//            val preview = Preview.Builder()
//                .build()
//                .also {
//                    it.setSurfaceProvider(_binding.viewFinder.surfaceProvider)
//                }

            //Image
            imageCapture = ImageCapture.Builder()
                /**
                 * Use ImageCapture.Builder.setCaptureMode() to configure the capture mode when taking a photo:
                 * - CAPTURE_MODE_MINIMIZE_LATENCY: optimize image capture for latency.
                 * - CAPTURE_MODE_MAXIMIZE_QUALITY: optimize image capture for image quality.
                 * The capture mode defaults to CAPTURE_MODE_MINIMIZE_LATENCY
                 */
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            // Select back camera as a default
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                // Unbind use cases before rebinding
                cameraProvider.unbindAll()

                // Bind use cases to camera
                //cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture)
                cameraProvider.bindToLifecycle(this, cameraSelector, imageCapture)
            } catch(exc: Exception) {
                Log.e("edgeyolo app camera:", "Use case binding failed", exc)
            }
        }, ContextCompat.getMainExecutor(requireContext()))
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}
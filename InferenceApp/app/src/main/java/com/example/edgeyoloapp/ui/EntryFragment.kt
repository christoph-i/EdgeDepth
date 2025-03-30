package com.example.edgeyoloapp.ui

import android.os.Bundle
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import com.example.edgeyoloapp.R
import com.example.edgeyoloapp.databinding.FragmentEntryBinding

class EntryFragment : Fragment() {

    companion object {
        fun newInstance() = EntryFragment()
    }

    private lateinit var _binding: FragmentEntryBinding

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentEntryBinding.inflate(inflater, container, false)
        return _binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        _binding.odOnlyButton.setOnClickListener {
            parentFragmentManager.beginTransaction()
                .replace(R.id.container, ODOnlyFragment.newInstance())
                .addToBackStack(null)
                .commit()
        }

        _binding.disNetButton.setOnClickListener {
            parentFragmentManager.beginTransaction()
                .replace(R.id.container, DisNetFragment.newInstance())
                .addToBackStack(null)
                .commit()
        }

        _binding.midasButton.setOnClickListener {
            parentFragmentManager.beginTransaction()
                .replace(R.id.container, MidasFragment.newInstance())
                .addToBackStack(null)
                .commit()
        }

        _binding.edgeyoloDepthButton.setOnClickListener {
            parentFragmentManager.beginTransaction()
                .replace(R.id.container, EdgeyoloDepthFragment.newInstance())
                .addToBackStack(null)
                .commit()
        }

    }
}
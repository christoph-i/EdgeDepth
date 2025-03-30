package com.example.edgeyoloapp.Networks.EdgeYolo;

import static com.example.edgeyoloapp.Networks.EdgeYolo.EdgeYolo.LABELS;

import android.graphics.RectF;

import com.example.edgeyoloapp.Networks.Recognition;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class EdgeYoloPostprocessor {
    public Map<Integer, List<Recognition>> sortDetectionsByType(
            List<Recognition> detections) {
        Map<Integer, List<Recognition>> sortedDetections = new HashMap<>();
        detections.forEach(detection -> {
            List<Recognition> recognitions = sortedDetections.computeIfAbsent(detection.getType(), k -> new ArrayList<>());
            recognitions.add(detection);
        });
        return sortedDetections;
    }

    public float box_iou(RectF a, RectF b) {
        float intersection = box_intersection(a, b);
        float union = box_union(a, b, intersection);

        if (union == 0.f) return 0.f;

        return intersection / union;
    }

    public float box_intersection(RectF a, RectF b) {
        float w = overlap(a.left, b.left, a.right, b.right);
        float h = overlap(a.top, b.top, a.bottom, b.bottom);
        if (w < 0 || h < 0) return 0;
        return w * h;
    }

    public float box_union(RectF a, RectF b, float intersection) {
        return (a.right - a.left) * (a.bottom - a.top) + (b.right - b.left) * (b.bottom - b.top) - intersection;
    }

    public float overlap(float a1, float b1, float a2, float b2) {
        return Math.min(a2, b2) - Math.max(a1, b1);
    }

    public List<Recognition> nms(List<Recognition> detections) {
        List<Recognition> nmsList = new ArrayList<>();
        Map<Integer, List<Recognition>> sortedDetections = sortDetectionsByType(detections);

        sortedDetections.forEach((type, recognitions) -> {
            while (recognitions.size() > 0) {
                // https://stackoverflow.com/a/29444594 seems to be fastest
                Recognition[] array = recognitions.toArray(new Recognition[0]);
                Arrays.sort(array, (a1, a2) -> Float.compare(a2.getConfidence(), a1.getConfidence()));
                // insert detection with max confidence
                Recognition max = array[0];
                nmsList.add(max);
                recognitions.clear();

                for (int i = 1; i < array.length; i++) {
                    Recognition detection = array[i];
                    RectF b = detection.getLocation();
                    if (box_iou(max.getLocation(), b) < EdgeYolo.NMS_IOU_THRESHOLD) {
                        recognitions.add(detection);
                    }
                }
            }
        });

        return nmsList;
    }

    private int yoloArgmax(float[] yoloDetection, boolean includeDepth) {
        if (yoloDetection == null || yoloDetection.length <= 0) return -1;

        int yoloOffset = 5;
        int argmax = yoloOffset;

        int maxIndex = yoloDetection.length;
        if (includeDepth) {
            maxIndex--;
        }

        for (int i = argmax; i < maxIndex; i++) {
            argmax = (yoloDetection[i] > yoloDetection[argmax]) ? i : argmax;
        }

        return argmax - yoloOffset;
    }

    public List<Recognition> toRecognitions(float[] output, int valuesPerDetectin, boolean includeDepth) {
        List<Recognition> detections = new ArrayList<>();
        for(int i = 0; i < output.length; i+=valuesPerDetectin) {
            float[] yoloDetection = Arrays.copyOfRange(output, i, i+valuesPerDetectin);

            final float confidence = yoloDetection[4];

            int classId = yoloArgmax(yoloDetection, includeDepth);
            float typeScore = yoloDetection[5 + classId];

            final float score = typeScore * confidence;

            if (score >= EdgeYolo.MINIMUM_CONFIDENCE) {
                //float cutoffTopThird = EdgeYolo.INPUT_SIZE / 2.0f;
                final float xPos = yoloDetection[0] / EdgeYolo.INPUT_SIZE;
                final float yPos = yoloDetection[1] / EdgeYolo.INPUT_SIZE;
                final float w = yoloDetection[2] / EdgeYolo.INPUT_SIZE;
                final float h = yoloDetection[3] / EdgeYolo.INPUT_SIZE;

                final float top    = Math.max(0.f, yPos - h / 2);
                final float bottom = Math.min(1.f, yPos + h / 2);
                final float left   = Math.max(0.f, xPos - w / 2);
                final float right  = Math.min(1.f, xPos + w / 2);

                final RectF rect = new RectF(left, top, right, bottom);

                String className = (includeDepth) ?  EdgeYoloDepth.LABELS.get(classId) :EdgeYolo.LABELS.get(classId);
                if (!includeDepth && EdgeYolo.REDUCE_CLASSES_TO_VEHICLE_SIGN) {
                    if (classId == 4 || classId == 5 || classId == 8) {
                        classId = 1;
                        className = "Vehicle";
                    } else if (classId != 0) {
                        // class is not vehicle or sign -> ignore in reduced case
                        continue;
                    }
                }
                Integer depth = null;
                if (includeDepth == true) {
                    depth = (int) (yoloDetection[7] * 41.0); // Approximate scaling to metric value (exact scaling factor needs calibration but is irrelevant for speed testing)
                }
                detections.add(new Recognition("0", className, classId, score, rect, left, top, right, bottom, depth));
            }

        }

        return detections;

    }
}

package com.example.edgeyoloapp.Networks;

import android.graphics.RectF;

public class Recognition {
    /**
     * A unique identifier for what has been recognized. Specific to the class, not the instance of
     * the object.
     */
    private final String id;

    /**
     * Display name for the recognition.
     */
    private final String title;

    private final int type;

    /**
     * A sortable score for how good the recognition is relative to others. Higher should be better.
     */
    private final Float confidence;

    /**
     * Optional location within the source image for the location of the recognized object.
     */
    private RectF location;

    private final float left;
    private final float top;
    private final float right;
    private final float bottom;
    private Integer depthPrediction;

    public Recognition(final String id, final String title, int type, final Float confidence, final RectF location, float left, float top, float right, float bottom, Integer depthPrediction) {
        this.id = id;
        this.title = title;
        this.type = type;
        this.confidence = confidence;
        this.location = location;
        this.left = left;
        this.top = top;
        this.right = right;
        this.bottom = bottom;
        this.depthPrediction = depthPrediction;
    }

    public String getId() {
        return id;
    }

    public String getTitle() {
        return title;
    }

    public int getType() {
        return type;
    }

    public Float getConfidence() {
        return confidence;
    }

    public RectF getLocation() {
        return new RectF(location);
    }

    public float getLeft() {
        return left;
    }

    public float getTop() {
        return top;
    }

    public float getRight() {
        return right;
    }

    public float getBottom() {
        return bottom;
    }

    public Integer getDepthPrediction() {
        return depthPrediction;
    }

    public void setDepthPrediction(int depthPrediction) {
        this.depthPrediction = depthPrediction;
    }

    public void setLocation(RectF location) {
        this.location = location;
    }

    /**
     * Calculate and return the relative x, y, width, and height values of the bounding box.
     * @return an array containing x, y, width, and height.
     */
    public float[] getXYWH() {
        float x = (left + right) / 2.0f;
        float y = (top + bottom) / 2.0f;
        float width = right - left;
        float height = bottom - top;
        return new float[]{x, y, width, height};
    }

    public float getDiagonalSize() {
        float width = right - left;
        float height = bottom - top;
        return (float) Math.sqrt(width * width + height * height);
    }

    @Override
    public String toString() {
        String resultString = "";
        if (id != null) {
            resultString += "[" + id + "] ";
        }

        if (title != null) {
            resultString += title + " ";
        }

        if (confidence != null) {
            resultString += String.format("(%.1f%%) ", confidence * 100.0f);
        }

        if (location != null) {
            resultString += location + " ";
        }

        return resultString.trim();
    }
}

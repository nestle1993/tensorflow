/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.example.android.tflitehotworddemo;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.os.SystemClock;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;


/**
 * A classifier specialized to detect keyword using TensorFlow.
 */
public class TensorFlowKeywordSpotting {
    private static final String TAG = "tf.KeywordSpotting";

    // Config values.
    private final int fftSize = 320;//400;
    private final int hopSize = 160;
    private final int maxSignalLength = 3840; // 3680 buffer size + 160 remain
    private final int numFrames = (maxSignalLength - fftSize) / hopSize + 1;
    private final int numLayers = 2;
    private final int hiddenSize = 128;

    // Decoder config
    private final float threshold = 0.2f;
    private final int maxKeepLength = 15; // at most 10 segments (3s)  will be kept for classify
    private final int numClasses = 6; // 0 space 1 ni 2 hao 3 le 4 otherWords 5 blank_ctc
    private final String labelSeq = "123";
    private final char otherWordIdx = '4';
    private final List<String> fuzzyLabels = Arrays.asList("143", "23", "423", "1423", "243", "1443");

    // data
    private float[] floatValues;
    private float[] floatValuesRemain;
    private ArrayList<float[][]> stftMatrix;
    private float[][] stateMatrix;
    private float[][] outputStatesMatrix;
    private float[][] outputLogitMatrix;

    private Classifier classifier;
    private Interpreter tflite;

    public TensorFlowKeywordSpotting(Activity activity, String modelPath) throws IOException{
        tflite = new Interpreter(loadModelFile(activity, modelPath));
        classifier = new Classifier(threshold, maxKeepLength, numClasses, otherWordIdx);

        stateMatrix = new float[numLayers][hiddenSize];
        for (int i = 0; i < numLayers; i ++) {
            Arrays.fill(stateMatrix[i], 0.0f);
        }
        outputStatesMatrix = new float[numLayers][hiddenSize];
        outputLogitMatrix = new float[numFrames][numClasses];
        floatValuesRemain = new float[0];
    }

    /** Memory-map the model file in Assets. */
    private MappedByteBuffer loadModelFile(Activity activity, String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(modelPath);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    private float[] concat(float[] first, float[] second) {
        float[] result = Arrays.copyOf(first, first.length + second.length);
        System.arraycopy(second, 0, result, first.length, second.length);
        return result;
    }

    private boolean evaluate(String s, boolean strict) {
        if (strict) {
            return s.contains(labelSeq);
        } else {
            for (String lb : fuzzyLabels) {
                if (s.endsWith(lb)) {
                    return true;
                }
            }
            return false;
        }
    }

    private float[] padding(float[] origin, int paddingLength) {
        float[] padding = new float[paddingLength - origin.length];
        return concat(origin, padding);
    }


    public void feed(float[] segment) {
        floatValues = concat(floatValuesRemain, segment);
        int remainLen = (floatValues.length - fftSize) % hopSize + fftSize - hopSize;
        floatValuesRemain =
                Arrays.copyOfRange(floatValues, floatValues.length - remainLen, floatValues.length);

        assert floatValues.length <= maxSignalLength;
        Log.e(TAG, "float value length" + floatValues.length);
        // one possiblity that floatValues.length < mSL: the first package
        if (floatValues.length < maxSignalLength) {
            float[] paddingZeros = new float[maxSignalLength - floatValues.length];
            Arrays.fill(paddingZeros, 0.0f);
            floatValues = concat(paddingZeros, floatValues);
        }
        stftMatrix = Util.stft(floatValues, fftSize, hopSize);
    }


    public boolean classify() {
        Log.e(TAG, "Classify keyword");

        // Prepare input and output matrixes
        Object[] inputs = new Object[numFrames + numLayers];
        for (int i = 0; i < numFrames; ++i) {
            inputs[i] = stftMatrix.get(i);
        }
        for (int i = 0; i < numLayers; ++i) {
            inputs[numFrames + i] = Util.expandDim(stateMatrix[i]);
        }
        HashMap outputs = new HashMap();
        outputs.put(Integer.valueOf(0), outputLogitMatrix);
        outputs.put(Integer.valueOf(1), outputStatesMatrix);

        // Run
        long startTime = SystemClock.uptimeMillis();
        tflite.runForMultipleInputsOutputs(inputs, outputs);
        long endTime = SystemClock.uptimeMillis();
        Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));

        // Decode
        String result = classifier.ctcDecode(outputLogitMatrix);
        Log.e(TAG, "RESULT: " + result);

        // Update state matrix
        for (int i = 0; i < numLayers; i ++) {
            for (int j = 0; j < hiddenSize; j ++) {
                stateMatrix[i][j] = outputStatesMatrix[i][j];
            }
        }

        return evaluate(result, false);
    }

    public void clear(){
        classifier.clear();
        for (int i = 0; i < numLayers; i ++) {
            Arrays.fill(stateMatrix[i], 0.0f);
        }
        for (int i = 0; i < numFrames; i ++) {
            Arrays.fill(outputLogitMatrix[i], 0.0f);
        }
        if (floatValues != null) {
            floatValues = null;
        }
        floatValuesRemain = new float[0];
        Log.e(TAG,"Clean state~~~~~~");
    }

    public void close() {
        tflite.close();
        tflite = null;
    }
}

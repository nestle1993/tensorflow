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

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
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
    //private static final String model_dir = "xiaobudian_0.037_melinput.tflite";
    //private static final String model_dir = "xiaobudian_0.037.tflite";
    private String modelDir;

    // Config values.
    private final int fftSize = 320;
    private final int stftFeatureSize = fftSize / 2 + 1;
    private final int nMel = 60;
    private final String melBasisFileName = "mel_basis.txt";
    private final String melBasisSparseFileName = "mel_basis_sparse.txt";
    private final String hannWindowFileName = "hann_window_320.txt";
    private final int hopSize = 160;
    private final int maxSignalLength = 3840; // 3680 buffer size + 160 remain
    private final int numFrames = (maxSignalLength - fftSize) / hopSize + 1;
    private final int numLayers = 2;
    private final int hiddenSize = 128;

    // Decoder config
    private float threshold = 0.3f;
    private final int maxKeepLength = 15; // at most 10 segments (3s)  will be kept for classify
    private final int numClasses = 6; // 0 space 1 ni 2 hao 3 le 4 otherWords 5 blank_ctc
    private final String labelSeq = "123";
    private final char otherWordIdx = '4';
    private final List<String> fuzzyLabels =
            //Arrays.asList("143", "23", "423", "1423", "243", "1443");
            Arrays.asList("23");
    private String decodeResult;

    // data
    private float[] floatValues;
    private float[] floatValuesRemain;
    private ArrayList<float[][]> stftMatrix;
    private ArrayList<float[][]> melMatrix;
    private float[][] stateMatrix;
    private float[][] outputStatesMatrix;
    private float[][] outputLogitMatrix;
    private float[] hannWindow;
    // quantized ones
    private ArrayList<byte[][]> quantizedStftMatrix;
    private ArrayList<byte[][]> quantizedMelMatrix;
    private byte[][] quantizedStateMatrix;
    private byte[][] quantizedOutputStatesMatrix;
    private byte[][] quantizedOutputLogitMatrix;
    float quantizeStftMean = 0;
    float quantizeStftStd = 85f;
    float quantizeStateMean = 127.5f;
    float quantizeStateStd = 127.5f;

    private float[][] kernelMatrix;
    String type = "online_1.6"; //quantize_mel, quantize_mel_sparse, noquantize_stft, noquantize_mel_sparse
    boolean useQuantizedModel;
    boolean useMelSparse;
    boolean useStftInput;

    float[][] melBasis;
    ArrayList<HashMap<Integer, Float>> melBasisSparse;

    private Classifier classifier;
    private Interpreter tflite;

    public TensorFlowKeywordSpotting(Activity activity) throws IOException{
        modelType(type);

        tflite = new Interpreter(loadModelFile(activity, modelDir));
        classifier = new Classifier(threshold, maxKeepLength, numClasses, otherWordIdx);

        if (!useQuantizedModel) {
            stateMatrix = new float[numLayers][hiddenSize];
            for (int i = 0; i < numLayers; i ++) {
                Arrays.fill(stateMatrix[i], 0.0f);
            }
            outputStatesMatrix = new float[numLayers][hiddenSize];
            outputLogitMatrix = new float[numFrames][numClasses];
            kernelMatrix = new float[6][128];
        } else {
            quantizedStateMatrix = new byte[numLayers][hiddenSize];
            for (int i = 0; i < numLayers; i++) {
                Arrays.fill(quantizedStateMatrix[i], Util.float2Quantize(0.f, quantizeStateMean, quantizeStateStd));
            }
            quantizedOutputStatesMatrix = new byte[numLayers][hiddenSize];
            quantizedOutputLogitMatrix = new byte[numFrames][numClasses];
        }

        floatValuesRemain = new float[0];

        // read mel basis from file
        if (!useStftInput) {
            if (!useMelSparse) {
                melBasis = new float[stftFeatureSize][nMel];
                loadMelBasisFile(activity, melBasisFileName, melBasis);
            } else {
                melBasisSparse = loadMelBasisSparseFile(activity, melBasisSparseFileName);
            }
        }
        hannWindow = loadHannWindowFile(activity, hannWindowFileName);
    }

    private void modelType(String type) {
        if ("quantize_mel".equals(type)) {
            useQuantizedModel = true;
            useMelSparse = false;
            useStftInput = false;
            threshold = 0.5f;
            modelDir = "test_graph.tflite";
        } else if ("quantize_mel_sparse".equals(type)) {
            useQuantizedModel = true;
            useMelSparse = true;
            useStftInput = false;
            threshold = 0.5f;
            modelDir = "test_graph.tflite";
        } else if ("noquantize_stft".equals(type)) {
            useQuantizedModel = false;
            useMelSparse = false;
            useStftInput = true;
            modelDir = "xiaobudian_0.037.tflite";//"graph_stft_noquantize.tflite";
        } else if ("noquantize_mel_sparse".equals(type)) {
            useQuantizedModel = false;
            useMelSparse = true;
            useStftInput = false;
            modelDir = "graph_mel_noquantize.tflite";
        } else if ("online_1.6".equals(type)) {
            useQuantizedModel = false;
            useMelSparse = false;
            useStftInput = true;
            modelDir = "xbd_tf1.6.tflite";
        } else {
            modelDir = "no_type";
        }
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

    private float[] loadHannWindowFile(Activity activity, String filePath) throws IOException {
        int i = 0;
        float[] ret = new float[fftSize];
        InputStream in = activity.getAssets().open(filePath);
        //ArrayList<String> lines = new ArrayList<>();
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new InputStreamReader(in));
            String tempString = null;
            while ((tempString = reader.readLine()) != null) {
                String[] numberSplit = tempString.split(" ");
                assert numberSplit.length == fftSize;
                for (int j = 0; j < numberSplit.length; j ++) {
                    ret[j] = Float.parseFloat(numberSplit[j]);
                }
                i ++;
            }
            assert i == 1;
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e1) {
                }
            }
        }
        return ret;
    }

    private void loadMelBasisFile(Activity activity, String filePath, float[][] melBasis) throws IOException {
        int i = 0;
        InputStream in = activity.getAssets().open(filePath);
        //ArrayList<String> lines = new ArrayList<>();
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new InputStreamReader(in));
            String tempString = null;
            while ((tempString = reader.readLine()) != null) {
                String[] featureSplit = tempString.split(" ");
                assert featureSplit.length == nMel;
                for (int j = 0; j < featureSplit.length; j ++) {
                    melBasis[i][j] = Float.parseFloat(featureSplit[j]);
                }
                i ++;
            }
            assert i == stftFeatureSize;
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e1) {
                }
            }
        }
    }

    private ArrayList<HashMap<Integer, Float>> loadMelBasisSparseFile(
            Activity activity, String filePath) throws IOException {
        ArrayList<HashMap<Integer, Float>> ret = new ArrayList<>();
        InputStream in = activity.getAssets().open(filePath);
        BufferedReader reader = null;
        try {
            reader = new BufferedReader(new InputStreamReader(in));
            String tempString = null;
            while ((tempString = reader.readLine()) != null) {
                String[] featureSplit = tempString.split("\t");
                HashMap<Integer, Float> sparseFeature = new HashMap<>();
                for (int j = 0; j < featureSplit.length; j ++) {
                    String[] indexValuePair = featureSplit[j].split(" ");
                    sparseFeature.put(Integer.parseInt(indexValuePair[0]), Float.parseFloat(indexValuePair[1]));
                }
                ret.add(sparseFeature);
            }
            assert ret.size() == stftFeatureSize;
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException e1) {
                }
            }
        }
        return ret;
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
                if (lb.equals(s) || s.contains(labelSeq)) {
                    return true;
                }
            }
            return false;
        }
    }

    public void feed(float[] segment) {
        floatValues = concat(floatValuesRemain, segment);
        int remainLen = (floatValues.length - fftSize) % hopSize + fftSize - hopSize;
        floatValuesRemain =
                Arrays.copyOfRange(floatValues, floatValues.length - remainLen, floatValues.length);

        assert floatValues.length <= maxSignalLength;
        //Log.e(TAG, "float value length" + floatValues.length);
        // one possiblity that floatValues.length < mSL: the first package
        if (floatValues.length < maxSignalLength) {
            float[] paddingZeros = new float[maxSignalLength - floatValues.length];
            Arrays.fill(paddingZeros, 0.0f);
            floatValues = concat(paddingZeros, floatValues);
        }


        if (!useQuantizedModel) {
            if (useStftInput) {
                /*
                hannWindow = new float[fftSize];
                for (int i = 0; i < fftSize; i ++)
                    hannWindow[i] = 1.f;
                */
                stftMatrix = Util.stft(floatValues, fftSize, hopSize, hannWindow);
            } else {
                if (!useMelSparse) {
                    melMatrix = Util.mel(floatValues, fftSize, hopSize, melBasis);
                } else {
                    melMatrix = Util.melSparse(floatValues, fftSize, hopSize, melBasisSparse);
                }
            }
        } else {
            if (useStftInput) {
                quantizedStftMatrix =
                       Util.quantizedStft(floatValues, fftSize, hopSize, quantizeStftMean, quantizeStftStd);
            } else {
                if (!useMelSparse) {
                    quantizedMelMatrix =
                            Util.quantizedMel(floatValues, fftSize, hopSize, quantizeStftMean, quantizeStftStd, melBasis);
                } else {
                    quantizedMelMatrix =
                            Util.quantizedMelSparse(
                                    floatValues, fftSize, hopSize, quantizeStftMean, quantizeStftStd, melBasisSparse, hannWindow);
                }
            }
        }
    }


    public boolean classify() {
        //Log.e(TAG, "Classify keyword");

        // Prepare input and output matrixes
//        Object[] inputs = new Object[2];
//        float[][][] tmp = new float[numFrames][1][stftFeatureSize];
//        for (int i = 0; i < numFrames; ++i) {
//            for (int j = 0; j < stftMatrix.get(i)[0].length; ++ j) {
//                tmp[i][0][j] = stftMatrix.get(i)[0][j];
//            }
//        }
//        float[][][] tmp2 = new float[numLayers][1][hiddenSize];
//        for (int i = 0; i < numLayers; ++i) {
//            for (int j = 0; j < hiddenSize; ++j) {
//                tmp2[i][0][j] = stateMatrix[i][j];
//            }
//        }
//        inputs[0] = tmp;
////        ArrayList<float[][]> tmp = new ArrayList<>();
////        tmp.add(Util.expandDim(stateMatrix[0]));
////        tmp.add(Util.expandDim(stateMatrix[1]));
//        inputs[1] = tmp2;
//        //System.out.println(stftMatrix.get(0).getClass());

        Object[] inputs = new Object[numFrames + numLayers];
        for (int i = 0; i < numFrames; ++i) {
            if (!useQuantizedModel) {
                if (useStftInput) {
                    inputs[i] = stftMatrix.get(i);
                } else {
                    inputs[i] = melMatrix.get(i);
                }
            } else {
                if (useStftInput) {
                    inputs[i] = quantizedStftMatrix.get(i);
                } else {
                    inputs[i] = quantizedMelMatrix.get(i);
                }
            }
        }
        for (int i = 0; i < numLayers; ++i) {
            if (!useQuantizedModel) {
                inputs[numFrames + i] = Util.expandDim(stateMatrix[i]);
            } else {
                inputs[numFrames + i] = Util.quantizedExpandDim(quantizedStateMatrix[i]);
            }
        }
        HashMap outputs = new HashMap();

        if (!useQuantizedModel) {
            outputs.put(Integer.valueOf(0), outputLogitMatrix);
            outputs.put(Integer.valueOf(1), outputStatesMatrix);
            //outputs.put(Integer.valueOf(2), kernelMatrix);
        } else {
            outputs.put(Integer.valueOf(0), quantizedOutputLogitMatrix);
            outputs.put(Integer.valueOf(1), quantizedOutputStatesMatrix);
        }



        // Run
        long startTime = SystemClock.uptimeMillis();
        tflite.runForMultipleInputsOutputs(inputs, outputs);
        long endTime = SystemClock.uptimeMillis();
        //Log.d(TAG, "Timecost to run model inference: " + Long.toString(endTime - startTime));

        // Decode
        if (!useQuantizedModel) {
            decodeResult = classifier.ctcDecode(outputLogitMatrix);
        } else {
            decodeResult = classifier.quantizedCtcDecode(quantizedOutputLogitMatrix);
        }
        Log.e(TAG, "RESULT: " + decodeResult);

        // Update state matrix
        for (int i = 0; i < numLayers; i ++) {
            for (int j = 0; j < hiddenSize; j ++) {
                if (!useQuantizedModel) {
                    stateMatrix[i][j] = outputStatesMatrix[i][j];
                } else {
                    quantizedStateMatrix[i][j] = quantizedOutputStatesMatrix[i][j];
                    //System.out.println("state" + (quantizedOutputStatesMatrix[i][j] & 0xFF));
                }
                /*
                if (outputStatesMatrix[i][j] < quantizeStateMin) {
                    quantizeStateMin = outputStatesMatrix[i][j];
                }
                if (outputStatesMatrix[i][j] > quantizeStateMax) {
                    quantizeStateMax = outputStatesMatrix[i][j];
                }
                */
            }
        }
        return evaluate(decodeResult, false);
    }

    public String getDecodeResult() {
        return decodeResult;
    }

    public void clear(){
        classifier.clear();
        if (!useQuantizedModel) {
            for (int i = 0; i < numLayers; i++) {
                Arrays.fill(stateMatrix[i], 0.0f);
            }
            for (int i = 0; i < numFrames; i++) {
                Arrays.fill(outputLogitMatrix[i], 0.0f);
            }
        } else {

            for (int i = 0; i < numLayers; i ++) {
                Arrays.fill(quantizedStateMatrix[i], Util.float2Quantize(0.f, quantizeStateMean, quantizeStateStd));
            }
            for (int i = 0; i < numFrames; i ++) {
                Arrays.fill(quantizedOutputLogitMatrix[i], (byte)0);
            }
        }

        if (floatValues != null) {
            floatValues = null;
        }
        floatValuesRemain = new float[0];
        //Log.e(TAG,"Clean state~~~~~~");
    }

    public void close() {
        tflite.close();
        tflite = null;
    }
}

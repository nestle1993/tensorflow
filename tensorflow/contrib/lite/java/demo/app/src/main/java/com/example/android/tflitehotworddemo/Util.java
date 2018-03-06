package com.example.android.tflitehotworddemo;


import android.util.Log;

import org.jtransforms.fft.FloatFFT_1D;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.lang.Math;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;


public class Util {
    public static float[] byte2float(byte[] b) {
        byte bLength = 2;
        short[] s = new short[b.length / bLength];
        for (int iLoop = 0; iLoop < s.length; iLoop++) {
            byte[] temp = new byte[bLength];
            for (int jLoop = 0; jLoop < bLength; jLoop++) {
                temp[jLoop] = b[iLoop * bLength + jLoop];
            }
            s[iLoop] = byteArrayToShort(temp);
        }
        float[] f = short2float32(s);
        return f;
    }

    public static short byteArrayToShort(byte[] b) {
        return ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).getShort();
    }

    public static float[] short2float32(short[] ar) {
        int n_bytes = 2;
        float dst[] = new float[ar.length];
        float scale = 1.f / (float) (1 << ((8 * n_bytes) - 1));
        for (int i = 0; i < ar.length; i++) {
            dst[i] = scale * ar[i];
        }
        return dst;
    }

    public static float[] concat(float[] first, float[] second) {
        float[] result = Arrays.copyOf(first, first.length + second.length);
        System.arraycopy(second, 0, result, first.length, second.length);
        return result;
    }

    // softmax in-place
    public static void softmax(float[][] ar) {
        int t = ar.length;
        for (int i = 0; i < t; i ++) {
            float expSum = 0.f;
            int h = ar[i].length;
            for (int j = 0; j < h; j ++) {
                expSum += Math.exp(ar[i][j]);
            }
            for (int j = 0; j < h; j ++) {
                ar[i][j] = (float)Math.exp(ar[i][j]) / expSum;
            }
        }
    }

    public static int argmax(float[] ar, int st, int end) {
        float max = ar[st];
        int index = st;
        for (int i = st + 1; i < end; i++) {
            if (max < ar[i]) {
                max = ar[i];
                index = i;
            }
        }
        return index;
    }

    public static float[][] expandDim(float[]ar) {
        float[][] ret = new float[1][ar.length];
        for (int i = 0; i < ar.length; i ++) {
            ret[0][i] = ar[i];
        }
        return ret;
    }

    public static byte[][] quantizedExpandDim(byte[]ar) {
        byte[][] ret = new byte[1][ar.length];
        for (int i = 0; i < ar.length; i ++) {
            ret[0][i] = ar[i];
        }
        return ret;
    }

    public static float vad(float[] sig){
        float sum = 0.f;
        for(float f:sig) {
            sum += Math.abs(f);
        }
        Log.i("sum", String.valueOf(sum));
        return sum;
    }

    public static byte float2Quantize(float v, float mean, float std) {
        return (byte)((int)(v * std + mean) & 0xFF);
    }

    public static byte[][] float2Quantize2D(float[][] v, float mean, float std) {
        byte[][] ret = new byte[v.length][v[0].length];
        for (int i = 0; i < v.length; i ++) {
            for (int j = 0; j < v[i].length; j ++) {
                ret[i][j] = (byte)((int)(v[i][j] * std + mean) & 0xFF);
            }
        }
        return ret;
    }

    private static float[][] matmul2D(float[][] a, float[][] b) {
        int dim1 = a.length;
        int dim2 = b[0].length;
        int dim3 = a[0].length;
        //assert dim3 == b.length;
        float[][] ret = new float[dim1][dim2];
        for (int i = 0; i < dim1; i ++) {
            for (int j = 0; j < dim2; j ++) {
                float sum = 0.f;
                for (int k = 0; k < dim3; k ++) {
                    sum += a[i][k] * b[k][j];
                }
                ret[i][j] = sum;
            }
        }
        return ret;
    }

    private static float[][] customizedSparseMatmul(float[][] a, ArrayList<HashMap<Integer, Float>> b) {
        int dim1 = a.length;
        int dim3 = b.size();
        float[][] ret = new float[dim1][dim3];
        for (int i = 0; i < dim1; i ++) {
            for (int j = 0; j < dim3; j ++) {
                float sum = 0.f;
                for (Map.Entry<Integer, Float> entry : b.get(j).entrySet()) {
                    int index = entry.getKey();
                    float value = entry.getValue();
                    sum += a[i][index] * value;
                }
                ret[i][j] = sum;
            }
        }
        return ret;
    }

    private static float[][] abs(float[] fftOutput) {
        /*
        * Input: n length
        * Output: n / 2 + 1 length
        *
        * if n is even then
        *
        *  ret[0] = fft_output[0]
        *  ret[k] = sqrt(fft_output[k*2]^2 + fft_output[k*2+1]^2), 1 <= k < n/2
        *  ret[n/2] = fft_output[1]
        *
        * if n is odd then
        *
        *  ret[0] = fft_output[0]
        *  ret[k] = sqrt(fft_output[k*2]^2 + fft_output[k*2+1]^2), 1 <= k < (n+1)/2
        *  ret[(n-1)/2] = fft_output[1]
        *
        *  Note that for reducing copy time, we return float matrix with shape [1][freq_size],
        *  which is required as tflite model input.
        */
        int n = fftOutput.length;
        float[][] ret = new float[1][n / 2 + 1];
        ret[0][0] = Math.abs(fftOutput[0]);
        for (int k = 1; k < n / 2; k ++) {
            ret[0][k] = (float)Math.sqrt(
                    Math.pow(fftOutput[k * 2], 2) + Math.pow(fftOutput[k * 2 + 1], 2));
        }
        ret[0][n / 2] = Math.abs(fftOutput[1]);
        return ret;
    }

    private static void addWindowInPlace(float[] signal, float[] window) {
        /* add window in place. */
        for (int i = 0; i < signal.length; i ++) {
            signal[i] *= window[i];
        }
    }

    public static ArrayList<float[][]> stft(float[] signal, int winLength, int hopLength, float[] window) {
        int signalLength = signal.length;
        int numFrames = (int)Math.floor((signalLength - winLength) / hopLength) + 1;
        ArrayList<float[][]> stftMatrix = new ArrayList<>();
        FloatFFT_1D fftUtil = new FloatFFT_1D(winLength);

        float[] fftComplex = new float[winLength];
        for (int i = 0; i < numFrames; i ++) {
            System.arraycopy(signal, i * hopLength, fftComplex, 0, winLength);
            // add window in-place
            addWindowInPlace(fftComplex, window);
            // do fft in-place
            fftUtil.realForward(fftComplex);
            stftMatrix.add(abs(fftComplex));
        }
        return stftMatrix;
    }

    private static byte[][] quantizedAbs(float[] fftOutput, float mean, float std) {
        int n = fftOutput.length;
        byte[][] ret = new byte[1][n / 2 + 1];
        ret[0][0] = float2Quantize(Math.abs(fftOutput[0]), mean, std);
        for (int k = 1; k < n / 2; k ++) {
            ret[0][k] = float2Quantize(
                    (float)Math.sqrt(Math.pow(fftOutput[k * 2], 2) + Math.pow(fftOutput[k * 2 + 1], 2)),
                    mean, std);
        }
        ret[0][n / 2] = float2Quantize(Math.abs(fftOutput[1]), mean, std);
        return ret;
    }

    public static ArrayList<byte[][]> quantizedStft(
            float[] signal, int winLength, int hopLength, float mean, float std) {
        int signalLength = signal.length;
        int numFrames = (int)Math.floor((signalLength - winLength) / hopLength) + 1;
        ArrayList<byte[][]> stftMatrix = new ArrayList<>();
        FloatFFT_1D fftUtil = new FloatFFT_1D(winLength);

        float[] fftComplex = new float[winLength];
        for (int i = 0; i < numFrames; i ++) {
            System.arraycopy(signal, i * hopLength, fftComplex, 0, winLength);
            // do fft in-place
            fftUtil.realForward(fftComplex);
            stftMatrix.add(quantizedAbs(fftComplex, mean, std));
        }
        return stftMatrix;
    }

    public static ArrayList<float[][]> mel(
            float[] signal, int winLength, int hopLength, float[][] melBasis) {
        int signalLength = signal.length;
        int numFrames = (int)Math.floor((signalLength - winLength) / hopLength) + 1;
        ArrayList<float[][]> melMatrix = new ArrayList<>();
        FloatFFT_1D fftUtil = new FloatFFT_1D(winLength);

        float[] fftComplex = new float[winLength];
        for (int i = 0; i < numFrames; i ++) {
            System.arraycopy(signal, i * hopLength, fftComplex, 0, winLength);
            // do fft in-place
            fftUtil.realForward(fftComplex);
            melMatrix.add(matmul2D(abs(fftComplex), melBasis));
        }
        return melMatrix;
    }

    public static ArrayList<byte[][]> quantizedMel(
            float[] signal, int winLength, int hopLength, float mean, float std, float[][] melBasis) {
        int signalLength = signal.length;
        int numFrames = (int)Math.floor((signalLength - winLength) / hopLength) + 1;
        ArrayList<byte[][]> melMatrix = new ArrayList<>();
        FloatFFT_1D fftUtil = new FloatFFT_1D(winLength);

        float[] fftComplex = new float[winLength];
        for (int i = 0; i < numFrames; i ++) {
            System.arraycopy(signal, i * hopLength, fftComplex, 0, winLength);
            // do fft in-place
            fftUtil.realForward(fftComplex);
            melMatrix.add(float2Quantize2D(matmul2D(abs(fftComplex), melBasis), mean, std));
        }
        return melMatrix;
    }

    public static ArrayList<float[][]> melSparse(
            float[] signal, int winLength, int hopLength,
            ArrayList<HashMap<Integer, Float>> melBasisSparse) {
        int signalLength = signal.length;
        int numFrames = (int)Math.floor((signalLength - winLength) / hopLength) + 1;
        ArrayList<float[][]> melMatrix = new ArrayList<>();
        FloatFFT_1D fftUtil = new FloatFFT_1D(winLength);

        float[] fftComplex = new float[winLength];
        for (int i = 0; i < numFrames; i ++) {
            System.arraycopy(signal, i * hopLength, fftComplex, 0, winLength);
            // do fft in-place
            fftUtil.realForward(fftComplex);
            melMatrix.add(customizedSparseMatmul(abs(fftComplex), melBasisSparse));
        }
        return melMatrix;
    }

    public static ArrayList<byte[][]> quantizedMelSparse(
            float[] signal, int winLength, int hopLength, float mean,
            float std, ArrayList<HashMap<Integer, Float>> melBasisSparse, float[] window) {
        int signalLength = signal.length;
        int numFrames = (int)Math.floor((signalLength - winLength) / hopLength) + 1;
        ArrayList<byte[][]> melMatrix = new ArrayList<>();
        FloatFFT_1D fftUtil = new FloatFFT_1D(winLength);

        float[] fftComplex = new float[winLength];
        for (int i = 0; i < numFrames; i ++) {
            System.arraycopy(signal, i * hopLength, fftComplex, 0, winLength);
            // add window in-place
            addWindowInPlace(fftComplex, window);
            // do fft in-place
            fftUtil.realForward(fftComplex);
            melMatrix.add(float2Quantize2D(customizedSparseMatmul(abs(fftComplex), melBasisSparse), mean, std));
        }
        return melMatrix;
    }
}

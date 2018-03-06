package com.example.android.tflitehotworddemo;


import android.util.Log;

import org.jtransforms.fft.FloatFFT_1D;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.lang.Math;
import java.util.ArrayList;
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

    public static float vad(float[] sig){
        float sum = 0.f;
        for(float f:sig) {
            sum += Math.abs(f);
        }
        Log.i("sum", String.valueOf(sum));
        return sum;
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

    public static ArrayList<float[][]> stft(float[] signal, int winLength, int hopLength) {
        int signalLength = signal.length;
        int numFrames = (int)Math.floor((signalLength - winLength) / hopLength) + 1;
        ArrayList<float[][]> stftMatrix = new ArrayList<>();
        FloatFFT_1D fftUtil = new FloatFFT_1D(winLength);

        float[] fftComplex = new float[winLength];
        for (int i = 0; i < numFrames; i ++) {
            System.arraycopy(signal, i * hopLength, fftComplex, 0, winLength);
            // do fft in-place
            fftUtil.realForward(fftComplex);
            stftMatrix.add(abs(fftComplex));
        }
        return stftMatrix;
    }
}

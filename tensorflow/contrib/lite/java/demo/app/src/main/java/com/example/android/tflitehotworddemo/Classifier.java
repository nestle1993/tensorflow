package com.example.android.tflitehotworddemo;

import android.util.Log;

import java.util.Map;


public class Classifier {
    private final float threshold;
    private final int maxKeepLength;
    private final int numClasses;
    private final char otherWordIdx;
    private String accumuResult;

    private int preWord = -1;

    public Classifier(float threshold, int maxKeepLength, int numClasses, char otherWordIdx) {
        this.threshold = threshold;
        this.maxKeepLength = maxKeepLength;
        this.numClasses = numClasses;
        this.otherWordIdx = otherWordIdx;
        this.accumuResult = "";
    }

    public String ctcDecode(float[][] logit) {
        StringBuilder sb = new StringBuilder();
        int t = logit.length;

        // softmax in-place
        Util.softmax(logit);
        for (int i = 0; i < t; i ++) {
            // don't consider space 0 and ctc_blank numClasses - 1
            int pos = Util.argmax(logit[i], 1, numClasses - 1);
            if (logit[i][pos] > threshold) {
                if (preWord == -1 || preWord != pos) {
                    sb.append(pos);
                }
                preWord = pos;
            } else {
                preWord = -1;
            }
            i ++;
        }

        // to avoid repeat keywords
        int j = 0;
        if (accumuResult.length() > 0 && sb.length() > 0) {
            while (accumuResult.charAt(accumuResult.length() - 1) != otherWordIdx &&
                    j < sb.length() &&
                    sb.charAt(j) == accumuResult.charAt(accumuResult.length() - 1)) {
                j ++;
            }
        }
        accumuResult = accumuResult + sb.substring(j);
        if (accumuResult.length() > maxKeepLength)
            accumuResult = accumuResult.substring(accumuResult.length() - maxKeepLength);

        return accumuResult;
    }

    public void clear() {
        accumuResult = "";
    }

}

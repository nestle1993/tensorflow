package com.example.android.tflitehotworddemo;

import android.app.Activity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;

import com.example.android.tflitecamerademo.R;


public class RecordingActivity  extends Activity {
    RecordingThread recordingThread;
    private Button startButton,stopButton;
    private RecordingThread.SpeechCallback callback = new RecordingThread.SpeechCallback() {
        @Override
        public void onSuccess(byte[] result) {

        }

        @Override
        public void onError(byte[] result) {

        }
    };
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_record);
        startButton = (Button)findViewById(R.id.record_start);
        startButton.setEnabled(true);
        stopButton = (Button)findViewById(R.id.record_stop);
        stopButton.setEnabled(false);
        startButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onClickRecord(v);
            }
        });
        stopButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                onClickStopRecord(v);
            }
        });
    }
    public void onClickRecord(View v){
        startButton.setEnabled(false);
        stopButton.setEnabled(true);
        recordingThread = new RecordingThread(this, this);
        recordingThread.startRecording();
    }
    public void onClickStopRecord(View v){
        recordingThread.stopRecording(callback);
        startButton.setEnabled(true);
        stopButton.setEnabled(false);
    }
}

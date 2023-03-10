package com.example.object_detection_app;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.Context;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.Window;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;


public class CameraFaceActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2{
    private static final String TAG="MainActivity";

    private Mat mRgba;
    private Mat mGray;
    private CameraBridgeViewBase mOpenCvCameraView;

    private CascadeClassifier cascadeClassifier;

    private BaseLoaderCallback mLoaderCallback =new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status){
                case LoaderCallbackInterface
                        .SUCCESS:{
                    Log.i(TAG,"OpenCv Is loaded");
                    mOpenCvCameraView.enableView();
                }
                default:
                {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public CameraFaceActivity(){
        Log.i(TAG,"Instantiated new "+this.getClass());
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        int MY_PERMISSIONS_REQUEST_CAMERA=0;
        // if camera permission is not given it will ask for it on device
        if (ContextCompat.checkSelfPermission(CameraFaceActivity.this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(CameraFaceActivity.this, new String[] {Manifest.permission.CAMERA}, MY_PERMISSIONS_REQUEST_CAMERA);
        }

        setContentView(R.layout.activity_camera);

        mOpenCvCameraView=(CameraBridgeViewBase) findViewById(R.id.frame_Surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        //load the model
        try{
            InputStream is=getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
            File cascadeDir=getDir("cascade", Context.MODE_PRIVATE); //creating a folder
            File mCascadeFile=new File(cascadeDir,"haarcascade_frontalface_alt.xml"); //creating a file on that folder
            FileOutputStream os=new FileOutputStream(mCascadeFile);
            byte[] buffer=new byte[4096];
            int byteRead;
            //writing that file from raw folder
            while((byteRead=is.read(buffer))!= -1){
                os.write(buffer,0,byteRead);
            }
            is.close();
            os.close();

            //loading file from cascade folder created above
            cascadeClassifier=new CascadeClassifier(mCascadeFile.getAbsolutePath());
            //model loaded

        }
        catch (IOException e){
            Log.i(TAG,"Cascade file not found");
        }

    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()){
            //if load success
            Log.d(TAG,"Opencv initialization is done");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        else{
            //if not loaded
            Log.d(TAG,"Opencv is not loaded. try again");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0,this,mLoaderCallback);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView !=null){
            mOpenCvCameraView.disableView();
        }
    }

    public void onDestroy(){
        super.onDestroy();
        if(mOpenCvCameraView !=null){
            mOpenCvCameraView.disableView();
        }

    }

    public void onCameraViewStarted(int width ,int height){
        mRgba=new Mat(height,width, CvType.CV_8UC4);
        mGray =new Mat(height,width,CvType.CV_8UC1);
    }
    public void onCameraViewStopped(){
        mRgba.release();
    }
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame){

        mRgba=inputFrame.rgba();
        mGray=inputFrame.gray();

        // in processing pass mRgba to cascaderec class

        mRgba=CascadeRec(mRgba);

        return mRgba;

    }

    private Mat CascadeRec(Mat mRgba){
        // original frame is -90 degree so we have to rotate is to 90 to get proper face for detection
        Core.flip(mRgba.t(),mRgba,1);

        //convert it into RGB
        Mat mRbg=new Mat();
        Imgproc.cvtColor(mRgba,mRbg,Imgproc.COLOR_RGBA2RGB);

        int height=mRbg.height();
        //minimum size of face in frame
        int absoluteFaceSize=(int)(height*0.1);

        MatOfRect faces=new MatOfRect();
        if(cascadeClassifier !=null)
        {
            cascadeClassifier.detectMultiScale(mRbg,faces,1.1,2,2,
                    new Size(absoluteFaceSize,absoluteFaceSize),new Size());
        }

        //loop through all faces
        Rect[] facesArray=faces.toArray();
        for( int i=0;i<facesArray.length;i++){
            //draw face on original from mRgba
            Imgproc.rectangle(mRgba,facesArray[i].tl(),facesArray[i].br(),new Scalar(0,255,0,255),2);

        }
        //rotate back original frame to -90 degree
        Core.flip(mRgba.t(),mRgba,0);

        return mRgba;
    }



}





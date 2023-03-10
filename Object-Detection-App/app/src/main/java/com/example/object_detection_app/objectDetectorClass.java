package com.example.object_detection_app;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.widget.LinearLayout;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.StringTokenizer;
import java.util.TreeMap;

public class objectDetectorClass {

    //this is used to load model and predict
    private Interpreter interpreter;
    //store all label in array
    private List<String> labelList;
    private int INPUT_SIZE;
    private int PIXEL_SIZE=3;//for RGB
    private int IMAGE_MEAN=0;
    private float IMAGE_STD=255.0f;
    //use to initialize gpu in app
    private GpuDelegate gpuDelegate;
    private int height=0;
    private int width=0;


    objectDetectorClass(AssetManager assetManager, String modelPath, String labelPath,int inputSize) throws IOException{
        INPUT_SIZE=inputSize;
        //use to define gpu or cpu //no. of threads
        Interpreter.Options options=new Interpreter.Options();
        gpuDelegate=new GpuDelegate();
        options.addDelegate(gpuDelegate);
        //set it according to the device used for my Samsung A51 there are 3 threads
        options.setNumThreads(3);
        //loading model
        interpreter=new Interpreter(loadModelFile(assetManager,modelPath),options);
        //load labelmap
        labelList=loadLabelList(assetManager,labelPath);
    }

    private List<String> loadLabelList(AssetManager assetManager, String labelPath)throws IOException {
        //to store label
        List<String> labelList=new ArrayList<>();
        //create a new reader
        BufferedReader reader=new BufferedReader(new InputStreamReader(assetManager.open(labelPath)));
        String line;
        //loop through each line and store it to labelList
        while((line=reader.readLine())!=null){
            labelList.add(line);
        }
        reader.close();
        return labelList;


    }

    private ByteBuffer loadModelFile(AssetManager assetManager, String modelPath)throws IOException {
        //used to get description of file
        AssetFileDescriptor fileDescriptor=assetManager.openFd(modelPath);
        FileInputStream inputStream=new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel=inputStream.getChannel();
        long startOffset=fileDescriptor.getStartOffset();
        long declaredLenght=fileDescriptor.getDeclaredLength();

        return fileChannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLenght);
    }

    public Mat recognizeImage(Mat mat_image){

        //Rotate original image by 90 degree to get portrait frame
        Mat rotated_mat_image=new Mat();
//        Core.flip(mat_image.t(),rotated_mat_image,1);

        Mat a=mat_image.t();
        Core.flip(a,rotated_mat_image,1);
        a.release();
        //if this is not done we will get improper predictions, less no of object
        //convert to bitmap
        Bitmap bitmap=null;
        bitmap=Bitmap.createBitmap(rotated_mat_image.cols(),rotated_mat_image.rows(),Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(rotated_mat_image,bitmap);

        //define height and width
        height=bitmap.getHeight();
        width=bitmap.getWidth();

        //scale the bitmap to input size of the model
        Bitmap scaledBitmap=Bitmap.createScaledBitmap(bitmap,INPUT_SIZE,INPUT_SIZE,false);

        //convert bitmap to bytebuffer as model input should be in
        ByteBuffer byteBuffer=convertBitmapToByteBuffer(scaledBitmap);


        //defining output
        //10 : top 10 object detected
        //4 : their coordinates
//        float[][][]result=new float[1][10][4];   //this method of output does not work apparently so we will create a treemap of three arrays (boxes,score,classes)
        Object[]input=new Object[1];
        input[0]=byteBuffer;


        Map<Integer,Object> output_map=new TreeMap<>();
        float[][][] boxes=new float[1][10][4];
        //10 : top 10 object detected
        //4 : their coordinates
        float[][]scores=new float[1][10];
        //stores scores of 10 objects
        float[][]classes=new float[1][10];
        //stores class of object

        //add it to object_map;
        output_map.put(0,boxes);
        output_map.put(1,classes);
        output_map.put(2,scores);

        //predict
        interpreter.runForMultipleInputsOutputs(input,output_map);

        //draw boxes and label preparations
        Object value=output_map.get(0);
        Object Object_class=output_map.get(1);
        Object score=output_map.get(2);

        //loop through each object
        //as outputs has only 10 boxes

        for(int i=0;i<10;i++){
            float class_value=(float) Array.get(Array.get(Object_class,0),i);
            float score_value=(float) Array.get(Array.get(score,0),i);

            //define threshold for score
            if(score_value>0.5){
                Object box1=Array.get(Array.get(value,0),i);

                //multiplying with Original height and width of frame
                float top=(float) Array.get(box1,0)*height;
                float left=(float) Array.get(box1,1)*width;
                float bottom=(float) Array.get(box1,2)*height;
                float right=(float) Array.get(box1,3)*width;

                //draw rectangle in Original frame  start point          end point of box         color of box             thickness
                Imgproc.rectangle(rotated_mat_image,new Point(left,top), new Point(right,bottom), new Scalar(255,155,155),2);

                //write text on frame
                //                                string of class name of object   start point                                     color of text       size of text
                Imgproc.putText(rotated_mat_image,labelList.get((int) class_value),new Point(left, top),3,1,new Scalar(100,100,100),2);

            }
        }


        //Before returning rotate back by -90 degree
//        Core.flip(rotated_mat_image.t(),mat_image,0);  #this does not work for some reason

        Mat b=rotated_mat_image.t();
        Core.flip(b,mat_image,0);
        b.release();

        return mat_image;
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {

        //https://stackoverflow.com/questions/17256094/what-does-floatpar4-16-255-255-0f-mean  info about method

        ByteBuffer byteBuffer;
        //some model input should be quant=0 others quant=1
        //this one is quant=0

        int quant=0;
        int size_images=INPUT_SIZE;
        if(quant==0){
            byteBuffer=ByteBuffer.allocateDirect(1*size_images*size_images*3);
        }
        else{
            byteBuffer=ByteBuffer.allocateDirect(4*1*size_images*size_images*3);
        }
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] intValues=new int[size_images*size_images];
        bitmap.getPixels(intValues,0,bitmap.getWidth(),0,0,bitmap.getWidth(),bitmap.getHeight());
        int pixel =0;
        for(int i=0;i<size_images;++i){
            for(int j=0;j<size_images;++j){
                final int val=intValues[pixel++];
                if(quant==0){
                    byteBuffer.put((byte) ((val>>16)&0xFF));
                    byteBuffer.put((byte) ((val>>8)&0xFF));
                    byteBuffer.put((byte) (val&0xFF));
                }
                else{
                    byteBuffer.putFloat((((val>>16)&0xFF))/255.0f);
                    byteBuffer.putFloat((((val>>8)&0xFF))/255.0f);
                    byteBuffer.putFloat((((val)&0xFF))/255.0f);
                }
            }
        }

        return byteBuffer;
    }


}

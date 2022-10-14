package com.example.dementiadetection;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.example.dementiadetection.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;

public class MainActivity extends AppCompatActivity {
    private Button select, predict;
    private ImageView imageView;
    private TextView output;
    private Bitmap img;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        select = findViewById(R.id.select);
        predict = findViewById(R.id.predict);
        imageView=findViewById(R.id.brainimg);
        output=findViewById(R.id.output);

        select.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent, 100);
            }
        });

        predict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                if (img!=null){
                    img=Bitmap.createScaledBitmap(img, 128,128,true);
                    try {
                        Model model = Model.newInstance(getApplicationContext());

                        // Creates inputs for reference.
                        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 128, 128, 3}, DataType.FLOAT32);

                        TensorImage tensorImage  = new TensorImage(DataType.FLOAT32);
                        tensorImage.load(img);
                        ByteBuffer byteBuffer = tensorImage.getBuffer();


                        inputFeature0.loadBuffer(byteBuffer);

                        // Runs model inference and gets result.
                        Model.Outputs outputs = model.process(inputFeature0);
                        TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
                        if (outputFeature0.getFloatArray()[0]==1.0){
                            output.setText("Dementia PRESENT");
                        }
                        else{
                            output.setText("Dementia ABSENT");
                        }

                        //output.setText(outputFeature0.getFloatArray()[0] + "\n"+ outputFeature0.getFloatArray()[0]);
                        // Releases model resources if no longer used.
                        model.close();
                    } catch (IOException e) {
                        // TODO Handle the exception
                    }

                }
                else{
                    Toast.makeText(MainActivity.this,"Please Select Image First", Toast.LENGTH_SHORT).show();
                }

            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable @org.jetbrains.annotations.Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode==100){
            imageView.setImageURI(data.getData());

            Uri uri = data.getData();

            try {
                img=MediaStore.Images.Media.getBitmap(this.getContentResolver(),uri);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
}
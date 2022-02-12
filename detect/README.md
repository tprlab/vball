# Volleybal game stages detection

Lets consider volleybal game process as flow of 4 stages:
- no play
- play
- stand (like before serve)
- cheering

The idea is to detect people and classify their positions to determine the current stage

## Requirements

- Python3
- numpy
- opencv-contrib-python
- tensorflow
- sklearn

## Usage
1. First extracting frames from video:

  *ffmpeg -i video.mp4 -r 1 frames/%05d.jpg*
  
2. Then run MobileSSD detector over the frames and write the detections in json format into folder *json*
 
  Use my another repo - [detectppl](https://github.com/tprlab/detectppl) as a reference
  
3. Call 

  ```
  data = test_utils.read_data("json")
  ```
  
  Then array of masks will be generated
  
 4. Use this array as input for classification:

```
pred = predict_knn.predict(data)
```

or

```
pred = predict_keras.predict(data)
```

5. Call **test_utils.collect_rallies** to generate rallies list

## Training

First 2 steps are the same:

1. First extracting frames from video:

  *ffmpeg -i video.mp4 -r 1 frames/%05d.jpg*
  
2. Then run MobileSSD detector over the frames and write the detections in json format into folder *json*
 
  Use my another repo - [detectppl](https://github.com/tprlab/detectppl) as a reference

3.Assuming the detections are in *json* folder, ajust variable W and H (input frames width and height) in *test_utils.py* and run:

    gen_data.py
    
    
    After that there should be a folder *data* with image masks
    
 4. Classify images by manually puting them into different folder:
 - noplay
 - play
 - stand
 - cheer

 5. Run **train_keras.py** or **train_knn.py**

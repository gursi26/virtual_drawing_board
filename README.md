# Virtual drawing board
Drawing on a virtual canvas using hand pose estimation

### Camera demo
![](imgs/demo.gif)

### Libraries used
<hr>
- Pytorch 
- Mediapipe 
- OpenCV

To test the whiteboard, follow the following steps:
- Open the file `draw_vid.py`. Look for the variable named `model_path` and change it to point to the file `120.pt` in the ***models*** folder.
- Navigate into the project folder using your terminal and run the following command :

```
python draw_vid.py
```
- Clenching a fist with your index finger sticking out allows you to **draw**.

<img src ="imgs/draw.png" width="260px" />

- Closing your hand into a fist allows you to **erase**, where the red box represents the bounds of the eraser.

<img src ="imgs/erase.png" width="260px" />

- Holding your hand with all of your fingers open does nothing.

<img src ="imgs/none.png" width="260px" />

### References
<hr>
[Mediapipe repository](https://github.com/google/mediapipe.git)
[Hand pose estimation tutorial](https://www.youtube.com/watch?v=NZde8Xt78Iw&t=983s)

Licensed under the [MIT License](LICENSE)

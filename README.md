# Virtual drawing board
Drawing on a virtual canvas using hand pose estimation

### Camera demo
![](imgs/demo.gif)
<hr>

### Libraries used
- Pytorch 
- Mediapipe 
- OpenCV

To test the whiteboard, run `main.py`.

- Clenching a fist with your index finger sticking out allows you to **draw**.

<img src ="imgs/draw.png" width="260px" />

- Closing your hand into a fist allows you to **erase**, where the red box represents the bounds of the eraser.

<img src ="imgs/erase.png" width="260px" />

- Holding your hand with all of your fingers open does nothing.

<img src ="imgs/none.png" width="260px" />
<hr>

### References

- [Mediapipe repository](https://github.com/google/mediapipe.git)<br>
- [Hand pose estimation tutorial](https://www.youtube.com/watch?v=NZde8Xt78Iw&t=983s)

Licensed under the [MIT License](LICENSE)

<h1 align="center"> Virtual drawing board ✍️</h1>

<h2 align="center">


![python](https://forthebadge.com/images/badges/made-with-python.svg)


[![PyTorch](https://img.shields.io/badge/-PyTorch-white?style=flat-square&logo=PyTorch)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/-OpenCV-blueviolet?style=flat-square&logo=OpenCV)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/-MediaPipe-white?style=flat-square&logo=Clyp)](https://mediapipe.dev/)
[![license](https://img.shields.io/github/license/gursi26/virtual_drawing_board.svg?style=flat-square)](https://github.com/gursi26/virtual_drawing_board/blob/main/LICENSE)

</h2>

### Drawing on a virtual canvas using hand pose estimation


</br>
<h3 align="center">

![](imgs/demo.gif)

</h3>
</br>
<hr>


## Libraries used
- PyTorch 
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

## References

- [Mediapipe repository](https://github.com/google/mediapipe.git)
- [Hand pose estimation tutorial](https://www.youtube.com/watch?v=NZde8Xt78Iw&t=983s)

<h3 align="right">

Licensed under the [MIT License](LICENSE)

</h3>

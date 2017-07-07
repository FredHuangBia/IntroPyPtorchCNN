# Run the example
First you need to make sure the python version is **python3**
Then install these libraries: **tqdm, numpy, pytorch, scipy**
```sh
$ pip3 install http://download.pytorch.org/whl/cu80/torch-0.1.12.post2-cp35-cp35m-linux_x86_64.whl 
$ pip3 install tqdm
$ pip3 install numpy
$ pip3 install scipy
```

tqdm is the beautiful progress-bar library
numpy is a linear algebra and matrix library
scipy is an image processing library

If any of the above failed during installation, please try again with 'sudo'.

After cloning this repo, copy the **IMG** folder into the directory **data**. The **IMG** folder is on our GPU under /home/titan/Fred/

Now you are done! Run the **main.py** to start training your neural network!
```sh
$ python3 main.py
```

If you want to watch the car driving it own or generate your own training data, please follow the instruction here: 
https://github.com/udacity/self-driving-car-sim

As well as the video tutorial here: (Notice that they were using other DL library originally)
https://www.youtube.com/watch?v=EaY5QiZwSP4

Have fun! Try to make a small and powerful model and run a full loop!
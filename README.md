# MNIST Digit Classifier

This project is a continuation of chapter 4 in the Fastai textbook.  In notes.md you will find the binary case being discussed (from the book) then the multi-class case is reasoned and implemented (from scratch).

## Data

Got the data from Kaggle in .csv form.  It was the first/easiest place I could grab all of the digits from since Fastai only had digits 3 and 7.  I recommend finding an easier method, as you may notice, collating the data is a headache.  

## Notebooks

Most of the notebooks are just working files to test things out.  I recommend looking at Benchmarked MNIST if you want to see a practical implementation of broadcasting.   

## MNIST.py

Contains the cleanest version of the model running on MNIST data.  In it you will find a neural network class and a trainer class.

## To run

`pip3 install requirements.txt` then `python3 MNIST.py`

## Future plans

I plan on taking the skeleton of this model and porting to plain Pytorch -> tinygrad.  I need to get a better handle on Pytorch first.  I think the port to Pytorch will be trivial. The port to tinygrad may be more difficult. 






# datagen

These python scripts are used to multiply the data from the VOC2012 dataset.

To generate the data, simply create the text file using write_text.py to create the batches of the dataset.
Then run the dataset_gen.py to create the dataset.

Please check the path in the files before running the script

To run write_text.py, simply run the command 'python write_text.py <number of data per file>'.
<number of data per file> is the number of the data per file you want.
Ex: 'python write_text.py 500' will generate text files each contain 500 data.

To run dataset_gen.py, simply run the command 'python dataset_gen.py --text <file>'.
<file> is the name of the file that write_text.py generated.
Ex: 'python dataset_gen.py --text 1' will generate a numpy data from text file 1

The dataset_gen.py will generate 3 outputs of numpy array: image data, bounding box data, and class data.
Image data will be in format (n, 3, 200, 200)
Bounding box data will be in format (n, 50, 4)
Class data will be in format (n, 50)

Class data is:


# INSA Lyon - Deep Learning Project

## Usage

### Train the network

```sh
python3 train.py
```

### Evaluate the classifier

```sh
python3 test.py
```

### Find faces on full images

```sh
python3 face_finder.py input_image.jpg [output_image_path.png]
```

### Other datasets used

#### Faces

-   [Labelled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/#download)

Download the _All images aligned with deep funneling_ dataset then convert it using the `lfw_convert.sh` script.

#### Non-faces

-   [Kaggle Natural Images (with the Person class excluded)](https://www.kaggle.com/prasunroy/natural-images)

Download the dataset, remove the `person` folder (or use it like the LFW dataset), then use the `archives/nofaces.py` util to generate false-positive images.

docker build . -t tomtom:1.0
docker run --rm -v $(pwd):/tomtom tomtom:1.0 --preprocess=1 --retrain=0 --predict=0 --prevId='TomTom_2020_11_20_15_51_32' --train='/tomtom/data/train.csv' --test='/tomtom/data/test.csv' --outpath='/tomtom/data/'
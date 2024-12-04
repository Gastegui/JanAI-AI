# Where to get the datasets

## Food-101 (5GB)
curl -L -o ~/Downloads/food-101.tar.gz http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz && tar -xvf ~/Downloads/food-101.tar.gz 

## FooDD (5GB)
curl -L -o ~/Downloads/fooDD.zip https://www.kaggle.com/api/v1/datasets/download/rusqi29/food-detection-dataset-for-calorie-measurement && unzip ~/Downloads/fooDD.zip

## Tipical Brazilian Foods (136MB)

curl -L -o ~/Downloads/26-tipical-brazilian-foods.zip\ https://www.kaggle.com/api/v1/datasets/download/sc0v1n0/26-tipical-brazilian-foods && unzip ~/Downloads/26-tipical-brazilian-foods.zip

Then, replace all the non-ASCII characters into ASCII caracters and the whitespaces with underscores (Acarajé -> Acaraje, Pão de queijo -> Pao_de_queijo)
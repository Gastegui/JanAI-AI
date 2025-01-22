# Where to get the datasets

## Food-101 (5GB)
    cd ~/Downloads && curl -L -o ./food-101.tar.gz http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz && tar -xvf ./food-101.tar.gz && rm ./food-101.tar.gz

### The next ones are not used for anything anymore

## FooDD (5GB)
    cd ~/Downloads && curl -L -o ./fooDD.zip https://www.kaggle.com/api/v1/datasets/download/rusqi29/food-detection-dataset-for-calorie-measurement && unzip ./fooDD.zip && rm ./fooDD.zip

## Tipical Brazilian Foods (136MB)

    cd ~/Downloads && mkdir tipical_brazilian_foods && cd tipical_brazilian_foods && curl -L -o ./26-tipical-brazilian-foods.zip https://www.kaggle.com/api/v1/datasets/download/sc0v1n0/26-tipical-brazilian-foods && unzip ./26-tipical-brazilian-foods.zip && rm ./26-tipical-brazilian-foods.zip

Then, replace all the non-ASCII characters into ASCII caracters and the whitespaces with underscores (Acarajé -> Acaraje, Pão de queijo -> Pao_de_queijo)
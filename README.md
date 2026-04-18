# DDoS Detection and Mitigation with Machine Learning and Ryu SDN

## Description

The purpose of this project is to create a module for the Ryu SDN Controller that can detect in-progress DoS attacks using machine learning.

## Getting Started

### Dependencies

This project is separated into two parts: ML training, and the Ryu Module. Since Ryu is no longer maintained, Ryu needs a specific version of python and some packages. However the ML portion is run using the most recent version of python(3.14). 

#### ML Training Environment Dependencies
```
Python 3.14
Pip 26.0.1
```
You shouldn't *need* to specify versions, but specifying them anyway for posterity. 
```
pip install pandas==3.0.2
pip install numpy==2.4.4
pip install matplotlib==3.10.8
pip install joblib==1.5.3
pip install scikit-learn==1.8.0
```

#### Ryu Environment Dependencies
```
Python 3.9.7
```
```
pip install setuptools==67.6.1
python -m pip install -U pip==20.3.4
pip install ryu
pip install eventlet==0.30.2
```
After installing these ryu should work.
To test if ryu is working run:
```
ryu-manager --version
```
This should return:
```
ryu-manager 4.34
```
### Installation

write more here

### Executing program

* write more


## Author

Ian Scheetz
[link](https://github.com/ian-1979)

## License

This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments

Training data provided by:
[link](https://github.com/neelimabonangi/Ddos-detection-sdn-ml/tree/master)

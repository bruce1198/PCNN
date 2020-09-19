# PCNN

### Usage

1. Model setting can be found in DL_config.xlsx
- YOLOv2
- AlexNet
- VGG16
- VGG19

2. Create the folders data and models. 
```
PCNN
└─── data  
└─── models
| DL_config.xlsx
│ main.py  
│ ...
│ README.md
```
3. run `alex.py` to get the basic model in
```
PCNN
└─── models
| └─── model
| ...
| README.md
```
4. run `python main.py` to get the slicing information.
- YOLOv2 -> data/prefetch0.json
- AlexNet -> data/prefetch1.json
- VGG16 &nbsp;&nbsp;-> data/prefetch2.json
- VGG19 &nbsp;&nbsp;-> data/prefetch3.json
5. load the weights from model you have created and build the sliced model in `slice.py`

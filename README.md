## Deep Learning for Vanishing Point Detection Using an Inverse Gnomonic Projection

### Requirements
* Python 2.7
* [Caffe 1.0-RC5](https://github.com/BVLC/caffe/tree/rc5)
* ImageMagick 6.8.8-1
* what the ``requirements.txt`` says

Using other versions of these packages may yield different results.

### Setup
Get the code, install requirements and build LSD:
```
git clone https://www.tnt.uni-hannover.de:3000/kluger/Vanishing_Points_GCPR17.git
cd <folder>
pip install -r requirements.txt
cd lsdpython
python setup.py build_ext --inplace
```



### Examples

![example](assets/figure3.jpg)

![example](assets/figure4.jpg)

![example](assets/figure1.jpg)

![example](assets/figure2.jpg)



### References

If you use the code provided here, please cite:
```
@inproceedings{kluger2017deep,
  title={Deep learning for vanishing point detection using an inverse gnomonic projection},
  author={Kluger, Florian and Ackermann, Hanno and Yang, Michael Ying and Rosenhahn, Bodo},
  booktitle={German Conference on Pattern Recognition (GCPR)},
  year={2017}
}
```
The paper can be found on [arXiv](https://arxiv.org/abs/1707.02427).

The benchmark datasets used in the paper can be found here:
* [York Urban Dataset](http://www.elderlab.yorku.ca/resources/york-urban-line-segment-database-information/)
* [Eurasian Cities Dataset](http://graphics.cs.msu.ru/en/research/projects/msr/geometry)
* [Horizon Lines in the Wild](http://www.cs.uky.edu/~jacobs/datasets/hlw/)

The example images show landmarks in Hannover, Germany:
* [Welfenschloss](https://www.flickr.com/photos/shepard4711/40441168973/in/photolist-8KSNgP-24BDyA2-dHQyus-RsMpqr-qfCQuS-91DBT-5aprBZ-7uavc5-7u6BEk-7u6Agv-7u6zBR-7ua1Am-7u6mU4-7u6cdk-7u657F)
* [Lichthof at Leibniz University](https://commons.wikimedia.org/wiki/File:Atrium_Lichthof_main_building_Welfenschloss_Leibniz_Universitatet_Hannover_Am_Welfengarten_Nordstadt_Hannover_Germany.jpg)
* [Ihme-Zentrum](https://commons.wikimedia.org/wiki/Category:Ihme-Zentrum?uselang=de#/media/File:Ihme-Zentrum_Spinnereistrasse_Hanover_Germany.jpg)
* [Nord LB](https://www.flickr.com/photos/dierkschaefer/5999546112/in/photolist-6EywNo-pdpBA8-a97hon-eQ6474-a9acHm-a9a9AG-a9af3d-R5SNyF-a97tck-eQhHCJ-fruEuZ-eQi2tE-eQhk8d-qnVgrW-24fRi2L-eQhyxE-bymrtQ-kU7Apk-a9a74Y-2bxix-PRf3sv-SXwgoU-dyUjRC-jbB22-rgmqm-24awG1H-4zjzyq-TMEpHD-Rer4CD-rt82Av-rgiWa)

The Python implementation of the LSD line segment detector was taken from here: https://github.com/xanxys/lsd-python
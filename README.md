
## Hand Detector
The hand detectors are trained on (1) 100K and (2) 100K+ego images from 100DOH dataset. 

### Performance
<!-- ROW: faster_rcnn_X_101_32x8d_FPN_3x -->
<table><rbody>
<tr>
<tr><td align="center">Name</td>
<td align="center">Data</td>
<td align="center">Box AP</td>
<td align="center">Model</td>
</tr>

<tr>
<td align="left"><a href="faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml">Faster-RCNN X101-FPN</a></td>
<td align="left">100K</td>
<td align="center">90.32%</td>
<td align="center"><a href="https://drive.google.com/file/d/1o6-zmZTehpLozAOibm2uqUu--WIKh88R/view?usp=sharing">Google Drive</a></td>
</tr>

<tr>
<td align="left"><a href="faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml">Faster-RCNN X101-FPN</a></td>
<td align="left">100K+ego</td>
<td align="center">90.46%</td>
<td align="center"><a href="https://drive.google.com/file/d/1OqgexNM52uxsPG3i8GuodDOJAGFsYkPg/view?usp=sharing">Google Drive</a></td>
</tr>

</tbody></table>

### Environment
- Set up detectron2 environment as in [install.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md)


### Train
```
CUDA_VISIBLE_DEVICES=4,5,6,7 python trainval_net.py --num-gpus 4 --config-file faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml
```


### Evaluation
```
CUDA_VISIBLE_DEVICES=4,5,6,7 python trainval_net.py --num-gpus 4 --config-file faster_rcnn_X_101_32x8d_FPN_3x_100DOH.yaml --eval-only MODEL.WEIGHTS path/to/model.pth
```

### Demo
```
CUDA_VISIBLE_DEVICES=1 python demo.py
```

### Citation

If this work is helpful in your research, please cite:
```
@INPROCEEDINGS{Shan20, 
    author = {Shan, Dandan and Geng, Jiaqi and Shu, Michelle  and Fouhey, David},
    title = {Understanding Human Hands in Contact at Internet Scale},
    booktitle = CVPR, 
    year = {2020} 
}
```
When you use the model trained on our ego data, make sure to also cite the original datasets ([Epic-Kitchens](https://epic-kitchens.github.io/2018), [EGTEA](http://cbs.ic.gatech.edu/fpv/) and [CharadesEgo](https://prior.allenai.org/projects/charades-ego)) that we collect from and agree to the original conditions for using that data.
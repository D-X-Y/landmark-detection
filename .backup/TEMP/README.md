# Landmark Detection

```
sh scripts/300W/batch.sh 3
sh scripts/style-robust/batch.sh 0 WGAN
```

###
Baseline CPM
```
3.959 7.024 4.559
```

Baseline HG-V1 ADAM-V1
```
3.309 5.637 3.765
```

Baseline HG-V1 ADAM-V2
```
3.330 5.620 3.779
```

```
bash scripts/augm/300W_HG_ADAM.sh 0
bash scripts/style-robust/300W-Style-LSGAN.sh 0 instance 0 DET
bash scripts/300W/300W_CPM_SGD.sh 0 V1 V1 DET
bash scripts/300W/300W_HG_ADAM.sh 0 V1 V3 DET
```

### Robust-Main
```
bash scripts/style-robust/300W-Robust.sh 2 instance
```

### Visualization
python ./exps/vis.py --meta ./SNAPSHOT/eval-epoch-109-110-01-03.pth --save ./SNAPSHOT/01/

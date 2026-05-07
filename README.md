# **Hierarchical Symmetric Normalization Registration Using Deformation‑Inverse Network**

This repository provides the official implementation of **Hierarchical Symmetric Normalization Registration Using Deformation‑Inverse Network**, a learning‑based framework for symmetric deformable image registration. The method jointly estimates forward and inverse transformations and enforces inverse‑consistency through a hierarchical architecture.

---

## **Training**

To train your own model, you may need to adapt the data‑loading utilities in `voxelmorph/functions.py` depending on your dataset structure and file formats. The provided training script can run out‑of‑the‑box as long as the input directory follows a consistent structure and all training images share the same spatial dimensions.

A typical dataset directory should look like:

```
data/
└── subject/
    ├── image.nii.gz
    └── seg.nii.gz
```

Given a dataset directory `./data` and an output directory `./output`, you can train the model using:

```bash
python ./train.py --datapath ./data --model-dir ./output
```

Model checkpoints will be saved to the directory specified by `--model-dir`.

---

## **Registration**

To register a pair of images using a trained model, use the `test.py` script and specify the corresponding model weights. For example, assuming the forward and inverse models are stored as:

- `./data/exp2/0400.pt`
- `./data/exp2/0400_InverseNet.pt`

You can run:

```bash
python ./test.py \
    --datapath ./data \
    --model-dir ./output \
    --load-model ./data/exp2/0400.pt \
    --load_model_InverseNet ./data/exp2/0400_InverseNet.pt
```

This will generate the registered image and the associated deformation fields.

---

## **Citation**

If you find this repository helpful in your research, please consider citing:

```
@inproceedings{sha2024hierarchical,
  title={Hierarchical symmetric normalization registration using deformation-inverse network},
  author={Sha, Qingrui and Sun, Kaicong and Xu, Mingze and Li, Yonghao and Xue, Zhong and Cao, Xiaohuan and Shen, Dinggang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={662--672},
  year={2024},
  organization={Springer}
}
```

---

## **Reference**

This implementation builds upon the VoxelMorph framework:  
[https://github.com/voxelmorph/voxelmorph](https://github.com/voxelmorph/voxelmorph)


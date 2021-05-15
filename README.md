# super-resolutuion-with-partial-conv

**Used code from [pytorch-inpainting-with-partial-conv](https://github.com/naoto0804/pytorch-inpainting-with-partial-conv).**


This is an unofficial pytorch implementation of a paper, [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/pdf/1804.07723.pdf)

## Requirements
- Python 3.6+
- Pytorch 0.4.1+

```
pip install -r requirements.txt
```

## Usage

### Preprocess 
- download [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and place it somewhere. The dataset should contain `data_large`, and `test_large` as the subdirectories. Don't forget to specify the root of the dataset by `--root ROOT` when using `train.py` or `test.py`

- Generate masks by generate_mask.py **Note that the way of the mask generation is different from the original work**
```
python generate_data.py
```

### Train
```
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py
```

### Fine-tune
```
CUDA_VISIBLE_DEVICES=<gpu_id> python train.py --finetune --resume <checkpoint_name>
```
### Test
```
CUDA_VISIBLE_DEVICES=<gpu_id> python test.py --snapshot <snapshot_path>
```

## References
- [1]: [Image Inpainting for Irregular Holes Using Partial Convolutions](https://arxiv.org/pdf/1804.07723.pdf)
- [2]: [pytorch-inpainting-with-partial-conv](https://github.com/naoto0804/pytorch-inpainting-with-partial-conv#pytorch-inpainting-with-partial-conv)
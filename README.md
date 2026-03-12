# [NTIRE 2026 Challenge on Real-World Face Restoration](https://cvlai.net/ntire/2026/) @ [CVPR 2026](https://cvpr.thecvf.com/)

[![ntire](https://img.shields.io/endpoint.svg?url=https%3A%2F%2Fraw.githubusercontent.com%2Fjkwang28%2FNTIRE2026_RealWorld_Face_Restoration%2Fmain%2Ffigs%2Fdiamond_badge.json)](https://www.cvlai.net/ntire/2026/)
[![page](https://img.shields.io/badge/Project-Page-blue?logo=github&logoSvg)](https://ntire-face.github.io/2026/)
[![visitors](https://visitor-badge.laobi.icu/badge?page_id=jkwang28.NTIRE2026_RealWorld_Face_Restoration&right_color=violet)](https://github.com/jkwang28/NTIRE2026_RealWorld_Face_Restoration)
[![GitHub Stars](https://img.shields.io/github/stars/jkwang28/NTIRE2026_RealWorld_Face_Restoration?style=social)](https://github.com/jkwang28/NTIRE2026_RealWorld_Face_Restoration)

## About the Challenge

This challenge focuses on restoring real-world degraded face images. The task is to recover high-quality face images with rich high-frequency details from low-quality inputs. At the same time, the output should preserve facial identity to a reasonable degree. There are no restrictions on computational resources such as model size or FLOPs. The main goal is to **achieve the best possible image quality and identity consistency**. 
Participants are ranked based on visual quality while ensuring identity similarity above a threshold; final scores combine several no-reference IQA metrics and FID. 

## Challenge results
**Test Set** – 450 low-quality (LQ) images drawn from five real-world subsets (WIDER-Test, WebPhoto-Test, CelebChild-Test, LFW-Test, and CelebA) are provided for evaluation.  

**Identity Validation** – Cosine similarity is measured with a pretrained **AdaFace** model. Thresholds: 0.30 (WIDER & WebPhoto), 0.60 (LFW & CelebChild), 0.50 (CelebA). A submission fails if more than ten faces fall below the dataset-specific threshold.  

**Metrics** – Valid submissions are scored with six no-reference metrics: **CLIPIQA, MANIQA, MUSIQ, Q-Align, NIQE,** and **FID** (against FFHQ).  

**Overall Score**

$$
\text{Score} = \text{CLIPIQA} + \text{MANIQA} + \frac{\text{MUSIQ}}{100} + \max\left(0, \frac{10 - \text{NIQE}}{10}\right) + \frac{\text{QALIGN}}{5} + \max\left(0, \frac{100-\text{FID}}{100}\right).
$$

**Ranking rule** – Teams are first screened by the identity filter; qualifying entries are ranked descending by the overall score. Minor deviations between Codalab and reproduced scores are tolerated after code verification.  

**Resources** – Official evaluation scripts, pretrained models, and baseline code are available in this public repository.  

## About this repository

This repository summarizes the solutions submitted by the participants during the challenge. The model script and the pre-trained weight parameters are provided in the [models](./models) and [model_zoo](./model_zoo) folders. Each team is assigned a number according to the submission time of the solution. You can find the correspondence between the number and team in [test.select_model](./test.py). Some participants would like to keep their models confidential. Thus, those models are not included in this repository.

## How to test the baseline model?

The val data and test data could be downloaded at [Google Drive](https://drive.google.com/drive/folders/1ruyMYFuZNAQb9ntN77KhV86os1icZpU5?usp=sharing). 

1. `git clone https://github.com/jkwang28/NTIRE2026_RealWorld_Face_Restoration.git`
2. Download the sample model (CodeFormer, Team00) weights from [Google Drive](https://drive.google.com/drive/folders/1dg-R6JiNGM9jXyrf8ndd2kpHOPUvRjgb) and put the downloaded weights into `./model_zoo/team00_CodeFormer` folder.
3. Select the model you would like to test:
    ```bash
    CUDA_VISIBLE_DEVICES=0 python test.py --valid_dir [path to val data dir] --test_dir [path to test data dir] --save_dir [path to your save dir] --model_id 0
    ```
    - You can use either `--valid_dir`, or `--test_dir`, or both of them. Be sure the change the directories `--valid_dir`/`--test_dir` and `--save_dir`.
    - Switch models (default is CodeFormer) through commenting the code in [test.py](./test.py#L19).

## How to add your model?

> [!IMPORTANT]
>
> **🚨 Submissions that do not follow the official format will be rejected.**

1. Register your team in the [Google Spreadsheet](https://docs.google.com/spreadsheets/d/168HxDVVHaMp5d5F-WCkVKNavDAmGKLmA1mJlB5thAOY/edit?usp=drive_link) and get your team ID.
2. Put your the code of your model in folder:  `./models/[Your_Team_ID]_[Your_Model_Name]`

   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02
3. Put the pretrained model in folder: `./model_zoo/[Your_Team_ID]_[Your_Model_Name]`

   - Please zero pad [Your_Team_ID] into two digits: e.g. 00, 01, 02
   - Note: Please provide a download link for the pretrained model, if the file size exceeds **100 MB**. Put the link in `./model_zoo/[Your_Team_ID]_[Your_Model_Name]/[Your_Team_ID]_[Your_Model_Name].txt`: e.g. [team00_CodeFormer.txt](./model_zoo/team00_CodeFormer/team00_CodeFormer.txt)
4. Add your model to the model loader `test.py` as follows:

   - Edit the `else` to `elif` in [test.py](./test.py#L24), and then you can add your own model with model id.

   - `model_func` **must** be a function, which accept **4 params**. 

     - `model_dir`: the pretrained model. Participants are expected to save their pretrained model in `./model_zoo/` with in a folder named `[Your_Team_ID]_[Your_Model_Name]` (e.g., team00_CodeFormer). 

     - `input_path`: a folder contains several images in PNG format. 

     - `output_path`: a folder contains restored images in PNG format. Please follow the section Folder Structure. 

     - `device`: computation device.
5. Send us the command to download your code, e.g,

   - `git clone [Your repository link]`
   - We will add your code and model checkpoint to the repository after the challenge.

> [!TIP]
>
> Your model code does not need to be fully refactored to fit this repository. 
> Instead, you may add a lightweight external interface (e.g., `models.team00_CodeFormer.inference_codeformer.main`) that wraps your existing code, while keeping the original implementation unchanged.
>
> Refer to previous NTIRE challenge implementations for examples: 
> https://github.com/zhengchen1999/NTIRE2025_RealWorld_Face_Restoration/tree/main/models

## How to eval images using NR-IQA metrics and facial ID?

### Environments

```sh
conda create -n NTIRE-FR python=3.8
conda activate NTIRE-FR
pip install -r requirements.txt
```

### Metrics include:
1. **NIQE**, **CLIP-IQA**, **MANIQA**, **MUSIQ**, **Q-Align**  
   - Provided by [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch)

2. **FID** (using FFHQ as reference)  
   - Provided by [VQFR](https://github.com/TencentARC/VQFR)
3. **Facial ID Consistency**
   - Model Provided by [AdaFace](https://github.com/mk-minchul/AdaFace)

### Pretrained Weights
You should first create a folder named `pretrained` in the root directory and download the following weights into it:

- adaface_ir50_ms1mv2.ckpt ([Google Drive](https://drive.google.com/file/d/1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing))
- inception_FFHQ_512.pth ([Google Drive](https://drive.google.com/drive/folders/1k3RCSliF6PsujCMIdCD1hNM63EozlDIZ?usp=sharing))

### Folder Structure
The val data and test data could be downloaded at [Google Drive](https://drive.google.com/drive/folders/1ruyMYFuZNAQb9ntN77KhV86os1icZpU5?usp=sharing). 
```
input_LQ_dir
├── test
│   ├── CelebA
│   ├── CelebChild-Test
│   ├── ...
├── val
│   ├── CelebA
│   ├── CelebChild-Test
│   ├── ...
    
output_dir_test
├── CelebA
├── CelebChild-Test
├──...
output_dir_val
├── CelebA
├── CelebChild-Test
├──...
```

### Command to calculate metrics

```sh
python eval.py \
--mode "test" \
--output_folder "/path/to/your/output_dir_test" \
--lq_ref_folder "/path/to/input_LQ_dir" \
--metrics_save_path "./IQA_results" \
--gpu_ids 0 \
--use_qalign True 
```

The `eval.py` file accepts the following 6 parameters:
- `mode`: Choose whether to test images from the `test` set or the `val` set.
- `output_folder`: Path where the restored images are saved. Subdirectories should be organized by dataset names.
- `lq_ref_folder`: Path to the LQ images provided as input to the model. This path should be the parent directory of the `test` and `val` sets.
- `metrics_save_path`: Directory where the evaluation metrics will be saved.
- `device`: Computation devices. For multi-GPU setups, use the format `0,1,2,3`.
- `use_qalign`: Whether to use Q-Align or not, which will consume an additional 15GB of GPU memory.

### Weighted score

We use the following equation to calculate the final weighted score: 

$$
\text{Score} = \text{CLIPIQA} + \text{MANIQA} + \frac{\text{MUSIQ}}{100} + \max\left(0, \frac{10 - \text{NIQE}}{10}\right) + \frac{\text{QALIGN}}{5} + \max\left(0, \frac{100-\text{FID}}{100}\right).
$$

The score is calculated on the averaged IQA scores on all the val/test datasets. 

## NTIRE Real-world Face Restoration Challenge Series

Code repositories and accompanying technical report PDFs for each edition:  

- **NTIRE 2025**: [Github Repo](https://github.com/zhengchen1999/NTIRE2025_RealWorld_Face_Restoration) | [Report](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Chen_NTIRE_2025_Challenge_on_Real-World_Face_Restoration_Methods_and_Results_CVPRW_2025_paper.pdf) | [arXiv](https://arxiv.org/abs/2504.14600)

## Citation

If you find the code helpful in your research or work, please cite the following paper(s).

```
@inproceedings{ntiface26face,
  title={NTIRE 2026 challenge on real-world face restoration: Methods and results},
  author={Wang, Jingkai and Gong, Jue and Chen, Zheng and Liu, Kai and Li, Jiatong and Timofte, Radu and Zhang, Yulun and others},
  booktitle={CVPRW},
  year={2026}
}
```

## License and Acknowledgement

This code repository is release under [MIT License](LICENSE). 

Several implementations are taken from: [IQA-PyTorch](https://github.com/chaofengc/IQA-PyTorch), [VQFR](https://github.com/TencentARC/VQFR), [AdaFace](https://github.com/mk-minchul/AdaFace). 

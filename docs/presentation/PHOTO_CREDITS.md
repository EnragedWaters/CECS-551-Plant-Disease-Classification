# Photograph and visual credits

These nine small presentation illustrations use real dataset photos and recorded model outputs. No AI-generated photographs are used.
Photos retain the full frame: orientation correction, size reduction and JPEG encoding only. The model-input and feature displays are identified separately.

## Dataset credits

- **plantvillage:** S. P. Mohanty / PlantVillage. [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/). [Pinned source](https://huggingface.co/datasets/mohanty/PlantVillage/tree/9e97599868962bd0079b8db4b7f1efa9185fa1e7).
- **plantdoc:** D. Singh et al. / cropped repository by P. Kayal et al.. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Pinned source](https://github.com/pratikkayal/PlantDoc-Dataset/tree/5467f6012d78d1c446145d5f582da6096f852ae8).

The photo derivatives retain their source licenses. The dataset declarations do not independently establish the original provenance of every internet-sourced PlantDoc photograph. Visible image credits are preserved.

## Illustrations

| ID | Dataset label / display | File |
| --- | --- | --- |
| pv-apple-01 | Apple / Apple scab | [pv-apple-01.jpg](assets/pv-apple-01.jpg) |
| pv-apple-02 | Apple / Apple scab | [pv-apple-02.jpg](assets/pv-apple-02.jpg) |
| pd-apple-01 | Apple / Apple scab | [pd-apple-01.jpg](assets/pd-apple-01.jpg) |
| pd-apple-02 | Apple / Apple scab | [pd-apple-02.jpg](assets/pd-apple-02.jpg) |
| pv-grape-01 | Grape / Black rot | [pv-grape-01.jpg](assets/pv-grape-01.jpg) |
| pd-grape-01 | Grape / Black rot | [pd-grape-01.jpg](assets/pd-grape-01.jpg) |
| prepared-apple | Original preprocessing display | [prepared-apple.png](assets/prepared-apple.png) |
| late-frozen | Actual seed-551 final-block channel 0; shared color scale; not an attention map. | [late-frozen.png](assets/late-frozen.png) |
| late-finetune | Actual seed-551 final-block channel 0; shared color scale; not an attention map. | [late-finetune.png](assets/late-finetune.png) |

Exact relative source paths, original checksums, derivative checksums, dimensions and dataset revisions are in [assets/manifest.json](assets/manifest.json). Photo IDs appear in slide footers; notes give full source paths. The process diagram uses the original seed-551 CPU replay. Predicted category names agree with the original saved evaluation. Examples were selected after testing and are not an accuracy estimate.

To recreate these derivatives from the existing local evidence:

```bash
python scripts/prepare_presentation_assets.py --data-root /path/to/datasets --replay /path/to/replay.json
```

The deck builder uses these checked-in presentation derivatives; it does not require raw datasets or model weights.

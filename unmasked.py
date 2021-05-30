#!/usr/bin/env python3

import os
from pathlib import Path

import click
import cv2


@click.group()
def main():
    pass

@main.command("unmask")
#  @click.option("--gpus", type=int, default=0, help="number of GPUs to use")
@click.option("--checkpoint", type=str, default="./models/masks_ver3_model.ckpt")
@click.argument("input_image")
@click.argument("mask_image")
def unmask(input_image, mask_image, checkpoint):
    import torch
    from network import SNPatchGAN

    model_checkpoint_path = Path(checkpoint)
    if not model_checkpoint_path.exists():
        print("checkpoint path doesn't exist")
        os.exit(1)

    model = SNPatchGAN()
    model.load_from_checkpoint(checkpoint)
    model.freeze()

@main.command("export-model")
@click.argument("checkpoint", type=str, default="./models/masks_ver3_model.ckpt")
@click.argument("output", default=Path(os.getcwd()) / "unmasked-model.onnx")
def export_model(checkpoint, output):
    import torch
    from network import SNPatchGAN

    model_checkpoint_path = Path(checkpoint)
    if not model_checkpoint_path.exists():
        print("checkpoint path doesn't exist")
        os.exit(1)

    model = SNPatchGAN()
    model.load_from_checkpoint(checkpoint)

    #  model.to_onnx(
    #      output,
    #      example_outputs=(torch.ones((48, 3, 128, 128)), torch.ones((48, 3, 128, 128))), # output: coarse, refined
    #      export_params=True,
    #      opset_version=13,
    #  )
    script = model.to_torchscript()
    torch.jit.save(script, output)


@main.command("mask-face")
@click.option("--mask-type", type=str, default="random", show_default=True, help="Type of face mask to apply to the image")
@click.argument("image")
@click.argument("output_dir", default=os.getcwd())
def mask_face(image, mask_type, output_dir):
    import mask_the_face

    output_dir_path = Path(output_dir)
    image_path = Path(image)
    if not output_dir_path.exists():
        print("output directory doesn't exist")
        os.exit(1)
    if not image_path.exists():
        print("image not found")
        os.exit(1)

    masker = mask_the_face.Masker()
    try:
        masked_image, mask, _, _ = masker.apply_mask_file(str(image_path), mask_type=mask_type)
    except Exception as e:
        print("face masking failed")
        os.exit(2)

    masked_image_filename = image_path.stem + "-face_masked" + image_path.suffix
    mask_image_filename = image_path.stem + "-mask" + image_path.suffix

    cv2.imwrite(str(output_dir_path / masked_image_filename), masked_image)
    cv2.imwrite(str(output_dir_path / mask_image_filename), mask)

#  main.add_command(mask_face)

if __name__ == "__main__":
    main()

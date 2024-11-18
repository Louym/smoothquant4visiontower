import argparse

from termcolor import colored

import llava
from llava import conversation as clib
from llava.media import Image, Video
from llava.utils.media import extract_media
from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from llava.mm_utils import process_image, process_images
from smoothquant.calibration import get_static_decoder_layer_scales
from smoothquant.smooth import smooth_lm
import torch
import os

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "-m", type=str, default="Efficient-Large-Model/vila-qwen2-vl-7b-instruct-sft-video-256f-t8-20241031145522")
    parser.add_argument("--conv-mode", "-c", type=str, default="auto")
    parser.add_argument("--act_scales_path", type=str, default="/home/yuming/workspace/qwen/models/smoothquant/act_scales/nvila_siglip_video.pt")
    # parser.add_argument("--quant_path", type=str, default="/home/yuming/workspace/qwen/models/smoothquant/nvila_siglip_w8a8.pt")
    # parser.add_argument("--smooth_model_path", type=str, default="/home/yuming/workspace/qwen/models/smoothquant/nvila_siglip_smooth.pt")
    parser.add_argument("--decoder_layer_scales_path", type=str, default="/home/yuming/workspace/qwen/models/smoothquant/nvila_siglip_decoder_layer_scales.pt")
    # parser.add_argument("--media", type=str, default=["/home/yuming/workspace/vila_demo_video.mp4"])
    parser.add_argument("--media", type=str, default=["/home/yuming/workspace/space_woaudio.mp4"])
    args = parser.parse_args()

    # Load model
    model = llava.load(args.model_path,devices=[0])
    del model.llm
    del model.mm_projector
    torch.cuda.empty_cache()
    model=model.cuda().eval()
    get_act_scale=True
    quant=False
    if get_act_scale:
        prompt = []
        if args.media is not None:
            for media in args.media or []:
                if any(media.endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                    media = Image(media)
                elif any(media.endswith(ext) for ext in [".mp4", ".mkv", ".webm"]):
                    media = Video(media)
                else:
                    raise ValueError(f"Unsupported media type: {media}")
                prompt.append(media)
        conversation=[{"from": "human", "value": prompt}]
        media = extract_media(conversation, model.config)
        for name in media:
            if name == "image":
                if len(media["image"]) == 1 and model.config.image_aspect_ratio == "dynamic":
                    model.config.image_processor = model.vision_tower.image_processor
                    images = process_image(media["image"][0], model.config, None, enable_dynamic_res=True).half()
                    conversation[0]["value"] = conversation[0]["value"].replace(
                        DEFAULT_IMAGE_TOKEN, f"{DEFAULT_IMAGE_TOKEN}\n" * images.shape[0]
                    )
                else:
                    images = process_images(media["image"], model.vision_tower.image_processor, model.config).half()
                media[name] = [image for image in images]
            elif name == "video":
                print(model.vision_tower.image_processor)
                media[name] = [
                    process_images(images, model.vision_tower.image_processor, model.config).half()
                    for images in media[name]
                ]
            else:
                raise ValueError(f"Unsupported media type: {name}")
        print(model)
        images = torch.cat(media["video"], dim=1)
        print(images.shape)
        model.vision_tower=model.vision_tower.eval()
        # from smoothquant.calibration import get_act_scales
        # act_scales=get_act_scales(model.vision_tower,images)
    #     print(act_scales["vision_tower.vision_model.encoder.layers.26.self_attn.q_proj"])
    #     os.makedirs(os.path.dirname(args.act_scales_path), exist_ok=True)
    #     torch.save(act_scales, args.act_scales_path)

    # else:
    #     smooth_lm(model.vision_tower, act_scales, 0.85)
    #     model.save_pretrained(args.smooth_model_path)
    #     print(f"Saved smoothed model at {args.smooth_model_path}")
    # # if quant:
    #     from smoothquant.smooth import smooth_lm
    #     act_scales = torch.load(args.act_scales_path)
        # from smoothquant.fake_quant import quantize_llama_like
        # smooth_lm(model.vision_tower, act_scales, 0.5)
        # model_smoothquant_w8a8 = quantize_llama_like(model.vision_tower)
        decoder_layer_scales, raw_scales = get_static_decoder_layer_scales(model.vision_tower,images)
        os.makedirs(os.path.dirname(args.decoder_layer_scales_path), exist_ok=True)
        torch.save(decoder_layer_scales, args.decoder_layer_scales_path)
        # print(decoder_layer_scales)
        # int8_model = Int8OPTForCausalLM.from_float(model, decoder_layer_scales)
        # int8_model.save_pretrained(args.quant_path)
        # print(f"Saved int8 model at {args.quant_path}")
if __name__ == "__main__":
    main()

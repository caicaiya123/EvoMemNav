# Adapted from ConceptGraphs (https://github.com/concept-graphs/concept-graphs), MIT License.
# Only the functions used by EvoMemNav are kept.

# from line_profiler import profile
# from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
import torch
from PIL import Image

# def get_sam_predictor(cfg) -> SamPredictor:
#     if cfg.sam_variant == "sam":
#         sam = sam_model_registry[cfg.sam_encoder_version](checkpoint=cfg.sam_checkpoint_path)
#         sam.to(cfg.device)
#         sam_predictor = SamPredictor(sam)
#         return sam_predictor

#     if cfg.sam_variant == "mobilesam":
#         from MobileSAM.setup_mobile_sam import setup_model
#         # MOBILE_SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/mobile_sam.pt")
#         # checkpoint = torch.load(MOBILE_SAM_CHECKPOINT_PATH)
#         checkpoint = torch.load(cfg.mobile_sam_path)
#         mobile_sam = setup_model()
#         mobile_sam.load_state_dict(checkpoint, strict=True)
#         mobile_sam.to(device=cfg.device)

#         sam_predictor = SamPredictor(mobile_sam)
#         return sam_predictor

#     elif cfg.sam_variant == "lighthqsam":
#         from LightHQSAM.setup_light_hqsam import setup_model
#         HQSAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./EfficientSAM/sam_hq_vit_tiny.pth")
#         checkpoint = torch.load(HQSAM_CHECKPOINT_PATH)
#         light_hqsam = setup_model()
#         light_hqsam.load_state_dict(checkpoint, strict=True)
#         light_hqsam.to(device=cfg.device)

#         sam_predictor = SamPredictor(light_hqsam)
#         return sam_predictor

#     elif cfg.sam_variant == "fastsam":
#         raise NotImplementedError
#     else:
#         raise NotImplementedError


# Prompting SAM with detected boxes in a batch


# Prompting SAM with detected boxes in a batch


# @profile
def compute_clip_features_batched(
    image,
    detections,
    clip_model,
    clip_preprocess,
    clip_tokenizer,
    classes,
    device,
    prompt_h=None,
    prompt_w=None,
):

    image = Image.fromarray(image)
    padding = 20  # Adjust the padding amount as needed

    image_crops = []
    preprocessed_images = []
    text_tokens = []

    original_width, original_height = image.size
    if prompt_h is not None and prompt_w is not None:
        scale_w, scale_h = prompt_w / original_width, prompt_h / original_height
    else:
        scale_w, scale_h = 1, 1

    # Prepare data for batch processing
    for idx in range(len(detections.xyxy)):
        x_min, y_min, x_max, y_max = detections.xyxy[idx]
        image_width, image_height = image.size
        left_padding = min(padding, x_min)
        top_padding = min(padding, y_min)
        right_padding = min(padding, image_width - x_max)
        bottom_padding = min(padding, image_height - y_max)

        # get the image crop without padding
        image_crops.append(image.crop((x_min, y_min, x_max, y_max)))

        x_min -= left_padding
        y_min -= top_padding
        x_max += right_padding
        y_max += bottom_padding

        preprocessed_image = clip_preprocess(
            image.crop((x_min, y_min, x_max, y_max))
        ).unsqueeze(0)
        preprocessed_images.append(preprocessed_image)

        class_id = detections.class_id[idx]
        text_tokens.append(classes[class_id])

    # Convert lists to batches
    preprocessed_images_batch = torch.cat(preprocessed_images, dim=0).to(device)

    # Batch inference
    with torch.no_grad():
        image_features = clip_model.encode_image(preprocessed_images_batch)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # text_features = clip_model.encode_text(text_tokens_batch)
        # text_features /= text_features.norm(dim=-1, keepdim=True)

    # Convert to numpy
    image_feats = image_features.cpu().numpy()
    # text_feats = text_features.cpu().numpy()
    # image_feats = []
    text_feats = []

    # resize image crops
    if prompt_h is not None and prompt_w is not None:
        for i in range(len(image_crops)):
            crop_w, crop_h = image_crops[i].size
            image_crops[i] = image_crops[i].resize(
                (int(crop_w * scale_w), int(crop_h * scale_h))
            )

    return image_crops, image_feats, text_feats

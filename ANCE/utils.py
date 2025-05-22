from transformers import AutoTokenizer, T5ForConditionalGeneration, CLIPProcessor, AutoModel,CLIPVisionModel
from multi_model import MultiModal

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def load_model(args,device):
    select_layer=args.select_layer
    clip_model_name=args.clip_model_name
    t5_model_name=args.t5_model_name
    t5_tokenizer = AutoTokenizer.from_pretrained(t5_model_name)
    t5_model = AutoModel.from_pretrained(t5_model_name)
    t5_tokenizer.add_special_tokens(
        {"additional_special_tokens": [DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN]})
    t5_model.resize_token_embeddings(len(t5_tokenizer))
    t5_tokenizer.pad_token = t5_tokenizer.eos_token
    image_processor = CLIPProcessor.from_pretrained(clip_model_name)
    model = MultiModal(clip_model_name, t5_model, t5_tokenizer, args.lm_type, select_layer)
    model = model.to(device)
    return t5_tokenizer, model, image_processor

def get_img_patch_token_size(clip_model_name):
    clip_model = CLIPVisionModel.from_pretrained(clip_model_name)
    image_size=clip_model.config.image_size
    patch_size=clip_model.config.patch_size
    img_patch_token_size=int(image_size/patch_size)**2
    return img_patch_token_size


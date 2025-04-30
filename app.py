#!/usr/bin/env python

import os
import random
import uuid
import gradio as gr
import numpy as np
from PIL import Image
import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
from deep_translator import GoogleTranslator
import time
import logging
from huggingface_hub import hf_hub_download
import gc
import webbrowser

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DESCRIPTION = """
# تحويل النص إلى صورة - v5
تطوير: حسام فضل قدور

التعليمات:
1. أدخل وصفاً للصورة التي تريد إنشاءها
2. يمكنك استخدام اللغة العربية أو الإنجليزية
3. اضبط الخيارات المتقدمة حسب الحاجة
4. انقر على زر Create 
"""

MAX_SEED = np.iinfo(np.int32).max
DEFAULT_NEGATIVE_PROMPT = """(مشوه، محرف، مشوه:1.3)، رسم سيئ، تشريح سيئ، تشريح خاطئ، 
طرف إضافي، طرف مفقود، أطراف عائمة، (تشوه الأيدي والأصابع:1.4)، 
أطراف مفصولة، تشوه، مشوه، قبيح، مقزز، غير واضح، بتر، (NSFW:1.25)"""

# تنظيم الأمثلة حسب الفئات
prompt_categories = {
    "الأسلوب الفني": [
        "واقعي عالي الدقة",
        "رسم زيتي كلاسيكي",
        "رسم رقمي عصري",
        "أسلوب كرتوني",
        "تصوير فوتوغرافي احترافي"
    ],
    "الإضاءة والألوان": [
        "إضاءة دراماتيكية",
        "ألوان زاهية",
        "تدرجات الباستيل",
        "أبيض وأسود",
        "إضاءة طبيعية ناعمة"
    ],
    "التفاصيل التقنية": [
        "8K UHD",
        "عالي التفاصيل",
        "تقنية RAW",
        "حدة عالية",
        "عمق ميداني ضحل"
    ],
    "العناصر الإضافية": [
        "خلفية مموهة",
        "انعكاسات ضوئية",
        "ظلال ناعمة",
        "تأثير بوكيه",
        "تألق خفيف"
    ],
    "أمثلة كاملة": [
        "كابيبارا يرتدي بدلة أنيقة ويحمل لافتة مكتوب عليها 'مرحبًا بالعالم'",
        "بطاقة دعوة مزخرفة بألوان ذهبية وأرجوانية",
        "منشور إعلاني لحفل موسيقي",
        "غلاف كتاب خيالي",
        "إعلان لمطعم فاخر"
    ]
}

# إعداد مجلد الكاش والمخرجات
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

css = """
footer {visibility: hidden}
.container {max-width: 1200px; margin: auto; padding: 20px;}
.gr-button {
    background-color: #2196F3;
    border-radius: 8px;
    transition: background-color 0.3s ease;
}
.gr-button:hover {background-color: #1976D2;}
.gr-form {
    padding: 20px;
    border-radius: 10px;
    background-color: #f5f5f5;
}
.output-image {
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.prompt-builder {
    display: flex;
    flex-direction: column;
    gap: 15px;
    padding: 15px;
    background: #f8f9fa;
    border-radius: 10px;
    margin: 10px 0;
}
.category-container {
    background: white;
    padding: 10px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
}
.category-title {
    font-size: 1.1em;
    font-weight: bold;
    margin-bottom: 10px;
    color: #2196F3;
}
.prompt-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}
.prompt-chip {
    background-color: #e3f2fd;
    border: 1px solid #90caf9;
    border-radius: 15px;
    padding: 5px 12px;
    font-size: 0.9em;
    cursor: pointer;
    transition: all 0.2s ease;
    color: #1976d2;
}
.prompt-chip:hover {
    background-color: #90caf9;
    color: white;
    transform: translateY(-1px);
}
.prompt-input {
    border: 2px solid #e3f2fd;
    border-radius: 8px;
    padding: 12px;
    font-size: 1em;
    width: 100%;
    margin-top: 10px;
}
"""

# إعداد رابط الـ favicon
favicon_html = """
<link rel="icon" type="image/png" href="favicon_v2.png">
"""

def generate_sequential_filename(base_dir=OUTPUT_DIR):
    counter = 1
    while True:
        filename = os.path.join(base_dir, f"{counter}.png")
        if not os.path.exists(filename):
            return filename
        counter += 1

def optimize_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def get_device_settings():
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        total_memory = device_props.total_memory / (1024**3)
        
        if total_memory < 6:
            return {
                "device": "cuda",
                "dtype": torch.float16,
                "memory_efficient": True,
                "max_size": 512,
                "batch_size": 1,
                "attention_slice": True
            }
        elif total_memory < 8:
            return {
                "device": "cuda",
                "dtype": torch.float16,
                "memory_efficient": True,
                "max_size": 768,
                "batch_size": 1,
                "attention_slice": True
            }
        else:
            return {
                "device": "cuda",
                "dtype": torch.float16,
                "memory_efficient": False,
                "max_size": 1024,
                "batch_size": 1,
                "attention_slice": False
            }
    else:
        return {
            "device": "cpu",
            "dtype": torch.float32,
            "memory_efficient": True,
            "max_size": 512,
            "batch_size": 1,
            "attention_slice": False
        }

def initialize_model():
    try:
        optimize_memory()
        device_settings = get_device_settings()
        logger.info(f"إعدادات الجهاز: {device_settings}")
        
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "fluently/Fluently-XL-Final",
            torch_dtype=device_settings["dtype"],
            use_safetensors=True,
            low_cpu_mem_usage=True,
            cache_dir=CACHE_DIR
        )
        
        if device_settings["device"] == "cuda":
            pipe = pipe.to("cuda", torch_dtype=device_settings["dtype"])
            
            if device_settings["memory_efficient"]:
                if hasattr(pipe, 'enable_model_cpu_offload'):
                    pipe.enable_model_cpu_offload()
                if hasattr(pipe, 'enable_sequential_cpu_offload'):
                    pipe.enable_sequential_cpu_offload()
                if hasattr(pipe, 'enable_vae_tiling'):
                    pipe.enable_vae_tiling()
                if hasattr(pipe, 'enable_vae_slicing'):
                    pipe.enable_vae_slicing()
                if device_settings["attention_slice"]:
                    pipe.enable_attention_slicing(1)
                
                if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                    torch.cuda.set_per_process_memory_fraction(0.7)
        else:
            pipe = pipe.to("cpu")
            
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True
        )
        
        for attempt in range(3):
            try:
                pipe.load_lora_weights(
                    "ehristoforu/dalle-3-xl-v2",
                    weight_name="dalle-3-xl-lora-v2.safetensors",
                    adapter_name="dalle"
                )
                pipe.set_adapters("dalle")
                logger.info("تم تحميل LoRA بنجاح")
                break
            except Exception as e:
                if attempt == 2:
                    logger.warning(f"فشل تحميل LoRA بعد 3 محاولات: {e}")
                else:
                    time.sleep(1)
                    continue
        
        return pipe
    except Exception as e:
        error_msg = f"خطأ في تهيئة النموذج: {str(e)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def add_to_prompt(current_prompt, new_text):
    """إضافة نص جديد إلى البرومبت الحالي"""
    if not current_prompt:
        return new_text
    return f"{current_prompt}, {new_text}"

def translate_to_english(text: str) -> str:
    """
    Translate text to English with robust error handling.
    
    Args:
        text (str): Input text to translate
    
    Returns:
        str: Translated text or original text if translation fails
    """
    try:
        # First attempt: Google Translator
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        return translated
    except Exception as e:
        logger.warning(f"Translation failed with Google Translator: {e}")
        
        try:
            # Fallback: Simple transliteration or keep original
            return text  # Keeping original text as a safe fallback
        except Exception as translation_error:
            logger.error(f"Complete translation failure: {translation_error}")
            return text

def validate_prompt(prompt: str) -> bool:
    """
    Validate the input prompt to prevent potential misuse.
    
    Args:
        prompt (str): Input text prompt
    
    Returns:
        bool: Whether the prompt is valid
    """
    # Basic validation to prevent empty or extremely short prompts
    if not prompt or len(prompt.strip()) < 3:
        return False
    
    # Optional: Add more sophisticated prompt validation
    # For example, check for inappropriate content
    inappropriate_keywords = ['nsfw', 'explicit', 'violent']
    if any(keyword in prompt.lower() for keyword in inappropriate_keywords):
        return False
    
    return True

def open_output_folder():
    output_dir = os.path.abspath(OUTPUT_DIR)
    if os.path.exists(output_dir):
        webbrowser.open(f"file://{output_dir}")
    return f"تم فتح المجلد: {output_dir}"

def adjust_image_dimensions(width: int, height: int, max_size: int = 2048) -> tuple:
    """
    Intelligently adjust image dimensions to maintain aspect ratio and prevent excessive sizes.
    
    Args:
        width (int): Original width
        height (int): Original height
        max_size (int): Maximum allowed dimension
    
    Returns:
        tuple: Adjusted (width, height)
    """
    # Ensure dimensions are multiples of 8 for model compatibility
    def round_to_multiple_of_8(x):
        return max(64, min(max_size, (x // 8) * 8))
    
    # Maintain aspect ratio
    aspect_ratio = width / height
    
    if width > height:
        # Landscape
        new_width = min(width, max_size)
        new_height = round_to_multiple_of_8(int(new_width / aspect_ratio))
    else:
        # Portrait or Square
        new_height = min(height, max_size)
        new_width = round_to_multiple_of_8(int(new_height * aspect_ratio))
    
    return new_width, new_height

def get_style_prompt(style_preset: str) -> str:
    """
    Generate additional prompt based on selected artistic style.
    
    Args:
        style_preset (str): Selected artistic style
    
    Returns:
        str: Additional descriptive prompt for the style
    """
    STYLE_PROMPTS = {
        "واقعي": "realistic, photorealistic, highly detailed, 8k resolution",
        "كرتوني": "cartoon style, vibrant colors, cute, animated illustration",
        "رسم زيتي": "oil painting, artistic, textured brushstrokes, masterpiece",
        "رسم رقمي": "digital art, clean lines, modern graphic design, vector style",
        "سينمائية": "cinematic, dramatic lighting, film-like composition, epic scene"
    }
    return STYLE_PROMPTS.get(style_preset, "")

def generate(
    prompt: str,
    negative_prompt: str = "",
    use_negative_prompt: bool = False,
    seed: int = 0,
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3,
    randomize_seed: bool = False,
    style_preset: str = "",
    num_images: str = "1",
    progress=gr.Progress(track_tqdm=True),
):
    """
    Generate images with intelligent dimension handling and style integration.
    
    Args:
        prompt (str): Text description for image generation
        ... (other arguments remain the same)
    
    Returns:
        list: Generated image paths or error message
    """
    try:
        # Validate input prompt
        if not validate_prompt(prompt):
            raise ValueError("الرجاء إدخال وصف صالح وكافٍ للصورة")

        # Intelligent dimension adjustment
        adjusted_width, adjusted_height = adjust_image_dimensions(width, height)
        logger.info(f"Adjusted Dimensions: {adjusted_width}x{adjusted_height}")

        # Translate and enhance prompt with style
        english_prompt = translate_to_english(prompt)
        style_enhancement = get_style_prompt(style_preset)
        full_prompt = f"{english_prompt}, {style_enhancement}".strip(", ")
        logger.info(f"Enhanced Prompt: {full_prompt}")

        # Memory optimization
        optimize_memory()
        device_settings = get_device_settings()

        # Seed handling
        if randomize_seed:
            seed = random.randint(0, MAX_SEED)
        
        generator = torch.Generator(device=device_settings['device']).manual_seed(seed)

        # Negative prompt handling
        effective_negative_prompt = (
            DEFAULT_NEGATIVE_PROMPT if use_negative_prompt 
            else (negative_prompt or DEFAULT_NEGATIVE_PROMPT)
        )

        # Image generation
        num_inference_steps = 50  # Configurable based on quality vs. speed
        images = pipe(
            prompt=full_prompt,
            negative_prompt=effective_negative_prompt,
            width=adjusted_width,
            height=adjusted_height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            num_images_per_prompt=int(num_images)
        ).images

        # Save generated images
        output_images = []
        for idx, image in enumerate(images, 1):
            filename = generate_sequential_filename(base_dir=OUTPUT_DIR)
            image.save(filename)
            output_images.append(filename)
            logger.info(f"Saved image: {filename}")

        return output_images

    except Exception as e:
        error_message = f"خطأ في توليد الصورة: {str(e)}"
        logger.error(error_message)
        return [error_message]  # Return error as a list to maintain interface compatibility

def create_interface():
    """
    Create a comprehensive Gradio interface for text-to-image generation.
    
    Returns:
        gr.Blocks: Configured Gradio interface
    """
    with gr.Blocks(css=css) as demo:
        gr.Markdown(favicon_html + DESCRIPTION)
        
        # Main input and generation components
        with gr.Column():
            # Prompt input
            prompt = gr.Textbox(
                label="وصف الصورة",
                placeholder="أدخل وصفاً للصورة التي تريد إنشاءها...",
                lines=3
            )
            
            # Prompt builder section
            gr.Markdown("### منشئ البرومبت")
            with gr.Column() as prompt_builder:
                for category, items in prompt_categories.items():
                    gr.Markdown(f"#### {category}")
                    with gr.Row():
                        for item in items:
                            chip_btn = gr.Button(
                                item, 
                                variant="secondary", 
                                size="sm"
                            )
                            chip_btn.click(
                                add_to_prompt, 
                                inputs=[prompt, chip_btn], 
                                outputs=prompt
                            )
            
            # Generation controls
            with gr.Row():
                create_button = gr.Button("إنشاء الصورة", variant="primary")
                clear_button = gr.Button("مسح", variant="secondary")
                open_folder_button = gr.Button("عرض مجلد الصور", variant="secondary")
            
            # Output components
            with gr.Column():
                result = gr.Image(label="الصورة الناتجة", type="pil")
                generation_info = gr.Textbox(label="معلومات التوليد")
        
        # Advanced options accordion
        with gr.Accordion("الخيارات المتقدمة", open=False):
            with gr.Row():
                # Basic generation settings
                use_negative_prompt = gr.Checkbox(
                    label="استخدام الوصف السلبي", 
                    value=True
                )
                randomize_seed = gr.Checkbox(
                    label="بذرة عشوائية", 
                    value=True
                )
                num_images = gr.Radio(
                    choices=["1", "2", "3", "4"],
                    label="عدد الصور",
                    value="1"
                )
            
            # Negative prompt
            negative_prompt = gr.Textbox(
                label="الوصف السلبي",
                value=DEFAULT_NEGATIVE_PROMPT,
                lines=4
            )
            
            # Image dimensions
            with gr.Row():
                width = gr.Slider(
                    label="العرض", 
                    minimum=512, 
                    maximum=2048, 
                    step=64, 
                    value=1024
                )
                height = gr.Slider(
                    label="الارتفاع", 
                    minimum=512, 
                    maximum=2048, 
                    step=64, 
                    value=1024
                )
            
            # Generation parameters
            with gr.Row():
                guidance_scale = gr.Slider(
                    label="قوة الالتزام بالوصف",
                    minimum=1.0, 
                    maximum=20.0, 
                    step=0.5, 
                    value=7.5
                )
                seed = gr.Slider(
                    label="البذرة", 
                    minimum=0, 
                    maximum=MAX_SEED, 
                    step=1, 
                    value=0
                )
            
            # Artistic style selection
            style_preset = gr.Radio(
                choices=["واقعي", "كرتوني", "رسم زيتي", "رسم رقمي", "سينمائية"],
                label="النمط الفني",
                value="واقعي"
            )
        
        # Event bindings
        create_button.click(
            fn=generate,
            inputs=[
                prompt, 
                negative_prompt, 
                use_negative_prompt, 
                seed, 
                width, 
                height, 
                guidance_scale, 
                randomize_seed, 
                style_preset, 
                num_images
            ],
            outputs=[result, generation_info]
        )
        
        clear_button.click(
            lambda: [None, ""],
            outputs=[result, prompt]
        )
        
        open_folder_button.click(open_output_folder)
        
        return demo

# تهيئة النموذج
logger.info("بدء تهيئة النموذج...")
pipe = initialize_model()
logger.info("تم تهيئة النموذج بنجاح")

def find_available_port(start_port=7860, max_attempts=10):
    """
    Find an available port starting from start_port.
    
    Args:
        start_port (int): Port to start searching from
        max_attempts (int): Maximum number of ports to try
    
    Returns:
        int: Available port number
    """
    import socket
    
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    
    raise RuntimeError(f"Could not find an available port in range {start_port}-{start_port + max_attempts}")

# تشغيل التطبيق
if __name__ == "__main__":
    try:
        available_port = find_available_port()
        logger.info(f"Starting server on port {available_port}")
        demo = create_interface()
        demo.queue(max_size=20).launch(
            server_port=available_port,
            share=False,
            show_error=True,
            debug=True
        )
    except Exception as e:
        logger.error(f"Failed to launch application: {e}")
        raise
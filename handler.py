"""
RunPod Serverless Handler pour Video-R1-7B

Ce handler utilise Transformers directement pour une meilleure compatibilité.
"""

import runpod
import base64
import os
import tempfile
import torch

# Variables globales pour le modèle (chargé une seule fois)
model = None
processor = None


def load_model():
    """Charge le modèle Video-R1-7B avec Transformers"""
    global model, processor

    if model is not None:
        return  # Déjà chargé

    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    model_path = os.environ.get("MODEL_NAME", "Video-R1/Video-R1-7B")
    
    print(f"Loading model: {model_path}")

    # Charger le modèle avec Transformers (plus stable que vLLM pour ce modèle)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    print("Model loaded successfully!")


def handler(job):
    """
    Handler principal pour RunPod Serverless.
    
    Input attendu:
    {
        "video_frames": ["base64_frame1", "base64_frame2", ...],
        "question": "Votre question sur la vidéo",
        "problem_type": "free-form",  # optionnel
        "max_frames": 32  # optionnel
    }
    """
    from qwen_vl_utils import process_vision_info
    from PIL import Image
    from io import BytesIO
    import cv2
    import numpy as np

    # Charger le modèle si nécessaire
    load_model()

    job_input = job["input"]

    # Extraire les paramètres
    video_frames = job_input.get("video_frames", [])
    question = job_input.get("question", "Describe what happens in this video.")
    problem_type = job_input.get("problem_type", "free-form")
    max_frames = job_input.get("max_frames", 32)

    if not video_frames:
        return {"error": "No video frames provided"}

    # Template de prompt Video-R1
    QUESTION_TEMPLATE = (
        "{Question}\n"
        "Please think about this question as if you were a human pondering deeply. "
        "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc. "
        "It's encouraged to include self-reflection or verification in the reasoning process. "
        "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
    )

    TYPE_TEMPLATE = {
        "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
        "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
        "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
        "free-form": " Please provide your text answer within the <answer> </answer> tags.",
        "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
    }

    try:
        # Convertir les frames base64 en images PIL
        frames_pil = []
        for frame_b64 in video_frames[:max_frames]:
            img_data = base64.b64decode(frame_b64)
            img = Image.open(BytesIO(img_data)).convert("RGB")
            frames_pil.append(img)

        # Créer un fichier vidéo temporaire
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        # Sauvegarder comme vidéo
        if frames_pil:
            height, width = np.array(frames_pil[0]).shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(tmp_path, fourcc, 1, (width, height))

            for frame in frames_pil:
                frame_np = np.array(frame)
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            out.release()

        # Construire le message multimodal
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": tmp_path,
                        "max_pixels": 360 * 420,
                        "nframes": max_frames,
                    },
                    {
                        "type": "text",
                        "text": QUESTION_TEMPLATE.format(Question=question)
                        + TYPE_TEMPLATE.get(problem_type, TYPE_TEMPLATE["free-form"]),
                    },
                ],
            }
        ]

        # Préparer les inputs avec le processor
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Générer la réponse
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.1,
                top_p=0.001,
            )
        
        # Décoder la réponse
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        # Parser la réponse
        thinking = ""
        answer = ""

        if "<think>" in output_text and "</think>" in output_text:
            thinking = output_text.split("<think>")[1].split("</think>")[0].strip()

        if "<answer>" in output_text and "</answer>" in output_text:
            answer = output_text.split("<answer>")[1].split("</answer>")[0].strip()
        else:
            answer = output_text

        # Nettoyer le fichier temporaire
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

        return {
            "thinking": thinking,
            "answer": answer,
            "raw_output": output_text,
        }

    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


# Point d'entrée RunPod
runpod.serverless.start({"handler": handler})

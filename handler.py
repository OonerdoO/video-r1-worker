"""
RunPod Serverless Handler pour Video-R1-7B

BASÉ SUR LE CODE OFFICIEL: https://github.com/tulerfeng/Video-R1/blob/main/src/inference_example.py

Utilise:
- vLLM 0.7.2 (version officielle)
- transformers commit spécifique (336dc69d63d56f232a183a3e7f52790429b871ef)
- AutoProcessor.from_pretrained(model_path) comme dans le code officiel
"""

import runpod
import base64
import os
import tempfile

# Variables globales pour le modèle (chargé une seule fois)
llm = None
processor = None
tokenizer = None
sampling_params = None

# Configuration
MODEL_PATH = os.environ.get("MODEL_NAME", "Video-R1/Video-R1-7B")


def load_model():
    """Charge le modèle Video-R1-7B avec vLLM (code officiel)"""
    global llm, processor, tokenizer, sampling_params

    if llm is not None:
        return  # Déjà chargé

    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor, AutoTokenizer

    print(f"Loading model from: {MODEL_PATH}")

    # Initialiser vLLM (exactement comme le code officiel)
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=1,
        max_model_len=32768,  # Réduit pour GPU 80GB
        gpu_memory_utilization=0.85,
        limit_mm_per_prompt={"video": 1, "image": 1},
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        max_tokens=1024,
    )

    # Charger processor et tokenizer (exactement comme le code officiel)
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    print("Model loaded successfully!")


def handler(job):
    """
    Handler principal pour RunPod Serverless.
    Basé sur le code officiel inference_example.py
    """
    from qwen_vl_utils import process_vision_info
    from PIL import Image
    from io import BytesIO
    import cv2
    import numpy as np

    try:
        # Charger le modèle si nécessaire
        load_model()

        job_input = job["input"]

        # Extraire les paramètres
        video_frames = job_input.get("video_frames", [])
        question = job_input.get("question", "Describe what happens in this video.")
        problem_type = job_input.get("problem_type", "free-form")
        max_frames = job_input.get("max_frames", 32)  # Défaut: 32 frames, pas de limite
        # Résolution contrôlée par le client (defaut: 560x560 = 313600 pixels)
        max_pixels = job_input.get("max_pixels", 313600)

        print(f"Config: max_frames={max_frames}, max_pixels={max_pixels}")

        if not video_frames:
            return {"error": "No video frames provided"}

        # Template de prompt (exactement comme le code officiel)
        QUESTION_TEMPLATE = (
            "{Question}\n"
            "Please think about this question as if you were a human pondering deeply. "
            "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
            "It's encouraged to include self-reflection or verification in the reasoning process. "
            "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
        )

        TYPE_TEMPLATE = {
            "multiple choice": " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags.",
            "numerical": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags.",
            "OCR": " Please transcribe text from the image/video clearly and provide your text answer within the <answer> </answer> tags.",
            "free-form": " Please provide your text answer within the <answer> </answer> tags.",
            "regression": " Please provide the numerical value (e.g., 42 or 3.14) within the <answer> </answer> tags."
        }

        # Convertir les frames base64 en images PIL
        frames_pil = []
        for frame_b64 in video_frames[:max_frames]:
            img_data = base64.b64decode(frame_b64)
            img = Image.open(BytesIO(img_data)).convert("RGB")
            frames_pil.append(img)

        print(f"Processing {len(frames_pil)} frames...")

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

        # Construire le message multimodal (exactement comme le code officiel)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": tmp_path,
                        "max_pixels": max_pixels,  # Résolution contrôlée par le client
                        "nframes": len(frames_pil)  # Nombre réel de frames dans la vidéo temp
                    },
                    {
                        "type": "text",
                        "text": QUESTION_TEMPLATE.format(Question=question) + TYPE_TEMPLATE.get(problem_type, TYPE_TEMPLATE["free-form"])
                    },
                ],
            }
        ]

        # Convertir en prompt string (comme le code officiel)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Traiter la vidéo (comme le code officiel)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        # Préparer l'input vLLM (comme le code officiel)
        # Note: video_kwargs peut contenir des booléens, on ne garde que les listes
        mm_processor_kwargs = {}
        for key, val in video_kwargs.items():
            if isinstance(val, list) and len(val) > 0:
                mm_processor_kwargs[key] = val[0]
            elif not isinstance(val, bool):
                mm_processor_kwargs[key] = val
        
        llm_inputs = [{
            "prompt": prompt,
            "multi_modal_data": {"video": video_inputs[0]},
            "mm_processor_kwargs": mm_processor_kwargs,
        }]

        print("Running inference...")

        # Exécuter l'inférence
        outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
        output_text = outputs[0].outputs[0].text

        print(f"Generated response: {output_text[:200]}...")

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
        error_msg = str(e)
        error_tb = traceback.format_exc()
        print(f"ERROR: {error_msg}")
        print(f"TRACEBACK: {error_tb}")
        return {
            "error": error_msg,
            "traceback": error_tb
        }


# Point d'entrée RunPod
runpod.serverless.start({"handler": handler})

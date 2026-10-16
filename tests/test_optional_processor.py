from pathlib import Path
import importlib.util
import pytest


@pytest.mark.skipif(not Path("models/smolvlm2-256m/config.json").exists() or not importlib.util.find_spec("transformers"),
                    reason="Optional offline model processor files are not installed")
def test_six_frames_stay_six_images_under_native_budget():
    from PIL import Image
    from transformers import AutoProcessor
    from risksifter.annotators import bound_processor_frames
    processor = AutoProcessor.from_pretrained("models/smolvlm2-256m", local_files_only=True)
    edge = bound_processor_frames(processor)
    images = [Image.new("RGB", (960, 600), (i * 40, 50, 80)) for i in range(6)]
    messages = [{"role": "user", "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": "Describe the road."}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=images, return_tensors="pt")
    assert list(inputs["pixel_values"].shape) == [1, 6, 3, edge, edge]
    assert inputs["input_ids"].shape[1] < 800

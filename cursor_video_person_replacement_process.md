# Video person replacement process
_Exported on 12/12/2025 at 13:19:33 CET from Cursor (2.1.39)_

---

**User**

have a look in the repo, and tell me the script that can use segmentation mask as input condition and give me the output. my usercase is, I have a sports video of tennis, I want to change the person, by segmenting the person and then change it via promp through the network. have a detailed look, take your time, and tell me clearly the process how can I do it end-to-end.

---

**Cursor**

I'll explore this repository to understand how you can use segmentation masks as input conditions for your tennis video use case. Let me start by examining the structure and key files.

Now I have a complete understanding of the repository. Let me provide you with a comprehensive end-to-end guide for your use case.

---

## Complete Guide: Using Segmentation Mask for Tennis Video Person Replacement

### Overview

You want to:
1. Take a tennis video with a person
2. Segment the person
3. Replace/change the person using a text prompt

**Cosmos-Transfer2.5** supports this workflow using **segmentation control** with **on-the-fly or pre-computed masks**.

---

### The Script to Use

The main inference script is:

```bash
python examples/inference.py -i <your_spec.json> -o outputs/
```

---

### End-to-End Process

#### **Step 1: Environment Setup**

```bash
# Clone and setup
git clone git@github.com:nvidia-cosmos/cosmos-transfer2.5.git
cd cosmos-transfer2.5
git lfs pull

# Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync --extra=cu128
source .venv/bin/activate

# Login to HuggingFace and accept license
hf auth login
```

#### **Step 2: Prepare Your Input Files**

You need:
1. **Input video**: Your tennis video (e.g., `tennis_input.mp4`)
2. **Prompt file**: Text describing desired output (e.g., `tennis_prompt.txt`)
3. **Spec file**: JSON configuration (e.g., `tennis_seg_spec.json`)

#### **Step 3: Create the Specification File**

You have **two options** for segmentation:

---

##### **Option A: On-the-Fly Segmentation (Recommended for your use case)**

The system can automatically segment objects using a text prompt via **GroundingDINO + SAM2**.

Create `tennis_seg_spec.json`:

```json
{
    "name": "tennis_person_replacement",
    "prompt": "A professional badminton player in a modern indoor sports arena, wearing a red athletic outfit, performing a powerful serve. The lighting is dramatic with stadium lights. The court has professional markings.",
    "video_path": "tennis_input.mp4",
    "seg": {
        "control_prompt": "person tennis player",
        "control_weight": 1.0
    },
    "guidance": 5,
    "seed": 2025
}
```

**Key parameters:**
- `control_prompt`: Tells SAM2 what to segment (e.g., "person", "tennis player", "human")
- If `control_path` is omitted, segmentation is computed **on-the-fly** from the video

---

##### **Option B: Pre-computed Segmentation Mask**

If you have your own segmentation mask video (colored segmentation map), use:

```json
{
    "name": "tennis_person_replacement",
    "prompt": "A professional basketball player in a gymnasium, wearing blue jersey number 23, dribbling the ball. Modern sports facility with wooden floor.",
    "video_path": "tennis_input.mp4",
    "seg": {
        "control_path": "path/to/your/segmentation_mask.mp4",
        "control_weight": 1.0
    },
    "guidance": 5,
    "seed": 2025
}
```

---

##### **Option C: Using mask_prompt for Selective Region Control**

If you want the control to only apply to specific regions (e.g., only change the person, not the background):

```json
{
    "name": "tennis_person_selective",
    "prompt": "A robot playing tennis in a futuristic arena, metallic body with glowing joints, precise mechanical movements.",
    "video_path": "tennis_input.mp4",
    "seg": {
        "control_weight": 1.0,
        "mask_prompt": "person tennis player"
    },
    "guidance": 5,
    "seed": 2025
}
```

The `mask_prompt` creates a binary mask where the control is applied only to the segmented regions (white pixels).

---

#### **Step 4: Run Inference**

```bash
# Single GPU (requires ~65GB VRAM)
python examples/inference.py -i tennis_seg_spec.json -o outputs/tennis

# Multi-GPU for faster inference
torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py -i tennis_seg_spec.json -o outputs/tennis
```

---

#### **Step 5: Standalone Segmentation (Optional)**

If you want to generate segmentation masks separately using SAM2:

```python
from cosmos_transfer2._src.transfer2.auxiliary.sam2.sam2_model import VideoSegmentationModel

# Initialize the segmentation model
model = VideoSegmentationModel()

# Generate segmentation from text prompt
model(
    input_video="tennis_input.mp4",
    prompt="person tennis player",  # What to segment
    output_video="tennis_segmentation.mp4",
    binarize_video=False  # False for colored mask, True for binary
)
```

---

### Understanding the Segmentation Workflow

From the code in `cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_model.py`:

```55:73:cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_model.py
class VideoSegmentationModel:
    def __init__(self, **kwargs):
        """Initialize the model and load all required components."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize SAM2 predictor
        self.sam2_predictor = SAM2VideoPredictor.from_pretrained(SAM2_MODEL_CHECKPOINT).to(self.device)

        # Initialize GroundingDINO for text-based detection
        self.grounding_model_name = kwargs.get("grounding_model", GROUNDING_DINO_MODEL_CHECKPOINT)
        self.processor = AutoProcessor.from_pretrained(self.grounding_model_name)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.grounding_model_name).to(
            self.device
        )
```

The system uses:
1. **GroundingDINO** (base model) for text-to-bounding-box detection
2. **SAM2** (facebook/sam2-hiera-large) for video segmentation propagation

---

### Segmentation Format

The segmentation mask should be a **colored video** where each object class has a distinct RGB color. From `cosmos_transfer2/_src/transfer2/datasets/augmentors/seg.py`:

```22:48:cosmos_transfer2/_src/transfer2/datasets/augmentors/seg.py
PREDEFINED_COLORS_SEGMENTATION = np.array(
    [
        [255, 0, 0],  # Red
        [0, 255, 0],  # Green
        [0, 0, 255],  # Blue
        [255, 255, 0],  # Yellow
        [0, 255, 255],  # Cyan
        [255, 0, 255],  # Magenta
        # ... more colors
    ]
)
```

---

### Complete Example Spec for Your Tennis Use Case

```json
{
    "name": "tennis_to_robot",
    "prompt": "A humanoid robot playing tennis in a high-tech sports arena. The robot has a sleek silver metallic body with blue LED accents. Professional tennis court with modern stadium architecture. Cinematic lighting with dramatic shadows. 4K quality video.",
    "negative_prompt": "blurry, low quality, distorted, cartoon, anime, unrealistic physics",
    "video_path": "/path/to/your/tennis_video.mp4",
    "seg": {
        "control_prompt": "person human tennis player",
        "control_weight": 0.8
    },
    "depth": {
        "control_weight": 0.5
    },
    "guidance": 5,
    "seed": 2025,
    "resolution": "720",
    "num_video_frames_per_chunk": 93,
    "show_control_condition": true,
    "show_input": true
}
```

This uses:
- **Segmentation** (on-the-fly) to identify the person
- **Depth** (on-the-fly) to preserve spatial structure
- Multi-control for best results

---

### Summary

| Step | Command/Action |
|------|----------------|
| 1. Setup | `uv sync --extra=cu128 && source .venv/bin/activate` |
| 2. Create spec | JSON with `seg.control_prompt: "person"` for auto-segmentation |
| 3. Run inference | `python examples/inference.py -i spec.json -o outputs/` |
| 4. Output | Generated video in `outputs/` directory |

The key insight is that you **don't need to pre-compute segmentation masks** - by using `control_prompt` under the `seg` key, the system automatically segments using GroundingDINO + SAM2 and applies the transformation only to those regions.

```shellscript
python examples/inference.py -i <your_spec.json> -o outputs/
```

```shellscript
# Clone and setup
git clone git@github.com:nvidia-cosmos/cosmos-transfer2.5.git
cd cosmos-transfer2.5
git lfs pull

# Install dependencies
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync --extra=cu128
source .venv/bin/activate

# Login to HuggingFace and accept license
hf auth login
```

```json
{
    "name": "tennis_person_replacement",
    "prompt": "A professional badminton player in a modern indoor sports arena, wearing a red athletic outfit, performing a powerful serve. The lighting is dramatic with stadium lights. The court has professional markings.",
    "video_path": "tennis_input.mp4",
    "seg": {
        "control_prompt": "person tennis player",
        "control_weight": 1.0
    },
    "guidance": 5,
    "seed": 2025
}
```

```json
{
    "name": "tennis_person_replacement",
    "prompt": "A professional basketball player in a gymnasium, wearing blue jersey number 23, dribbling the ball. Modern sports facility with wooden floor.",
    "video_path": "tennis_input.mp4",
    "seg": {
        "control_path": "path/to/your/segmentation_mask.mp4",
        "control_weight": 1.0
    },
    "guidance": 5,
    "seed": 2025
}
```

```json
{
    "name": "tennis_person_selective",
    "prompt": "A robot playing tennis in a futuristic arena, metallic body with glowing joints, precise mechanical movements.",
    "video_path": "tennis_input.mp4",
    "seg": {
        "control_weight": 1.0,
        "mask_prompt": "person tennis player"
    },
    "guidance": 5,
    "seed": 2025
}
```

```shellscript
# Single GPU (requires ~65GB VRAM)
python examples/inference.py -i tennis_seg_spec.json -o outputs/tennis

# Multi-GPU for faster inference
torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py -i tennis_seg_spec.json -o outputs/tennis
```

```python
from cosmos_transfer2._src.transfer2.auxiliary.sam2.sam2_model import VideoSegmentationModel

# Initialize the segmentation model
model = VideoSegmentationModel()

# Generate segmentation from text prompt
model(
    input_video="tennis_input.mp4",
    prompt="person tennis player",  # What to segment
    output_video="tennis_segmentation.mp4",
    binarize_video=False  # False for colored mask, True for binary
)
```

```python
class VideoSegmentationModel:
    def __init__(self, **kwargs):
        """Initialize the model and load all required components."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize SAM2 predictor
        self.sam2_predictor = SAM2VideoPredictor.from_pretrained(SAM2_MODEL_CHECKPOINT).to(self.device)

        # Initialize GroundingDINO for text-based detection
        self.grounding_model_name = kwargs.get("grounding_model", GROUNDING_DINO_MODEL_CHECKPOINT)
        self.processor = AutoProcessor.from_pretrained(self.grounding_model_name)
        self.grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.grounding_model_name).to(
            self.device
        )
```

```python
PREDEFINED_COLORS_SEGMENTATION = np.array(
    [
        [255, 0, 0],  # Red
        [0, 255, 0],  # Green
        [0, 0, 255],  # Blue
        [255, 255, 0],  # Yellow
        [0, 255, 255],  # Cyan
        [255, 0, 255],  # Magenta
        # ... more colors
    ]
)
```

```json
{
    "name": "tennis_to_robot",
    "prompt": "A humanoid robot playing tennis in a high-tech sports arena. The robot has a sleek silver metallic body with blue LED accents. Professional tennis court with modern stadium architecture. Cinematic lighting with dramatic shadows. 4K quality video.",
    "negative_prompt": "blurry, low quality, distorted, cartoon, anime, unrealistic physics",
    "video_path": "/path/to/your/tennis_video.mp4",
    "seg": {
        "control_prompt": "person human tennis player",
        "control_weight": 0.8
    },
    "depth": {
        "control_weight": 0.5
    },
    "guidance": 5,
    "seed": 2025,
    "resolution": "720",
    "num_video_frames_per_chunk": 93,
    "show_control_condition": true,
    "show_input": true
}
```


import os

import torch

from config import ACTIONS, DIM, MODELS_DIR, NUM_FRAMES, NUM_JOINTS
from models import MotionTransformerDiffusion
from services import animate_skeleton_3d


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = os.path.join(MODELS_DIR, "stickman_diffusion.pth")

    model = MotionTransformerDiffusion().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    torch.manual_seed(7)

    with torch.no_grad():
        for action_name, action_idx in ACTIONS.items():
            noise = torch.randn((1, NUM_FRAMES, NUM_JOINTS, DIM), device=device)
            cond = torch.tensor([action_idx], dtype=torch.long, device=device)
            t = torch.zeros((1,), device=device)

            generated_motion = model(noise, t, cond)[0].cpu().numpy()
            output_filename = f"{action_name}_generated.gif"
            animate_skeleton_3d(
                generated_motion,
                output_filename=output_filename,
                fps=4,
                action_name=action_name,
            )
            print(f"Zapisano animacje: output/{output_filename}")


if __name__ == "__main__":
    main()

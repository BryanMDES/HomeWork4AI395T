from pathlib import Path
import fire
from matplotlib import pyplot as plt
import json

from .generate_qa import (draw_detections,extract_frame_info,extract_kart_objects,extract_track_info,)


def generate_caption(info_path: str, view_index: int, img_width: int = 150, img_height: int = 100) -> list:
    """
    Generate caption for a specific view.
    """
    # 1. Ego car
    # {kart_name} is the ego car.

    # 2. Counting
    # There are {num_karts} karts in the scenario.

    # 3. Track name
    # The track is {track_name}.

    # 4. Relative position
    # {kart_name} is {position} of the ego car.

    kart_objects = extract_kart_objects(info_path, view_index, img_width, img_height)
    track_name = extract_track_info(info_path)

    if not kart_objects:
        return []

    ego_kart = next(k for k in kart_objects if k["is_center_kart"])
    ego_name = ego_kart["kart_name"]
    ego_x, ego_y = ego_kart["center"]

    captions = []
    # 1. Ego car
    captions.append(f"{ego_name} is the ego car.")
    captions.append(f"The ego car is {ego_name}.")

    # 2. Counting
    num_karts = len(kart_objects)
    kart_word = "kart" if num_karts == 1 else "karts"
    if num_karts == 1:
      captions.append("There is 1 kart in the scenario.")
    else:
      captions.append(f"There are {num_karts} karts in the scenario.")
  
    captions.append(f"The scene contains {num_karts} {kart_word}.")

    # 3. Track name
    captions.append(f"The track is {track_name}.")
    captions.append(f"This is the {track_name} track.")

    for kart in kart_objects:
      if kart["is_center_kart"]:
        continue
      kart_name = kart["kart_name"]
      x,y = kart["center"]

      horizontal = "left" if x < ego_x else "right"
      vertical = "front" if y < ego_y else "back"

      captions.append(f"{kart_name} is {vertical} and {horizontal} of the ego car.")

    return captions

def check_caption(info_file: str, view_index: int):
    captions = generate_caption(info_file, view_index)

    print("\nCaption:")
    print("-" * 50)
    for i, caption in enumerate(captions):
        print(f"{i + 1}. {caption}")
        print("-" * 50)

    info_path = Path(info_file)
    base_name = info_path.stem.replace("_info", "")
    image_file = list(info_path.parent.glob(f"{base_name}_{view_index:02d}_im.jpg"))[0]

    annotated_image = draw_detections(str(image_file), info_file)

    plt.figure(figsize=(12, 8))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.title(f"Frame {extract_frame_info(str(image_file))[0]}, View {view_index}")
    plt.show()


"""
Usage Example: Visualize QA pairs for a specific file and view:
   python generate_captions.py check --info_file ../data/valid/00000_info.json --view_index 0

You probably need to add additional commands to Fire below.
"""
def generate_captions_for_split(data_dir: str, output_file: str):
    results = []

    data_path = Path(data_dir)
    info_files = sorted(data_path.glob("*_info.json"))

    for info_file in info_files:
        base_name = info_file.stem.replace("_info", "")

        image_files = sorted(info_file.parent.glob(f"{base_name}_*_im.jpg"))

        for image_file in image_files:
            _, view_index = extract_frame_info(str(image_file))
            captions = generate_caption(str(info_file), view_index)

            for caption in captions:
              results.append({
                "image_file": str(image_file),
                "caption": caption,})

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} caption pairs to {output_file}")


def main():
    fire.Fire({
        "check": check_caption,
        "generate": generate_captions_for_split,
    })


if __name__ == "__main__":
    main()

from pathlib import Path
import shutil

frames_folder = "..."   #insert frame folder path
output_folder = Path(frames_folder).parent / f"{Path(frames_folder).name}_reverse"

output_folder.mkdir(exist_ok=True)

# Leggi i file
frame_files = sorted(Path(frames_folder).glob('FrameBuffer_*.png'))
depth_files = sorted(Path(frames_folder).glob('Depth_*.png'))

print(f"Frame: {len(frame_files)}, Depth: {len(depth_files)}")

# Inverti e copia Frame
for frame_file in frame_files[::-1]:
    # Estrai il numero
    num = frame_file.name.replace('FrameBuffer_', '').replace('.png', '')
    new_idx = len(frame_files) - 1 - int(num)
    new_name = f"FrameBuffer_{new_idx}.png"
    shutil.copy2(frame_file, output_folder / new_name)
    print(f"✓ {frame_file.name} → {new_name}")

# Inverti e copia Depth
for depth_file in depth_files[::-1]:
    # Estrai il numero
    num = depth_file.name.replace('Depth_', '').replace('.png', '')
    new_idx = len(depth_files) - 1 - int(num)
    new_name = f"Depth_{new_idx}.png"
    shutil.copy2(depth_file, output_folder / new_name)
    #print(f"✓ {depth_file.name} → {new_name}")

print(f"\n✅ Fatto! {len(frame_files)} frame + {len(depth_files)} depth salvati in {output_folder}")
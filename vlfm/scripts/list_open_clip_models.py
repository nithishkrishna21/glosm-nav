
import open_clip

print("Available pretrained tags for ViT-H-14:")
try:
    available = open_clip.list_pretrained(model="ViT-H-14")
    for tag in available:
        print(f" - {tag}")
except Exception as e:
    print(f"Error listing models: {e}")
    # Fallback to printing all models if specific list fails
    print("\nAll Models:")
    print(open_clip.list_models())

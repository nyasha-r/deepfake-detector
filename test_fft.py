import cv2
from utils.features import extract_features

def main():
    # Load image
    img = cv2.imread("test.jpg")

    if img is None:
        print("❌ Image not found. Make sure test.jpg is in the project folder.")
        return

    # Extract features
    features = extract_features(img)

    # Print results nicely
    print("\n✅ Extracted Features:")
    print(f"Mean Energy       : {features[0]:.4f}")
    print(f"Std Deviation     : {features[1]:.4f}")
    print(f"Low Freq Energy   : {features[2]:.4f}")
    print(f"High Freq Energy  : {features[3]:.4f}")
    print(f"Energy Ratio      : {features[4]:.4f}")

if __name__ == "__main__":
    main()
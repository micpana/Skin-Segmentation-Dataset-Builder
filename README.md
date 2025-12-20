# Skin Segmentation Dataset Builder

An automated tool for converting **skin classification datasets** into **background-free skin segmentation datasets**, with a strong focus on **facial skin segmentation**, while still supporting **non-face skin images**.

This project is designed for **rapid prototyping**, **research**, and **production-ready dataset bootstrapping**.

---

## 🚀 What Problem Does This Solve?

Most publicly available skin datasets are **classification datasets**, structured like:

```
dataset_original/
├── train/
│   ├── Normal Skin/
│   ├── Acne/
│   └── ...
├── valid/
└── test/
```

These datasets are excellent for classification, but **not usable for segmentation tasks** without manual annotation.

Segmentation models require:
- Pixel-level masks
- A different dataset structure
- Careful handling of background pixels

### The Core Challenge

In skin segmentation:
- Background pixels dominate images
- Background becomes an overpowering class
- This introduces noise, imbalance, and poor generalization
- Even cropped face images still contain:
  - Hair
  - Eyes
  - Nostrils
  - Clothing
  - Background artifacts

### This Tool Solves That By:

✅ Automatically detecting **skin regions only**  
✅ Producing **standalone skin images**  
✅ Producing **standalone skin masks**  
✅ **Removing background entirely** (no background class)  
✅ Preserving **train / valid / test splits**  
✅ Working with:
- Face images (selfies)
- Partial skin images (arms, cheeks, neck, forehead)
- Skin-only datasets

---

## 🧠 Key Idea

Instead of labeling background pixels as a class, this tool:

> **Removes all non-skin pixels altogether**

This results in:
- Cleaner segmentation datasets
- No background domination
- Better class balance
- Faster convergence during training

---

## 🧩 How It Works (High Level)

For each image:

1. Attempt **face detection**
2. If a face is detected:
   - Prefer face crop (less noise)
3. If no face is detected:
   - Process the full image
4. Detect **skin pixels only**
5. Generate:
   - Skin-only image
   - Binary skin mask
6. Save outputs in segmentation dataset format

Face detection is **optional and non-blocking**.

---

## 🛠 Tools & Technologies Used

- **Python**
- **OpenCV** – image processing
- **MediaPipe** – optional face detection
- **HSV + YCrCb color space filtering** – skin detection
- **Morphological operations** – mask cleanup

No pretrained segmentation model is required.

---

## 📂 Input Dataset Format (Required)

The tool expects a **classification dataset** structured as follows:

```
dataset_original/
├── train/
│   ├── Normal Skin/
│   ├── Acne/
│   └── ...
├── valid/
│   ├── Normal Skin/
│   ├── Acne/
│   └── ...
└── test/
    ├── Normal Skin/
    ├── Acne/
    └── ...
```

---

## 📦 Output Dataset Format (Generated)

```
dataset/
├── images/
│   ├── train/
│   ├── valid/
│   └── test/
├── masks/
│   ├── train/
│   ├── valid/
│   └── test/
└── classes.txt
```

---

## 🔧 Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python build_dataset.py
```

---

## 📜 License

MIT License

---

## 📬 Contact & Support

**Michael Panashe Mudimbu**  
📧 Email: **michaelmudimbu@gmail.com**

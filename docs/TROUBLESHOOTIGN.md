# Troubleshooting Guide

Solutions for common issues and error messages.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Data & Preprocessing Issues](#data--preprocessing-issues)
- [Segmentation Issues](#segmentation-issues)
- [Training Issues](#training-issues)
- [Prediction Issues](#prediction-issues)
- [Performance Issues](#performance-issues)

---

## Installation Issues

### Issue: "ModuleNotFoundError: No module named 'tensorflow'"

**Error Message:**
```code
ModuleNotFoundError: No module named 'tensorflow'
```

**Causes:**
- Dependencies not installed
- Wrong Python version
- Pip installation failed

**Solutions:**

1. **Reinstall all dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Install individually:**
    ```bash
    pip install tensorflow>=2.10
    pip install opencv-python>=4.5
    pip install numpy>=1.20
    pip install matplotlib>=3.4
    pip install jupyter
    ```

3. **Check Python version:**
    ```bash
    python --version  # Should be 3.7 or higher
    ```

4. **Use virtual environment (recommended):**
    ```bash 
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

5. **For Google Colab (no installation needed):**
    ```bash
    # Run in first cell
    !pip install -q tensorflow opencv-python
    ```


### Issue: "ImportError: cannot import name 'cv2'"

**Causes:**
- OpenCV not installed or wrong version
- Using old cv2 API

**Solution:**
```bash
pip uninstall opencv-python
pip install opencv-python==4.7.0.72
```


### Issue: Pip takes too long / Won't install

**Solution - Use Conda instead:**
```bash
conda create -n sinhala python=3.9
conda activate sinhala
conda install tensorflow opencv numpy matplotlib jupyter
```


# Data & Preprocessing Issues

### Issue: "File not found" or "Cannot read image"

**Error Message:**
```code
ValueError: Image not found
```

**Causes:**
- Wrong file path
- File doesn't exist
- Encoding issues

**Solutions:**

1. Check file path:
    ```python
    import os

    # Verify path exists
    path = 'dataset/ක/image.png'
    print(os.path.exists(path))  # Should print: True
    ```

2. List available files:
    ```python
    import os

    dataset_path = 'dataset'
    for char in os.listdir(dataset_path):
        char_path = os.path.join(dataset_path, char)
        files = os.listdir(char_path)
        print(f"{char}: {len(files)} files")
    ```

3. Use absolute path:
    ```python
    import os

    img_path = os.path.abspath('dataset/ක/image.png')
    img = cv2.imread(img_path)
    ```

4. For Colab - mount drive first:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')

    img_path = '/content/drive/MyDrive/dataset/ක/image.png'
    img = cv2.imread(img_path)
    ```

    
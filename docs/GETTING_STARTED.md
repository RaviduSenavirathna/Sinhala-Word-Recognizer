# Getting Started with Sinhala Word Recognizer

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.7 or higher
- pip (Python package manager)
- Git
- A GPU (optional, but recommended for faster training)

## Quick Start

### Step 1: Clone the Repository

```bash
git clone https://github.com/RaviduSenavirathna/Sinhala-Word-Recognizer.git
cd Sinhala-Word-Recognizer
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Your Dataset
Your dataset should be organized in the following structure:

```code
dataset/
├── ක/
│   ├── image1.png
│   ├── image2.png
│   └── ...
├── ග/
│   ├── image1.png
│   └── ...
├── ත/
│   └── ...
└── ... (other Sinhala characters)
```

Each folder represents a Sinhala character class, and contains handwritten samples of that character.
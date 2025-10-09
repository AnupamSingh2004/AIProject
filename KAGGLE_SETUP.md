# 🔑 Kaggle API Setup Guide

## Why You Need This

To download the Fashion Product Images dataset (44k products), you need Kaggle API credentials.

## Step-by-Step Setup

### Option 1: Using Kaggle API (Recommended)

#### 1. Get Your Kaggle API Token

1. Go to [https://www.kaggle.com](https://www.kaggle.com) and sign in (or create an account)
2. Click on your profile picture (top right)
3. Select **"Settings"** from the dropdown
4. Scroll down to the **"API"** section
5. Click **"Create New Token"**
6. This will download a file called `kaggle.json`

#### 2. Install the API Token

**On Linux/Mac:**
```bash
# Create the kaggle directory
mkdir -p ~/.kaggle

# Move the downloaded kaggle.json file
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set proper permissions
chmod 600 ~/.kaggle/kaggle.json
```

**On Windows:**
```cmd
# Create the kaggle directory
mkdir %USERPROFILE%\.kaggle

# Move the downloaded kaggle.json file
move %USERPROFILE%\Downloads\kaggle.json %USERPROFILE%\.kaggle\
```

#### 3. Verify Setup

```bash
conda activate AI
kaggle --version
```

You should see something like: `Kaggle API 1.x.x`

#### 4. Download the Dataset

```bash
cd /home/anupam/code/AIProject
conda activate AI
python scripts/download_dataset.py
```

This will:
- Download ~25GB of data
- Extract to `data/raw/`
- Create `data/raw/styles.csv` and `data/raw/images/`
- Take 10-30 minutes depending on internet speed

### Option 2: Manual Download (Alternative)

If you have issues with the API:

1. Go to the dataset page: [https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
2. Click **"Download"** button
3. Extract the downloaded zip file
4. Copy contents to your project:
   ```bash
   # Copy the styles.csv file
   cp /path/to/downloaded/styles.csv /home/anupam/code/AIProject/data/raw/
   
   # Copy the images folder
   cp -r /path/to/downloaded/images /home/anupam/code/AIProject/data/raw/
   ```

## What the Dataset Contains

- **styles.csv**: Metadata for 44,000+ fashion products
  - Columns: id, gender, masterCategory, subCategory, articleType, baseColour, season, year, usage, productDisplayName
- **images/**: 44,000+ product images (JPG format, various sizes)
  - Example: `1163.jpg`, `1164.jpg`, etc.

## After Download

Once downloaded, verify it worked:

```bash
conda activate AI
python verify_setup.py
```

You should now see ✅ for the dataset check.

## Next Steps

After the dataset is downloaded:

```bash
# Preprocess the data
python scripts/preprocess_data.py

# Train the models
python train.py

# Or use the interactive menu
./quickstart.sh
```

## Troubleshooting

### "401 Unauthorized" Error
- Your kaggle.json file is incorrect or missing
- Re-download the token from Kaggle settings
- Check permissions: `chmod 600 ~/.kaggle/kaggle.json`

### "403 Forbidden" Error
- You need to accept the dataset's terms
- Go to the dataset page and click "Download" once (accept terms)
- Then run the download script again

### "Dataset not found" Error
- Check your internet connection
- Verify you're using the correct dataset name
- Try the manual download option

### Slow Download
- The dataset is 25GB, it will take time
- Use a stable internet connection
- Consider downloading overnight

## Dataset Citation

```
Fashion Product Images Dataset
Param Aggarwal
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset
```

---

Need help? Check the main README.md or TRAINING.md for more information.

#!/bin/bash
# Download ISIC 2018 and Chest X-ray datasets

echo "Starting data download..."

mkdir -p data/raw

# Use python script to download ISIC 2018
python -c "from src.data.download import download_isic2018, download_chest_xray; download_isic2018(); download_chest_xray()"

echo "Data download process finished!"

from icrawler.builtin import BingImageCrawler
import os

# Create folders
os.makedirs("data/real", exist_ok=True)
os.makedirs("data/fake", exist_ok=True)

# 🔵 Download REAL images
print("Downloading REAL images...")
real_crawler = BingImageCrawler(storage={"root_dir": "data/real"})
real_crawler.crawl(keyword="real human face portrait", max_num=200)

# 🔴 Download FAKE images
print("Downloading FAKE images...")
fake_crawler = BingImageCrawler(storage={"root_dir": "data/fake"})
fake_crawler.crawl(keyword="AI generated human face", max_num=200)

print("✅ Download complete!")
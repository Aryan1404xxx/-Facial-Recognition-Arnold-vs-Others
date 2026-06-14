from icrawler.builtin import GoogleImageCrawler

print("Downloading Arnold images...")
crawler = GoogleImageCrawler(storage={"root_dir": "data/arnold"})
crawler.crawl(keyword="Arnold Schwarzenegger face", max_num=80)

print("Downloading other people images...")
crawler = GoogleImageCrawler(storage={"root_dir": "data/others"})
crawler.crawl(keyword="celebrity face portrait", max_num=80)

print("Done!")

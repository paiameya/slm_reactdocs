import requests
from bs4 import BeautifulSoup
import re
import os
import sys

def get_sitemap_urls(sitemap_url):
    response = requests.get(sitemap_url)
    soup = BeautifulSoup(response.content, 'xml')
    urls = [element.text for element in soup.find_all('loc')]
    return urls

def scrape_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.title.string.strip() if soup.title else 'No_title'
    return title

def sanitize_filename(title):
    # Replace spaces with underscores and remove invalid characters
    filename = re.sub(r'[\/:*?"<>|]', '', title)  # Remove invalid characters
    return filename.replace(' ', '_') + '.txt'

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <sitemap_url>")
        sys.exit(1)

    sitemap_url = sys.argv[1]
    urls = get_sitemap_urls(sitemap_url)
    
    for url in urls:
        title = scrape_page(url)
        filename = sanitize_filename(title)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f'URL: {url}\nTitle: {title}\n')
            
        print(f'Scraped data for {title} saved to {filename}')

if __name__ == "__main__":
    main()


import asyncio
import aiohttp
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urldefrag
import time
import os
from aiohttp import ClientTimeout
from aiohttp_retry import RetryClient, ExponentialRetry

# Semaphore to limit concurrent connections
MAX_CONCURRENT_REQUESTS = 50

async def scrape_single_page(session, url, semaphore):
    """Scrape a single page asynchronously"""
    async with semaphore:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            }
            
            # Remove URL fragments before making the request
            clean_url = urldefrag(url).url
            
            async with session.get(clean_url, headers=headers) as response:
                if response.status != 200:
                    return {
                        'url': clean_url,
                        'title': None,
                        'text': None,
                        'error': f"Status code: {response.status}"
                    }
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                title = soup.title.text.strip() if soup.title else "No Title"
                
                for script in soup(["script", "style", "code"]):
                    script.decompose()
                
                text = soup.get_text(separator=' ', strip=True)
                
                # Get more links while we're parsing
                links = await get_all_links(soup, clean_url)
                
                return {
                    'url': clean_url,
                    'title': title,
                    'text': text,
                    'error': None,
                    'links': links
                }
                
        except Exception as e:
            return {
                'url': url,
                'title': None,
                'text': None,
                'error': str(e),
                'links': []
            }

async def get_all_links(soup, base_url):
    """Extract links from a BeautifulSoup object"""
    parsed_url = urlparse(base_url)
    base_domain = parsed_url.netloc
    
    links = set()
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(base_url, href)
        
        # Remove URL fragments
        clean_url = urldefrag(full_url).url
        
        if urlparse(clean_url).netloc == base_domain:
            links.add(clean_url)
    
    return list(links)

async def save_batch(batch_data, output_file, batch_num):
    """Save a batch of scraped data"""
    if not batch_data:
        return
        
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    # For the first batch, use 'w' mode, for subsequent batches use 'a' mode
    mode = 'w' if batch_num == 1 else 'a'
    
    with open(output_file, mode, encoding="utf-8") as file:
        all_text = ""
        for page in batch_data:
            all_text += f"Title: {page['title']}\n\n"
            all_text += f"Content: {page['text']}\n\n"
            all_text += "-" * 50 + "\n\n"
        
        file.write(all_text)
    
    print(f"Saved batch {batch_num} with {len(batch_data)} pages")

async def scrape_website(start_url, max_pages=100, batch_size=50, output_file="scraped_content.txt"):
    """Scrape website with concurrent requests"""
    start_time = time.time()
    
    # Track pages and urls
    visited = set()
    to_visit = [urldefrag(start_url).url]  # Start with clean URL
    results = []
    batch_num = 1
    
    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Configure timeout and retry
    timeout = ClientTimeout(total=30)
    retry_options = ExponentialRetry(attempts=3)
    
    print(f"Starting to scrape {start_url}...")
    print(f"Target: {max_pages} pages")
    
    # Track error counts but don't print every error
    error_count = 0
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        retry_client = RetryClient(session, retry_options=retry_options)
        
        while to_visit and len(results) < max_pages:
            # Process in batches for better throughput
            batch_urls = []
            while to_visit and len(batch_urls) < MAX_CONCURRENT_REQUESTS and len(results) + len(batch_urls) < max_pages:
                url = to_visit.pop(0)
                clean_url = urldefrag(url).url  # Remove URL fragments
                if clean_url not in visited:
                    visited.add(clean_url)
                    batch_urls.append(clean_url)
            
            if not batch_urls:
                break
            
            # Process the batch concurrently
            tasks = [scrape_single_page(retry_client, url, semaphore) for url in batch_urls]
            batch_results = await asyncio.gather(*tasks)
            
            # Process results and collect new links
            batch_success = []
            for page_data in batch_results:
                if page_data['error'] is None:
                    batch_success.append(page_data)
                    
                    # Add new links to visit
                    for link in page_data.get('links', []):
                        clean_link = urldefrag(link).url
                        if clean_link not in visited and clean_link not in to_visit:
                            to_visit.append(clean_link)
                else:
                    error_count += 1
            
            # Add successful results
            results.extend(batch_success)
            
            # Save in batches to avoid memory issues with large scrapes
            if len(batch_success) > 0:
                await save_batch(batch_success, output_file, batch_num)
                batch_num += 1
            
            print(f"Progress: {len(results)}/{max_pages} pages scraped")
    
    # Calculate and display performance metrics
    end_time = time.time()
    total_time = end_time - start_time
    pages_per_second = len(results) / total_time if total_time > 0 else 0
    
    print(f"Scraping completed in {total_time:.2f} seconds.")
    print(f"Scraped {len(results)} pages ({pages_per_second:.2f} pages/second).")
    print(f"Encountered {error_count} errors during scraping.")
    print(f"All content saved to {output_file}")
    
    return results

if __name__ == "__main__":
    website_url = "https://alnafi.com/"
    output_file = "scraped_content.txt"
    
    # Run the scraper
    asyncio.run(scrape_website(website_url, max_pages=500, output_file=output_file))
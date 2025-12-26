#!/usr/bin/env python3
"""
ArXiv Data Fetcher Script

This script automatically fetches research papers from ArXiv API
for reproducible ML pipeline data versioning.

Usage:
    python scripts/fetch_arxiv_data.py
    python scripts/fetch_arxiv_data.py --query "digital pathology" --max-results 100
"""

import argparse
import csv
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen

import defusedxml.ElementTree as ElementTree
import yaml  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("arxiv_fetch.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


class ArXivFetcher:
    """Fetch papers from ArXiv API with rate limiting and error handling."""

    BASE_URL = "http://export.arxiv.org/api/query"
    RATE_LIMIT_DELAY = 3  # seconds between requests

    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build_query_url(self, query: str, start: int = 0, max_results: int = 50) -> str:
        """Build ArXiv API query URL."""
        params = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        return f"{self.BASE_URL}?{urlencode(params)}"

    def fetch_papers(self, query: str, max_results: int = 50) -> list[dict[str, Any]]:
        """
        Fetch papers from ArXiv API.

        Args:
            query: Search query string
            max_results: Maximum number of papers to fetch

        Returns:
            List of paper dictionaries
        """
        logger.info(
            f"Fetching papers with query: '{query}', max_results: {max_results}"
        )

        papers: list[dict[str, Any]] = []
        start = 0
        batch_size = min(50, max_results)  # ArXiv API limit is ~50 per request

        while len(papers) < max_results:
            current_batch = min(batch_size, max_results - len(papers))
            url = self.build_query_url(query, start, current_batch)

            logger.info(
                f"Fetching batch {start // batch_size + 1}: {current_batch} papers"
            )

            try:
                # Validate URL scheme for security
                parsed_url = urlparse(url)
                if parsed_url.scheme not in ("http", "https"):
                    logger.error(f"Invalid URL scheme: {parsed_url.scheme}")
                    break

                with urlopen(url) as response:  # nosec B310
                    xml_data = response.read().decode("utf-8")

                # Parse XML response
                root = ElementTree.fromstring(xml_data)
                entries = root.findall("{http://www.w3.org/2005/Atom}entry")

                if not entries:
                    logger.warning("No more entries found")
                    break

                for entry in entries:
                    paper = self._parse_paper_entry(entry)
                    if paper is not None:
                        papers.append(paper)

                start += current_batch

                # Rate limiting
                if len(papers) < max_results:
                    logger.info(
                        f"Rate limiting: waiting {self.RATE_LIMIT_DELAY} seconds..."
                    )
                    time.sleep(self.RATE_LIMIT_DELAY)

            except Exception as e:
                logger.error(f"Error fetching batch: {e}")
                break

        logger.info(f"Successfully fetched {len(papers)} papers")
        return papers[:max_results]

    def _parse_paper_entry(self, entry) -> dict[str, Any] | None:
        """Parse individual paper entry from ArXiv XML response."""
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        try:
            # Extract basic information
            arxiv_id = entry.find("atom:id", ns).text.split("/")[-1]
            title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
            abstract = entry.find("atom:summary", ns).text.strip().replace("\n", " ")

            # Extract authors
            authors = []
            for author in entry.findall("atom:author", ns):
                name = author.find("atom:name", ns)
                if name is not None:
                    authors.append(name.text)

            # Extract categories
            categories = []
            for category in entry.findall("atom:category", ns):
                term = category.get("term")
                if term:
                    categories.append(term)

            # Extract dates
            published = entry.find("atom:published", ns).text[:10]  # YYYY-MM-DD
            updated = entry.find("atom:updated", ns).text[:10]

            # Extract URLs
            arxiv_url = entry.find("atom:id", ns).text
            pdf_url = None
            for link in entry.findall("atom:link", ns):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href")
                    break

            # Extract journal reference if available
            journal_ref = entry.find("arxiv:journal_ref", ns)
            journal = (
                journal_ref.text if journal_ref is not None else f"arXiv:{arxiv_id}"
            )

            # Extract DOI if available
            doi_elem = entry.find("arxiv:doi", ns)
            doi = doi_elem.text if doi_elem is not None else arxiv_url

            # Parse year from published date
            year = int(published.split("-")[0])

            return {
                "title": title,
                "authors": ", ".join(authors),
                "journal": journal,
                "year": year,
                "doi": doi,
                "abstract": abstract,
                "keywords": ", ".join(categories),
                "cited_by": 0,  # Not available from ArXiv API
                "methodology": ", ".join(categories),
                "dataset_used": "N/A",  # Not available from ArXiv API
                "arxiv_id": arxiv_id,
                "arxiv_categories": ", ".join(categories),
                "published_date": published,
                "updated_date": updated,
                "pdf_url": pdf_url,
                "arxiv_url": arxiv_url,
            }

        except Exception as e:
            logger.error(f"Error parsing paper entry: {e}")
            return None

    def save_to_csv(
        self, papers: list[dict[str, Any]], filename: str = "arxiv_publications.csv"
    ):
        """Save papers to CSV file."""
        csv_path = self.output_dir / filename

        if not papers:
            logger.warning("No papers to save")
            return

        # Define CSV columns
        fieldnames = [
            "title",
            "authors",
            "journal",
            "year",
            "doi",
            "abstract",
            "keywords",
            "cited_by",
            "methodology",
            "dataset_used",
            "arxiv_id",
            "arxiv_categories",
            "published_date",
            "updated_date",
            "pdf_url",
            "arxiv_url",
        ]

        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(papers)

            logger.info(f"Saved {len(papers)} papers to {csv_path}")

        except Exception as e:
            logger.error(f"Error saving CSV file: {e}")
            raise

    def save_metadata(
        self,
        query: str,
        papers: list[dict[str, Any]],
        filename: str = "arxiv_metadata.yaml",
    ):
        """Save metadata about the dataset."""
        metadata_path = self.output_dir / filename

        # Calculate some statistics
        years = [p["year"] for p in papers if p.get("year")]
        categories = []
        for paper in papers:
            if paper.get("arxiv_categories"):
                categories.extend(paper["arxiv_categories"].split(", "))

        from collections import Counter

        category_counts = Counter(categories)

        metadata = {
            "dataset_info": {
                "name": "ArXiv Research Publications",
                "description": f'Research papers fetched from ArXiv API using query: "{query}"',
                "source": "ArXiv API (http://export.arxiv.org/api/query)",
                "created_at": datetime.now().isoformat(),
                "query_used": query,
                "total_papers": len(papers),
            },
            "statistics": {
                "year_range": {
                    "min": min(years) if years else None,
                    "max": max(years) if years else None,
                },
                "categories": dict(category_counts.most_common(10)),
                "total_categories": len(category_counts),
            },
            "data_schema": {
                "columns": [
                    "title",
                    "authors",
                    "journal",
                    "year",
                    "doi",
                    "abstract",
                    "keywords",
                    "cited_by",
                    "methodology",
                    "dataset_used",
                    "arxiv_id",
                    "arxiv_categories",
                    "published_date",
                    "updated_date",
                    "pdf_url",
                    "arxiv_url",
                ],
                "data_types": {
                    "title": "string",
                    "authors": "string (comma-separated)",
                    "journal": "string",
                    "year": "integer",
                    "doi": "string",
                    "abstract": "string",
                    "keywords": "string (comma-separated)",
                    "cited_by": "integer",
                    "methodology": "string",
                    "dataset_used": "string",
                    "arxiv_id": "string",
                    "arxiv_categories": "string (comma-separated)",
                    "published_date": "date (YYYY-MM-DD)",
                    "updated_date": "date (YYYY-MM-DD)",
                    "pdf_url": "string (URL)",
                    "arxiv_url": "string (URL)",
                },
            },
            "quality_metrics": {
                "completeness": {
                    "titles": sum(1 for p in papers if p.get("title")),
                    "abstracts": sum(1 for p in papers if p.get("abstract")),
                    "authors": sum(1 for p in papers if p.get("authors")),
                    "categories": sum(1 for p in papers if p.get("arxiv_categories")),
                }
            },
        }

        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"Saved metadata to {metadata_path}")

        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
            raise


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description="Fetch ArXiv papers for ML pipeline")
    parser.add_argument(
        "--query",
        default="cat:eess.IV OR cat:cs.CV OR cat:q-bio.QM",
        help="ArXiv search query (default: medical imaging categories)",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=50,
        help="Maximum number of papers to fetch (default: 50)",
    )
    parser.add_argument(
        "--output-dir", default="data/raw", help="Output directory (default: data/raw)"
    )
    parser.add_argument(
        "--csv-file", default="arxiv_publications.csv", help="CSV output filename"
    )
    parser.add_argument(
        "--metadata-file",
        default="arxiv_metadata.yaml",
        help="Metadata output filename",
    )

    args = parser.parse_args()

    # Initialize fetcher
    fetcher = ArXivFetcher(output_dir=args.output_dir)

    try:
        # Fetch papers
        papers = fetcher.fetch_papers(query=args.query, max_results=args.max_results)

        if not papers:
            logger.error("No papers were fetched")
            sys.exit(1)

        # Save data
        fetcher.save_to_csv(papers, args.csv_file)
        fetcher.save_metadata(args.query, papers, args.metadata_file)

        logger.info("ArXiv data fetching completed successfully!")
        logger.info(f"Papers saved: {len(papers)}")
        logger.info(f"CSV file: {args.output_dir}/{args.csv_file}")
        logger.info(f"Metadata file: {args.output_dir}/{args.metadata_file}")

    except KeyboardInterrupt:
        logger.info("Fetching interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

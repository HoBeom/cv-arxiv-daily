"""
Copyright 2024 - Chansung Park

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
# modified by HoBeom
import argparse
from qa.paper.download import (
    download_pdf_from_arxiv,
    get_papers_from_hf_daily_papers,
    get_papers_from_arxiv_ids
)
from qa.paper.parser import extract_text_and_figures
from qa.gen.gemini import get_basic_qa, get_deep_qa
# from qa.utils import push_to_hf_hub

def _process_hf_daily_papers(args):
    print("1. Get a list of papers from 🤗 Daily Papers")
    target_date, papers = get_papers_from_hf_daily_papers(args.target_date)
    print("...DONE")
    
    print("2. Generating QAs for the paper")
    for paper in papers:
        try:
            title = paper['title']
            abstract = paper['paper']['summary']
            arxiv_id = paper['paper']['id']
            authors = []

            print(f"...PROCESSING ON[{arxiv_id}, {title}]")
            print(f"......Extracting authors' names")
            for author in paper['paper']['authors']:
                if 'user' in author:
                    fullname = author['user']['fullname']
                else:
                    fullname = author['name']
                authors.append(fullname)
            print(f"......DONE")

            print(f"......Downloading the paper PDF")
            filename = download_pdf_from_arxiv(arxiv_id)
            print(f"......DONE")

            print(f"......Extracting text and figures")
            texts, figures = extract_text_and_figures(filename)
            text =' '.join(texts)
            print(f"......DONE")

            print(f"......Generating the seed(basic) QAs")
            qnas = get_basic_qa(text, gemini_api_key=args.gemini_api, trucate=30000)
            qnas['title'] = title
            qnas['abstract'] = abstract
            qnas['authors'] = ','.join(authors)
            qnas['arxiv_id'] = arxiv_id
            qnas['target_date'] = target_date
            qnas['full_text'] = text
            print(f"......DONE")

            print(f"......Generating the follow-up QAs")
            qnas = get_deep_qa(text, qnas, gemini_api_key=args.gemini_api, trucate=30000)
            del qnas["qna"]
            print(f"......DONE")

            print(f"......Exporting to HF Dataset repo at [{args.hf_repo_id}]")
            push_to_hf_hub(qnas, args.hf_repo_id, args.hf_token)
            print(f"......DONE")
        except:
            print(".......failed due to exception")
            continue

    print("...DONE")

def _process_arxiv_ids(args):
    arxiv_ids = args.arxiv_ids

    print(f"1. Get metadata for the papers [{arxiv_ids}]")
    papers = get_papers_from_arxiv_ids(arxiv_ids)
    print("...DONE")
    
    print("2. Generating QAs for the paper")
    for paper in papers:
        try:
            title = paper['title']
            target_date = paper['target_date']
            abstract = paper['paper']['summary']
            arxiv_id = paper['paper']['id']
            authors = paper['paper']['authors']

            print(f"...PROCESSING ON[{arxiv_id}, {title}]")
            print(f"......Downloading the paper PDF")
            filename = download_pdf_from_arxiv(arxiv_id)
            print(f"......DONE")

            print(f"......Extracting text and figures")
            texts, figures = extract_text_and_figures(filename)
            text =' '.join(texts)
            print(f"......DONE")

            print(f"......Generating the seed(basic) QAs")
            qnas = get_basic_qa(text, gemini_api_key=args.gemini_api, trucate=30000)
            qnas['title'] = title
            qnas['abstract'] = abstract
            qnas['authors'] = ','.join(authors)
            qnas['arxiv_id'] = arxiv_id
            qnas['target_date'] = target_date
            qnas['full_text'] = text
            print(f"......DONE")

            print(f"......Generating the follow-up QAs")
            qnas = get_deep_qa(text, qnas, gemini_api_key=args.gemini_api, trucate=30000)
            del qnas["qna"]
            print(f"......DONE")

            print(f"......Exporting to HF Dataset repo at [{args.hf_repo_id}]")
            # push_to_hf_hub(qnas, args.hf_repo_id, args.hf_token)
            print(f"......DONE")
        except:
            print(".......failed due to exception")
            continue


def process_arxiv_ids(arxiv_ids:list, gemini_api_key:str):
    import logging
    contents = dict()

    logging.info(f"1. Get metadata for the papers [{arxiv_ids}]")
    papers = get_papers_from_arxiv_ids(arxiv_ids)
    logging.info("...DONE")
    
    logging.info("2. Generating QAs for the paper")
    for paper in papers:
        try:
            title = paper['title']
            target_date = paper['target_date']
            abstract = paper['paper']['summary']
            arxiv_id = paper['paper']['id']
            authors = paper['paper']['authors']

            logging.info(f"...PROCESSING ON[{arxiv_id}, {title}]")
            logging.info(f"......Downloading the paper PDF")
            filename = download_pdf_from_arxiv(arxiv_id)
            logging.info(f"......DONE")

            logging.info(f"......Extracting text and figures")
            texts, figures = extract_text_and_figures(filename)
            text =' '.join(texts)
            logging.info(f"......DONE")

            logging.info(f"......Generating the seed(basic) QAs")
            qnas = get_basic_qa(text, gemini_api_key=gemini_api_key, trucate=30000)
            qnas['title'] = title
            qnas['abstract'] = abstract
            qnas['authors'] = ','.join(authors)
            qnas['arxiv_id'] = arxiv_id
            qnas['target_date'] = target_date
            qnas['full_text'] = text
            logging.info(f"......DONE")

            logging.info(f"......Generating the follow-up QAs")
            qnas = get_deep_qa(text, qnas, gemini_api_key=gemini_api_key, trucate=30000)
            del qnas["qna"]
            logging.info(f"......DONE")

            contents[arxiv_id] = qnas
        except Exception as e:
            logging.info(f".......failed due to exception {e}")
            continue
    
    return contents

def main(args):
    if args.hf_daily_papers:
        _process_hf_daily_papers(args)
    else:
        _process_arxiv_ids(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="auto paper analysis")
    parser.add_argument("--gemini-api", type=str)
    parser.add_argument("--hf-token", type=str)
    parser.add_argument("--hf-repo-id", type=str)

    parser.add_argument('--hf-daily-papers', action='store_true')
    parser.add_argument("--target-date", type=str, default=None)

    parser.add_argument('--arxiv-ids', nargs='+')
    args = parser.parse_args()

    main(args)
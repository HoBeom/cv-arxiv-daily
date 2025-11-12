import argparse
import datetime
import json
import logging
import os
import re
import time

import arxiv
import requests
import yaml
from arxiv import HTTPError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    level=logging.INFO,
)
logging.info(f'Using script: {os.path.abspath(__file__)}')

base_url = 'https://arxiv.paperswithcode.com/api/v0/papers/'
github_url = 'https://api.github.com/search/repositories'

PAGE_SIZE = 10
DELAY_SECONDS = 7
NUM_RETRIES = 6
MIN_BACKOFF = 5


def _requests_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=['GET'],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount('http://', adapter)
    s.mount('https://', adapter)
    s.headers.update({
        'User-Agent':
        'cv-arxiv-daily/1.0 (+github.com/HoBeom/cv-arxiv-daily)'
    })
    return s


_SESSION = _requests_session()


def _get_json_safe(url: str):
    try:
        resp = _SESSION.get(url, timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logging.error('requests error on %s: %s', url, e)
        return None


def _throttled_client(page_size: int = PAGE_SIZE) -> arxiv.Client:
    return arxiv.Client(
        page_size=page_size,
        delay_seconds=DELAY_SECONDS,
        num_retries=NUM_RETRIES,
    )


def _results_with_backoff(client: arxiv.Client, search: arxiv.Search):
    backoff = MIN_BACKOFF
    while True:
        try:
            for r in client.results(search):
                yield r
            break
        except HTTPError as e:
            if getattr(e, 'status', None) == 429:
                logging.warning(
                    'HTTP 429 Too Many Requests — %ss waiting',
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
            else:
                raise


def load_config(config_file: str) -> dict:
    """
    config_file: input config file path
    return: a dict of configuration
    """

    def pretty_filters(**config) -> dict:
        keywords = dict()
        ESCAPE = '"'
        QUOTA = ''
        OR = ' OR '

        def parse_filters(filters: list):
            ret = ''
            for idx, filter in enumerate(filters):
                if len(filter.split()) > 1:
                    ret += ESCAPE + filter + ESCAPE
                else:
                    ret += QUOTA + filter + QUOTA
                if idx != len(filters) - 1:
                    ret += OR
            return ret

        for k, v in config['keywords'].items():
            keywords[k] = parse_filters(v['queries'])
        return keywords

    with open(config_file, 'r', encoding='utf-8') as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)
        config_data['kv'] = pretty_filters(**config_data)
        logging.info(f'{config_data=}')
    return config_data


def get_authors(authors, first_author=False):
    output = str()
    if not first_author:
        output = ', '.join(str(author) for author in authors)
    else:
        output = authors[0]
    return output


def sort_papers(papers):
    output = dict()
    keys = list(papers.keys())
    keys.sort(reverse=True)
    for key in keys:
        output[key] = papers[key]
    return output


def get_code_link(qword: str) -> str:
    query = f'{qword}'
    params = {'q': query, 'sort': 'stars', 'order': 'desc'}
    try:
        r = _SESSION.get(github_url, params=params, timeout=5)
        r.raise_for_status()
        results = r.json()
    except Exception as e:
        logging.error('github search error: %s', e)
        return None
    code_link = None
    if results.get('total_count', 0) > 0:
        code_link = results['items'][0]['html_url']
    return code_link


def _split_or_query(query: str, max_terms: int = 2) -> list:
    terms = re.findall(r'"([^"]+)"', query)
    if not terms:
        return [query]
    subqs = []
    for i in range(0, len(terms), max_terms):
        chunk = terms[i:i + max_terms]
        subqs.append(' OR '.join([f'"{t}"' for t in chunk]))
    return subqs


def get_daily_papers(query='slam', max_results=2):
    logging.info('[arxiv throttle] page_size=%s, delay=%ss, retries=%s' %
                 (PAGE_SIZE, DELAY_SECONDS, NUM_RETRIES))
    content = dict()
    logging.info(f'subqueries={_split_or_query(query)}')
    effective_page_size = min(PAGE_SIZE, max_results)
    client = _throttled_client(page_size=effective_page_size)
    subqueries = _split_or_query(query, max_terms=2)
    for idx, subq in enumerate(subqueries):
        search = arxiv.Search(
            query=subq,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )
        for result in _results_with_backoff(client, search):
            paper_id = result.get_short_id()
            paper_title = result.title
            paper_url = result.entry_id
            code_url = base_url + paper_id
            paper_first_author = get_authors(result.authors, first_author=True)
            update_time = result.updated.date()

            logging.info(
                'Time=%s title=%s author=%s',
                update_time,
                paper_title,
                paper_first_author,
            )

            ver_pos = paper_id.find('v')
            if ver_pos == -1:
                paper_key = paper_id
            else:
                paper_key = paper_id[0:ver_pos]

            if paper_key in content:
                continue
            try:
                r = _get_json_safe(code_url)
                repo_url = None
                if 'official' in r and r['official']:
                    repo_url = r['official']['url']
                if repo_url is not None:
                    content[paper_key] = (
                        '|**{}**|**{}**|{} et.al.|[{}]({})|[link]({})|null|\n'.
                        format(
                            update_time,
                            paper_title,
                            paper_first_author,
                            paper_id,
                            paper_url,
                            repo_url,
                        ))
                else:
                    content[paper_key] = (
                        '|**{}**|**{}**|{} et.al.|[{}]({})|null|null|\n'.
                        format(
                            update_time,
                            paper_title,
                            paper_first_author,
                            paper_id,
                            paper_url,
                        ))

            except Exception as e:
                logging.error(f'exception: {e} with id: {paper_key}')
        if idx != len(subqueries) - 1:
            time.sleep(5)
    return content


def update_paper_links(filename):
    """
    weekly update paper links in json file
    """
    with open(filename, 'r') as f:
        content = f.read()
        if not content:
            m = {}
        else:
            m = json.loads(content)

        json_data = m.copy()

        for keywords, v in json_data.items():
            logging.info(f'{keywords=}')
            for paper_id, contents in v.items():
                logging.info(f'Requests:{paper_id=}')
                contents = str(contents)

                valid_link = False if '|null|' in contents else True
                if valid_link:
                    continue
                try:
                    code_url = base_url + paper_id  # TODO
                    r = _get_json_safe(code_url)
                    repo_url = None
                    if 'official' in r and r['official']:
                        repo_url = r['official']['url']
                        if repo_url is not None:
                            new_cont = contents.replace(
                                '|null|', f'|**[link]({repo_url})**|')
                            logging.info(
                                'ID=%s contents=%s',
                                paper_id,
                                new_cont,
                            )
                            json_data[keywords][paper_id] = str(new_cont)
                except Exception as e:
                    logging.error(f'exception: {e} with id: {paper_id}')
        # dump to json file
        with open(filename, 'w') as f:
            json.dump(json_data, f)


def update_json_file(filename, data_dict):
    """
    daily update json file using data_dict
    """
    try:
        with open(filename, 'r') as f:
            content = f.read()
            if not content:
                m = {}
            else:
                m = json.loads(content)
        json_data = m.copy()
    except FileNotFoundError:
        logging.error(f'FileNotFoundError: {filename}')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        json_data = {}
    except Exception as e:
        logging.error(f'exception: {e} with filename: {filename}')
        json_data = {}

    for data in data_dict:
        for keyword in data.keys():
            papers = data[keyword]

            if keyword in json_data.keys():
                json_data[keyword].update(papers)
            else:
                json_data[keyword] = papers

    with open(filename, 'w') as f:
        json.dump(json_data, f)


def json_to_md(filename, md_filename, task='', to_web=False, use_title=True):
    """
    @param filename: str
    @param md_filename: str
    @return None
    """

    def pretty_math(s: str) -> str:
        ret = ''
        match = re.search(r'\$.*\$', s)
        if match is None:
            return s
        math_start, math_end = match.span()
        space_trail = space_leading = ''
        if s[:math_start][-1] != ' ' and '*' != s[:math_start][-1]:
            space_trail = ' '
        if s[math_end:][0] != ' ' and '*' != s[math_end:][0]:
            space_leading = ' '
        ret += s[:math_start]
        ret += f'{space_trail}${match.group()[1:-1].strip()}${space_leading}'
        ret += s[math_end:]
        return ret

    DateNow = datetime.date.today()
    DateNow = str(DateNow)
    DateNow = DateNow.replace('-', '.')

    with open(filename, 'r') as f:
        content = f.read()
        if not content:
            data = {}
        else:
            data = json.loads(content)

    with open(md_filename, 'w+') as f:
        pass

    with open(md_filename, 'a+') as f:

        if use_title and to_web:
            f.write('---\n' + 'layout: default\n' + '---\n\n')

        for keyword in data.keys():
            day_content = data[keyword]
            if not day_content:
                continue
            f.write(f'## {keyword}\n\n')
            f.write('### Updated on ' + DateNow + '\n\n')

            if use_title:
                if to_web:
                    f.write(
                        '|Publish Date|Title|Authors|PDF|Code|Quick Look|\n')
                    f.write('|:---------|:-----------------------|:---------|'
                            ':------|:------|:------|\n')
                else:
                    f.write(
                        '|Publish Date|Title|Authors|PDF|Code|Quick Look|\n')
                    f.write('|---|---|---|---|---|---|\n')

            day_content = sort_papers(day_content)

            for _, v in day_content.items():
                if v is not None:
                    f.write(pretty_math(v))

            f.write('\n')

            f.write(f'<p align=right>(<a href=## {keyword}>back to top</a>)'
                    f'</p>\n\n')

    logging.info(f'{task} finished')


def format_keyword_name(keyword):
    """Format the keyword name to lowercase with spaces replaced by dashes."""
    return keyword.lower().replace(' ', '-')


def update_homepage(config, homepage_path='docs/index.md'):
    homepage_content = '## Topics\n\n'
    for topic, details in config['keywords'].items():
        keyward_path = format_keyword_name(topic)
        homepage_content += f'- [{topic}](./{keyward_path}.md)\n\n'

    os.makedirs(os.path.dirname(homepage_path), exist_ok=True)
    with open(homepage_path, 'w', encoding='utf-8') as f:
        f.write('---\n' + 'layout: default\n' + '---\n\n')
        f.write(homepage_content)


def demo(**config):
    keywords = config[
        'kv']  # This now contains the file paths for each keyword
    max_results = config['max_results']
    json_file_path = {
        k: v['json_readme_path']
        for k, v in config['keywords'].items()
    }
    md_readme_path = {
        k: v['md_readme_path']
        for k, v in config['keywords'].items()
    }

    for topic, query in keywords.items():
        logging.info(f'{topic=}')
        logging.info(f'{query=}')

        json_file = json_file_path[topic]
        logging.info(f'{json_file=}')
        md_file = md_readme_path[topic]
        logging.info(f'{md_file=}')
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        os.makedirs(os.path.dirname(md_file), exist_ok=True)

        if config['publish_readme']:
            if config['update_paper_links']:
                update_paper_links(json_file)
            else:
                content = get_daily_papers(
                    query=query, max_results=max_results)
                update_json_file(json_file, [{topic: content}])

            json_to_md(json_file, md_file, task='Update Readme')

        if config['publish_gitpage']:
            gitpage_path = f'docs/{format_keyword_name(topic)}.md'
            json_to_md(
                json_file,
                gitpage_path,
                task=f'Update GitPage {topic}',
                to_web=True)

    if config['publish_gitpage']:
        logging.info('Update GitPage Home')
        update_homepage(config)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path',
        type=str,
        default='config.yaml',
        help='configuration file path')
    parser.add_argument(
        '--update_paper_links',
        default=False,
        action='store_true',
        help='whether to update paper links etc.',
    )
    args = parser.parse_args()
    config = load_config(args.config_path)
    config = {**config, 'update_paper_links': args.update_paper_links}
    demo(**config)

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import requests
import tiktoken
import yaml
from dotenv import load_dotenv
from openai import OpenAI

def load_config(config_path: str = 'config.yaml', env_path: str = None):
    """
    Loads configuration from a YAML file and corresponding environment file.
    
    Args:
        config_path: Path to the YAML config file
        env_path: Optional path to specific .env file
        
    Returns:
        dict: Loaded configuration
    """
    # Load environment variables
    if env_path:
        print(f"Loading environment variables from {env_path}")
        load_dotenv(env_path, override=True)
    else:
        # Use same name as config file but with .env extension
        env_file = Path(config_path).with_suffix('.env')
        load_dotenv(env_file, override=True)
    
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize API section if not present
    if 'api' not in config:
        config['api'] = {}
    if 'llm' not in config['api']:
        config['api']['llm'] = {}
    if 'github' not in config['api']:
        config['api']['github'] = {}
    if 'slack' not in config['api']:
        config['api']['slack'] = {}
    
    # Define default values
    default_processing = {
        'limits': {
            'max_input_tokens_per_request': 100000
        },
        'batch': {
            'generated_files_directory': 'batch_files',
            'cleanup': True,
            'keep_days': 7
        },
        'history': {
            'default_lookback_hours': 24,
            'use_last_run_file': True
        }
    }

    # Initialize processing section if not present
    config.setdefault('processing', {})
    
    # Deep merge defaults with existing config
    for section, values in default_processing.items():
        config['processing'].setdefault(section, {})
        for key, default_value in values.items():
            if not config['processing'][section].get(key):
                config['processing'][section][key] = default_value
    
    # Add initialization for runtime section
    if 'runtime' not in config:
        config['runtime'] = {}
    if config['runtime'].get('debug') is None or config['runtime'].get('debug') == '':
        config['runtime']['debug'] = False
    
    # Define default values for API sections
    default_api = {
        'llm': {
            'model': 'gpt-4.1',
            'base_url': 'https://api.openai.com/v1'
        }
    }

    # Deep merge defaults with existing config
    for service, values in default_api.items():
        config['api'].setdefault(service, {})
        for key, default_value in values.items():
            if not config['api'][service].get(key):
                config['api'][service][key] = default_value
    
    # Set API keys from environment variables
    config['api']['llm']['key'] = os.getenv("LLM_API_KEY")
    config['api']['github']['token'] = os.getenv("GH_TOKEN")
    config['api']['slack']['token'] = os.getenv("SLACK_TOKEN")
    
    # Validate required fields
    required_fields = {
        'api.llm': ['key'],
        'api.github': ['token', 'owner'],
        'api.slack': ['token', 'channel']
    }
    
    for path, fields in required_fields.items():
        section = config
        for part in path.split('.'):
            if part not in section:
                raise ValueError(f"Missing required section '{path}' in config")
            section = section[part]
        for field in fields:
            if not section.get(field):
                raise ValueError(f"Missing required field '{field}' in {path} section")
    
    # Handle optional repo setting
    if 'repo' not in config['api']['github']:
        config['api']['github']['repo'] = None
    
    return config

def setup_tokenizer():
    # estimate for simplicity..
    return tiktoken.get_encoding("cl100k_base")

tokenizer = setup_tokenizer()

def calculate_tokens(text):
    """
    Calculates the number of tokens.
    
    Args:
        text: Input text to tokenize
        tokenizer: Tokenizer instance
        
    Returns:
        int: Number of tokens in the text
    """
    return len(tokenizer.encode(text, disallowed_special=()))

def get_last_run_time(last_run_file: Path, config: dict) -> datetime:
    """
    Retrieves the timestamp of the last successful run.
    
    Args:
        last_run_file: Path to the file storing the last run timestamp
        config: Configuration dictionary containing history settings
        
    Returns:
        datetime: Timestamp of last run, or configured time ago if no record exists
    """
    # Check if we should use the last run file
    if config['processing']['history']['use_last_run_file']:
        try:
            with open(last_run_file, 'r') as f:
                timestamp = float(f.read().strip())
                print(f"Using last run timestamp: {datetime.fromtimestamp(float(timestamp))}")
                return datetime.fromtimestamp(float(timestamp))
        except (FileNotFoundError, ValueError):
            print("Last run file not found or invalid, falling back to default lookback time")
    
    # Use the configured lookback time
    lookback_hours = config['processing']['history']['default_lookback_hours']
    lookback_time = datetime.now() - timedelta(hours=lookback_hours)
    print(f"Using default lookback time: {lookback_time} ({lookback_hours} hours)")
    return lookback_time

def update_last_run_time(last_run_file: Path):
    """
    Updates the last run timestamp.
    
    Args:
        last_run_file: Path to store the timestamp
    """
    with open(last_run_file, 'w') as f:
        f.write(f"{time.time():.3f}")

def fetch_org_repos(owner: str, headers: dict) -> list:
    """
    Fetches all repositories for an organization.
    
    Args:
        owner: GitHub organization name
        headers: Request headers including authentication
        
    Returns:
        list: List of repository names
    """
    github_url = f"https://api.github.com/orgs/{owner}/repos"
    repos = []
    page = 1
    
    while True:
        try:
            response = requests.get(
                github_url, 
                headers=headers,
                params={
                    "page": page, 
                    "per_page": 100,
                    "type": "all"
                }
            )
            response.raise_for_status()
            page_repos = response.json()
            
            if not page_repos:
                break
                
            repos.extend([repo["name"] for repo in page_repos])
            page += 1
        except requests.exceptions.RequestException as e:
            print(f"Error fetching organization repositories: {e}")
            return []
            
    return repos

def fetch_github_issues(
    github_token: str,
    owner: str,
    repo: str | None,
    last_run_file: Path,
    config: dict
) -> list:
    """
    Fetches GitHub issues created or updated since the last run time.
    
    Args:
        github_token: GitHub authentication token
        owner: GitHub organization/owner name
        repo: Specific repository name or None to fetch from all repos
        last_run_file: Path to the file storing last run timestamp
        config: Configuration dictionary
    """
    since = get_last_run_time(last_run_file, config)
    headers = {"Authorization": f"token {github_token}"}
    
    try:
        # Test token with a simple API call
        test_response = requests.get("https://api.github.com/user", headers=headers)
        test_response.raise_for_status()
    except requests.exceptions.RequestException:
        print("Error: Invalid GitHub token or API access issue")
        return []
    
    all_issues = []
    repos_to_check = []
    
    if repo:
        repos_to_check = [repo]
    else:
        repos_to_check = fetch_org_repos(owner, headers)
        print(f"Found {len(repos_to_check)} repositories in organization")
    
    for repo in repos_to_check:
        github_url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        params = {"since": since.isoformat()}
        page = 1
        
        repo_issues = []  # Track issues for current repo
        while True:
            params["page"] = page
            try:
                response = requests.get(github_url, headers=headers, params=params)
                response.raise_for_status()
                page_issues = response.json()
                if not page_issues:
                    break
                
                # Add repo name to each issue for better context
                for issue in page_issues:
                    issue['repository_name'] = repo
                
                repo_issues.extend(page_issues)  
                page += 1
            except requests.exceptions.RequestException as e:
                print(f"Error fetching GitHub issues for repo {repo}: {e}")
                break  
                
        print(f"Found {len(repo_issues)} issues/PRs in {repo}")
        all_issues.extend(repo_issues) 
    
    return all_issues

def fetch_issue_comments(comments_url: str, headers: dict, token_limit: int) -> str:
    """
    Retrieves and concatenates comments for a GitHub issue, respecting token limits.
    
    Args:
        comments_url: URL endpoint for the issue's comments
        headers: Request headers including authentication
        token_limit: Maximum number of tokens allowed
        
    Returns:
        str: Concatenated comments text, separated by '---'
    """
    comments_text = ""
    response = requests.get(comments_url, headers=headers)
    comments = response.json()
    
    for comment in comments:
        comment_body = comment.get("body", "")
        current_tokens = calculate_tokens(comments_text)
        new_comment_tokens = calculate_tokens(comment_body)
        
        if current_tokens + new_comment_tokens > token_limit:
            print("Token limit reached for comments.")
            break
        
        if comments_text:
            comments_text += "\n---\n"
        comments_text += comment_body
    
    return comments_text

def prepare_llm_job(
    model: str,
    requests: list,
    batch_dir: str = "batch_files"
) -> Tuple[list, Path]:
    """
    Prepares a list of requests for LLM batch processing and saves them to a file.
    
    Returns:
        Tuple[list, Path]: List of tasks and the directory path containing the files
    """
    
    tasks = []
    
    for i, request in enumerate(requests):
        title = request.get("title", "")
        body = request.get("body", "")
        issue_url = request.get("html_url", "")
        repo_name = request.get("repository_name", "")
        comments_text = request.get("comments_text", "")
        
        task = {
            "custom_id": f"issue-{i+1}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [
                    {"role": "system", "content": (
                        "You are a helpful assistant that summarizes GitHub issues and PRs. "
                        "IMPORTANT: Only include information that is explicitly present in the provided text. "
                        "Do not make assumptions or add information that isn't directly stated.\n\n"
                        "For each issue/PR, structure your response as follows:\n"
                        "1. Start with a brief TL;DR (1-2 sentences)\n"
                        "2. Provide a detailed summary of the main points, using only information from the source\n"
                        "3. If code samples are present, highlight important code samples exactly as shown\n"
                        "4. List any explicitly mentioned action items or pending questions\n"
                        "5. [AI Suggestions] Clearly mark any AI-generated suggestions with this prefix\n\n"
                        "Formatting guidelines:\n"
                        "- Use single * for bold text (Slack format), never use **\n"
                        "- Use triple backticks for code blocks without language specifiers\n"
                        "- Use > for quotes or important highlights\n"
                        "- Use bullet points (-) for lists\n\n"
                        "If the issue is a question:\n"
                        "1. Summarize the question exactly as presented\n"
                        "2. If providing an answer not found in the text, prefix with '[AI Suggestions]'\n"
                        "3. If the question remains open, note it in the action items\n\n"
                        "Keep summaries factual and based solely on the provided content. "
                        "If suggesting additional context or solutions, clearly mark them as '*AI's Note:*' "
                        "to distinguish them from the original content."
                    )},
                    {"role": "user", "content": f"Repository: {repo_name}\nTitle: {title}\nBody: {body}\nComments: {comments_text}"},
                ],
            },
            "metadata": {
                "issue_url": issue_url,
                "title": title,
                "repo_name": repo_name
            }
        }
        tasks.append(task)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    file_dir = Path(batch_dir) / timestamp
    file_dir.mkdir(parents=True, exist_ok=True)
    
    input_path = file_dir / "batch_input.jsonl"

    try:
        with open(input_path, "w") as file:
            for task in tasks:
                file.write(json.dumps(task) + "\n")
    except IOError as e:
        print(f"Error writing batch file: {e}")
        return None

    return file_dir

def ensure_batch_directory(batch_dir: str = "batch_files") -> Path:
    """
    Creates and returns the batch directory path if it doesn't exist.
    
    Args:
        batch_dir: Name of the directory to store batch files
        
    Returns:
        Path: Path object for the batch directory
    """
    batch_path = Path(batch_dir)
    batch_path.mkdir(exist_ok=True)
    return batch_path

def cleanup_batch_files(keep_days: int = 7, batch_dir: str = "batch_files") -> None:
    """
    Removes batch directories older than specified days.
    """
    batch_path = Path(batch_dir)
    
    if keep_days == 0:
        # Immediate deletion of all run directories
        for dir_path in batch_path.iterdir():
            if dir_path.is_dir():
                try:
                    shutil.rmtree(dir_path)
                    print(f"Cleaned up directory: {dir_path}")
                except Exception as e:
                    print(f"Error cleaning up {dir_path}: {e}")
        return
        
    # Normal cleanup for keep_days > 0
    cutoff_time = datetime.now() - timedelta(days=keep_days)
    for dir_path in batch_path.iterdir():
        if not dir_path.is_dir():
            continue
            
        try:
            dir_time = datetime.strptime(dir_path.name, "%Y%m%d_%H%M%S_%f")
            if dir_time < cutoff_time:
                shutil.rmtree(dir_path)
                print(f"Cleaned up old directory: {dir_path}")
        except (ValueError, OSError) as e:
            print(f"Error processing {dir_path}: {e}")

def monitor_batch_status(client, batch_id: str, interval: int = 10) -> dict:
    """
    Monitors the status of a batch processing job.
    
    Args:
        client: OpenAI client instance
        batch_id: ID of the batch job to monitor
        interval: Time in seconds between status checks (default: 10)
        
    Returns:
        dict: Final batch status
    """
    while True:
        batch_status = client.batches.retrieve(batch_id)
        print("Batch status: {}".format(batch_status.status))
        print(
            f"Completed requests: {batch_status.request_counts.completed} / {batch_status.request_counts.total}"
        )

        if batch_status.status.lower() in ["completed", "failed", "canceled"]:
            break

        time.sleep(interval)
    
    return batch_status

def submit_llm_job(client: OpenAI, last_run_file: Path, file_dir: Path) -> str:
    input_path = file_dir / "batch_input.jsonl"
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
        
    try:
        batch_input_file = client.files.create(
            file=open(input_path, "rb"),
            purpose="batch"
        )
    except Exception as e:
        print(f"Error uploading batch file: {e}")
        raise

    # Create batch request
    response = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Batch request submitted. Batch ID: {response.id}")

    update_last_run_time(last_run_file)
    return response.id

def retrieve_result_file_contents(batch_status, client, file_dir: Path) -> Path:
    """
    Saves the results of a completed batch job to a local file.
    """
    result_file_id = batch_status.output_file_id
    results = client.files.content(result_file_id).content
    
    result_path = file_dir / "batch_results.jsonl"
    
    with open(result_path, "wb") as file:
        file.write(results)
    print(f"\nResults saved to {result_path}")

def chunk_message(text: str, limit: int = 35000) -> list:
    """
    Splits a message into chunks of specified size limit.
    
    Args:
        text: Message text to split
        limit: Character limit per message (default 35000)
        
    Returns:
        list: List of message chunks
    """
    chunks = []
    while text:
        if len(text) <= limit:
            chunks.append(text)
            break
        
        # Find the last newline within the limit
        split_index = text[:limit].rfind('\n')
        if split_index == -1:  # No newline found, just split at limit
            split_index = limit
            
        chunks.append(text[:split_index])
        text = text[split_index:].lstrip()
    
    return chunks

def post_to_slack(channel: str, text: str, token: str, debug: bool = False):
    """
    Posts a message to a Slack channel, breaking into multiple messages if needed.
    
    Args:
        channel: Name of the Slack channel
        text: Message text to post
        token: Slack API token
        debug: If True, print messages instead of posting to Slack
        
    Raises:
        Prints error message if posting fails (e.g., bot not in channel)
    """
    if debug:
        print("\n=== DEBUG: would post to Slack ===")
        print(f"Channel: {channel}")
        print("Message content:")
        print(text)
        print("===================================\n")
        return

    slack_url = "https://slack.com/api/chat.postMessage"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }
    
    # Split message into chunks if needed
    chunks = chunk_message(text)
    
    for chunk in chunks:
        data = {
            "channel": channel,
            "text": chunk,
            "unfurl_links": False,
            "unfurl_media": False
        }
        response = requests.post(slack_url, headers=headers, json=data)
        response_data = response.json()
        
        if not response_data.get("ok"):
            error = response_data.get("error", "Unknown error")
            print(f"Failed to send message to Slack: {error}")
            if error == "not_in_channel":
                print("The bot is not in the channel. Please invite the bot to the channel.")
            break  # Stop sending chunks if there's an error
        
        # Add a small delay between chunks to avoid rate limiting
        if len(chunks) > 1:
            time.sleep(10)

def process_and_post_results(
    org_name: str,
    slack_channel: str,
    slack_token: str,
    file_dir: Path,
    debug: bool = False
) -> None:
    """
    Processes batch results and posts them to Slack.
    
    Args:
        org_name: GitHub organization name
        slack_channel: Slack channel to post to
        slack_token: Slack API token
        file_dir: Directory containing the input and output files
        debug: If True, print messages instead of posting to Slack
    """
    input_path = file_dir / "batch_input.jsonl"
    output_path = file_dir / "batch_results.jsonl"
    
    today_date = datetime.now().strftime("%B %d, %Y")
    issue_url_map = {}
    repo_results = {}
    
    # Create issue URL map
    with open(input_path, "r") as input_file:
        for line in input_file:
            task = json.loads(line)
            custom_id = task.get("custom_id", "N/A")
            metadata = task.get("metadata", {})
            issue_url_map[custom_id] = (
                metadata.get("issue_url", "No URL available"),
                metadata.get("title", ""),
                metadata.get("repo_name", "unknown")
            )
    
    # Process results and organize by repository
    with open(output_path, "r") as output_file:
        for line in output_file:
            result = json.loads(line)
            custom_id = result.get("custom_id", "N/A")
            response_content = result.get("response", {}).get("body", {}).get("choices", [{}])[0].get("message", {}).get("content", "No content available")
            issue_url, title, repo_name = issue_url_map.get(custom_id, ("No URL available", "No title available", "unknown"))
            
            if repo_name not in repo_results:
                repo_results[repo_name] = []
            
            repo_results[repo_name].append(f"*Title:* <{issue_url}|[{title}]>\n{response_content}\n──────────────────────────────────────\n\n")
    
    # Create combined message grouped by repository
    combined_message = f"*Latest Updates for {org_name} ({today_date})*\n\n"
    
    for repo_name, summaries in repo_results.items():
        combined_message += f"*Repository: {repo_name}*\n"
        combined_message += "".join(summaries)
        combined_message += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    
    # Split into chunks if necessary and post
    chunks = chunk_message(combined_message)
    for chunk in chunks:
        post_to_slack(slack_channel, chunk, slack_token, debug=debug)
    
    if debug:
        print("Debug mode: Skipped posting to Slack")
    else:
        print(f"Updates posted to Slack channel {slack_channel}")

def process_issue_content(issue: dict, max_input_tokens_per_request: int, headers: dict) -> dict:
    """
    Processes an issue's content to fit within token limits, including fetching comments if space allows.
    
    Args:
        issue: Dictionary containing issue data
        max_input_tokens_per_request: Maximum number of requested input tokens per request
        headers: Request headers for GitHub API
        
    Returns:
        dict: Updated issue with processed content including comments 
    """
    base_content = f"Repository: {issue['repository_name']}\nTitle: {issue['title']}\nBody: {issue['body']}"
    base_tokens = calculate_tokens(base_content)
    
    if base_tokens > max_input_tokens_per_request:
        print(f"Issue {issue['number']} exceeds token limit. Truncating body.")
        excess_tokens = base_tokens - max_input_tokens_per_request
        chars_to_remove = excess_tokens * 4
        issue['body'] = issue['body'][:-chars_to_remove]
    else:
        remaining_tokens = max_input_tokens_per_request - base_tokens
        if issue.get("comments", 0) > 0 and remaining_tokens > 0:
            issue['comments_text'] = fetch_issue_comments(
                issue.get("comments_url", ""),
                headers,
                remaining_tokens
            )
    
    return issue

def main():
    print("\n=== GitHub Issue/PR Summarizer Starting ===")
    parser = argparse.ArgumentParser(description='GitHub Issue/PR Summarizer')
    parser.add_argument('--config', default='config.yaml',
                       help='Path to the YAML configuration file')
    parser.add_argument('--env', help='Path to the environment file (optional)')
    args = parser.parse_args()
    
    try:
        config = load_config(args.config, args.env)
        print(f"\nConfiguration loaded successfully")
        print(f"- Organization: {config['api']['github']['owner']}")
        print(f"- Target repo: {config['api']['github']['repo'] or 'All repos'}")
        print(f"- Slack channel: {config['api']['slack']['channel']}")
    except (yaml.YAMLError, ValueError) as e:
        print(f"Configuration error: {e}")
        return
    
    # Create last_run file path from config path
    last_run_file = Path(args.config).with_suffix('.last_run')
    
    # Fetch issues first
    print(f"\nFetching GitHub issues/PRs since last run...")
    issues = fetch_github_issues(
        github_token=config['api']['github']['token'],
        owner=config['api']['github']['owner'],
        repo=config['api']['github']['repo'],
        last_run_file=last_run_file,
        config=config
    )
    
    if len(issues) == 0:
        print("No new updates to report")
        return
    
    print(f"Found {len(issues)} issues/PRs to process")
    
    # Before processing content
    print("\nProcessing issue/PR content and comments...")
    headers = {"Authorization": f"token {config['api']['github']['token']}"}
   
    for i in range(len(issues)):
        print(f"Processing issue {i+1}/{len(issues)}: {issues[i].get('title', 'No title')[:60]}...")
        issues[i] = process_issue_content(
            issues[i], 
            config['processing']['limits']['max_input_tokens_per_request'], 
            headers
        )
    
    # Before preparing job
    print("\nPreparing LLM batch job...")
    file_dir = prepare_llm_job(
        model=config['api']['llm']['model'],
        requests=issues,
        batch_dir=config['processing']['batch']['generated_files_directory']
    )
    
    # Before submitting job
    print("\nSubmitting batch job to LLM API...")
    client = OpenAI(
        api_key=config['api']['llm']['key'],
        base_url=config['api']['llm']['base_url']
    )
    
    batch_id = submit_llm_job(
        client=client,
        last_run_file=last_run_file,
        file_dir=file_dir
    )
    
    print("\nMonitoring batch job status...")
    batch_status = monitor_batch_status(client, batch_id)
    
    if batch_status.status.lower() == "completed":
        print("\nBatch job completed successfully")
        retrieve_result_file_contents(batch_status, client, file_dir)
        print("\nPosting results to Slack...")
        process_and_post_results(
            org_name=config['api']['github']['owner'],
            slack_channel=config['api']['slack']['channel'],
            slack_token=config['api']['slack']['token'],
            file_dir=file_dir,
            debug=config['runtime']['debug']
        )
    else:
        print(f"\nBatch job failed with status: {batch_status.status}")
            
    if config['processing']['batch']['cleanup']:
        print("\nCleaning up batch files...")
        cleanup_batch_files(
            keep_days=config['processing']['batch']['keep_days'],
            batch_dir=config['processing']['batch']['generated_files_directory']
        )

    print("\n=== GitHub Issue/PR Summarizer Completed ===\n")

if __name__ == "__main__":
    main()




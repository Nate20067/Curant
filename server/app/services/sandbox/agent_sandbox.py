import os 
import tempfile
import logging
import shutil
import docker
import httpx
import requests
from bs4 import BeautifulSoup
from typing import Optional, Tuple, Union, List, Dict
from urllib.parse import urlparse, quote
from docker.errors import APIError
from datetime import datetime
from app.services.sandbox.sandbox_helper import read_repo
from app.services.sandbox.sandbox_code_similarity import CodeReviewValidator
from app.services.sandbox.rate_limiter import RateLimiter
import pathlib


class Sandbox:
    """Sandbox environment for running containerized development operations"""
    
    def __init__(
        self,
        repo_url_str: str,
        branch_name: str = "agent-changes",
        image: str = "python:3.9",
        repo_branch: str | None = None
    ):
        """Initialize sandbox with docker client and temp directories"""
        self.client = docker.from_env() 
        #Creating temp directory to store the users repo
        temp_dir = pathlib.Path(tempfile.mkdtemp()) #creates temp dir
        self.temp_dir = temp_dir
        #Working directory for the agent to clone the github in
        self.workdir = temp_dir / "agent_repo"
        #Creating the repo object -> sets to None 
        self.repo = None 
        self.container = None
        self.container_workdir = "/workspace" #Directory mounted inside container
        self.branch_name = branch_name #Branch name for agent changes
        self.repo_branch = repo_branch
        self.agent_updates: List[Dict[str, str]] = []
        
        #Initializing code review validator for intelligent validation
        self.code_validator = CodeReviewValidator()
        
        #Initializing rate limiter for web requests (5 requests per 60 seconds per domain)
        self.web_rate_limiter = RateLimiter(max_calls=5, time_window=60.0)

        #Retrieving the docker file and the repo object
        docker_path, self.repo = read_repo(
            repo_object=self.repo,
            repo_str=repo_url_str,
            workdir=self.workdir,
            checkout_branch=self.repo_branch
        )
        
        #Creating new branch for agent changes
        self._create_branch(branch_name)
        
        #Starting the docker container with volume mount
        self.container = self.client.containers.run(
            image,
            detach=True,
            tty=True,
            stdin_open=True,
            working_dir=self.container_workdir,
            volumes={
                str(self.workdir): {
                    "bind": self.container_workdir,
                    "mode": "rw"
                }
            }
        )

    #Creating branch method -> creates new git branch for agent changes
    def _create_branch(self, branch_name: str):
        """Creates a new git branch for agent modifications"""
        try:
            #Creating new branch from current HEAD
            new_branch = self.repo.create_head(branch_name)
            #Checking out the new branch
            new_branch.checkout()
            logging.info(f"Created and checked out branch: {branch_name}")
        except Exception as e:
            #If branch exists checking it out instead
            logging.warning(f"Branch {branch_name} may exist, checking out: {e}")
            self.repo.heads[branch_name].checkout()

    #Designer tool methods
    def write_design_doc(self, file: str, content: str):
        """Create design document (designer tool)"""
        design_path = f"design_docs/{file}"
        return self.create_file(design_path, content)

    def update_design_doc(self, file: str, content: str):
        """Update design document (designer tool)"""
        design_path = f"design_docs/{file}"
        return self.update_file(design_path, content, append=False)

    def append_design_note(self, file: str, note: str):
        """Append note to design doc (designer tool)"""
        design_path = f"design_docs/{file}"
        return self.update_file(design_path, f"\n{note}", append=True)

    def save_task_breakdown(self, file: str, breakdown: str):
        """Save task breakdown (designer tool)"""
        design_path = f"design_docs/{file}"
        return self.create_file(design_path, breakdown)

    def read_design_doc(self, file: str):
        """Read design document (designer tool)"""
        design_path = f"design_docs/{file}"
        return self.read_file(design_path)

    def save_diagram(self, file: str, diagram: str):
        """Save architecture diagram (designer tool)"""
        design_path = f"design_docs/{file}"
        return self.create_file(design_path, diagram)

    #Agent communication methods
    def write_agent_update(self, role: str, message: str, audience: Optional[str] = None) -> dict:
        """Store shared agent notes so designer/programmer/validator stay in sync."""
        role_name = (role or "agent").strip().lower()
        entry = {
            "role": role_name,
            "message": (message or "").strip(),
            "audience": (audience or "all").strip().lower(),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
        if not entry["message"]:
            raise ValueError("agent update message cannot be empty")

        self.agent_updates.append(entry)
        logging.debug("Agent update recorded: %s", entry)
        return {
            "success": True,
            "entry": entry,
            "total_updates": len(self.agent_updates)
        }

    def read_agent_updates(self, limit: int = 5, include_roles: Optional[list] = None, audience: Optional[str] = None) -> dict:
        """Retrieve recent shared agent notes."""
        notes = self.agent_updates
        if include_roles:
            allowed = {role.strip().lower() for role in include_roles if role}
            notes = [entry for entry in notes if entry["role"] in allowed]
        audience_filter = (audience or "").strip().lower()
        if audience_filter:
            notes = [entry for entry in notes if entry["audience"] in (audience_filter, "all")]

        if limit and limit > 0:
            notes = notes[-limit:]

        return {
            "success": True,
            "updates": notes
        }

    #Creating a method for the agent to create a file
    def create_file(self, relative_file: str, content: str):
        """Creates a new file with the given content"""
        #Appending the path to the relative file that is passed to it
        path = pathlib.Path(self.workdir) / relative_file
        
        #Creating parent directories if they don't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        #Checking if file already exists in the directory
        file_exists = path.exists()
        
        #Condition to check if the file exists or not
        if not file_exists:
            #Going to create the file -> write agents content inside the file  
            with open(path, 'w') as write:
                write.write(content) #Writing agents content to the file
            #Adding file to git staging
            self.repo.index.add([relative_file])
            return {"success": True, "file": relative_file, "action": "created"}
        else:
            #If file exists calling update file method instead
            return self.update_file(relative_file, content)

    #Update file method -> allows for agent to update a given file in the list 
    def update_file(self, relative_file: str, content: str, append: bool = False):
        """Updates an existing file with new content or creates it if it doesn't exist"""
        #Appending the path to the relative file that is passed to it 
        target_file = pathlib.Path(self.workdir) / relative_file
        
        #Ensuring that the file is actually in the directory
        if not target_file.exists():
            #If file is not in directory calling create file 
            return self.create_file(relative_file=relative_file, content=content)
        
        #Creating parent directories if they don't exist
        target_file.parent.mkdir(parents=True, exist_ok=True)
        
        #Setting mode to append or write based on parameter
        mode = 'a' if append else 'w'
        
        #If file exists we are going to write/append to the file
        with open(target_file, mode) as write:
            #Writing/appending content to the file
            write.write(content)
        
        #Adding updated file to git staging
        self.repo.index.add([relative_file])
        
        action = "appended" if append else "updated"
        return {"success": True, "file": relative_file, "action": action}

    #Execute command method -> runs shell commands in the container
    def exec_file(self, command: Union[str, list], workdir: Optional[str] = None, environment: Optional[dict] = None) -> Tuple[int, str]:
        """Runs shell commands in the sandbox container -> allows for mv/exec/linux commands"""
        #Checking if container is initialized before running commands
        if not self.container:
            raise httpx.HTTPError("Sandbox container is not initialized")

        #Setting up the command to execute
        exec_cmd = command
        if isinstance(command, str):
            #Wrapping string commands in shell for proper execution
            exec_cmd = ["/bin/sh", "-c", command]
        
        try:
            #Executing the command inside the container
            exec_result = self.container.exec_run(
                cmd=exec_cmd,
                workdir=workdir or self.container_workdir,
                environment=environment,
                stdout=True,
                stderr=True,
            )
        except APIError as err:
            #Logging error if command execution fails
            logging.error("Failed running exec command: %s", err)
            raise httpx.HTTPError("Failed running exec command") from err

        #Getting output from the command execution
        output = exec_result.output
        if isinstance(output, (bytes, bytearray)):
            #Decoding bytes output to string format
            output = output.decode("utf-8", errors="replace")

        #Returning exit code and output from the command
        return exec_result.exit_code, output

    #Test file method -> runs tests on executed file
    def test_file(self, relative_file: str, test_command: str) -> dict:
        """Tests a file by executing test command in container"""
        #Running execution test with provided test command
        test_results = {
            'file': relative_file,
            'execution_success': False,
            'execution_output': '',
            'exit_code': None,
            'needs_revision': False,
            'message': ''
        }
        
        try:
            #Executing the test command in container
            exit_code, output = self.exec_file(test_command)
            test_results['exit_code'] = exit_code
            test_results['execution_success'] = (exit_code == 0)
            test_results['execution_output'] = output
            
            if exit_code != 0:
                #Test execution failed
                test_results['needs_revision'] = True
                test_results['message'] = f"Execution failed with exit code {exit_code}: {output}"
                logging.error(f"Test execution failed for {relative_file}: {output}")
            else:
                #Test execution passed
                test_results['message'] = f"Test execution passed successfully"
                logging.info(f"Test execution passed for {relative_file}")
                
        except Exception as e:
            #Exception during test execution
            test_results['needs_revision'] = True
            test_results['message'] = f"Test execution error: {str(e)}"
            logging.error(f"Test execution error for {relative_file}: {e}")
        
        return test_results

    #Validator tool methods
    def check_code_similarity(self, relative_file: str, design_spec: str, threshold: float = 0.45) -> dict:
        """Alias for check_code_review to match validator tool schema"""
        review = self.check_code_review(relative_file, design_spec, include_tests=False)
        # Return simplified result matching validator expectations
        return {
            'file': relative_file,
            'similarity_score': review['score'],
            'passes_threshold': review['score'] >= threshold,
            'issues': review['issues'],
            'approved': review['approved'],
            'needs_revision': review['needs_revision']
        }

    def report_runtime_error(self, error_output: str, context: str = "") -> dict:
        """Log runtime error for validator (validator tool)"""
        logging.error(f"Runtime error reported: {context}")
        logging.error(error_output)
        return {
            'error_logged': True,
            'error_output': error_output,
            'context': context,
            'message': 'Error has been logged and is ready for programmer review'
        }

    #Check code review method -> uses LLM to intelligently validate code against design spec
    def check_code_review(self, relative_file: str, design_spec: str, include_tests: bool = False) -> dict:
        """Uses AI to review if code correctly implements the design specification"""
        #Reading the file content for validation
        try:
            file_content = self.read_file(relative_file)
        except FileNotFoundError as e:
            #File not found error
            return {
                'file': relative_file,
                'approved': False,
                'score': 0.0,
                'issues': [f"File not found: {str(e)}"],
                'suggestions': [],
                'summary': f"File {relative_file} does not exist",
                'needs_revision': True
            }
        
        #Optionally run tests first to include execution results
        execution_results = None
        if include_tests:
            #Attempt to run tests and include results in review
            try:
                test_command = f"python {relative_file}"  #Basic execution test
                execution_results = self.test_file(relative_file, test_command)
            except Exception as e:
                logging.warning(f"Could not run tests for {relative_file}: {e}")
        
        try:
            #Performing intelligent code review with LLM
            review_result = self.code_validator.review_code(
                code_content=file_content,
                design_spec=design_spec,
                file_path=relative_file,
                execution_results=execution_results
            )
            
            #Adding file path to result
            review_result['file'] = relative_file
            
            #Logging review outcome
            if review_result['approved']:
                logging.info(f"Code review passed for {relative_file}: score={review_result['score']:.2f}")
            else:
                logging.warning(f"Code review failed for {relative_file}: {len(review_result['issues'])} issues found")
            
            return review_result
                
        except Exception as e:
            #Exception during code review
            logging.error(f"Code review error for {relative_file}: {e}")
            return {
                'file': relative_file,
                'approved': False,
                'score': 0.0,
                'issues': [f"Review error: {str(e)}"],
                'suggestions': [],
                'summary': f"Code review failed with error: {str(e)}",
                'needs_revision': True
            }

    #Sandbox method for agent to read a file from the directory
    def read_file(self, relative_file: str) -> str:
        """Reads and returns the contents of a file from the sandbox"""
        #Appending the path to the relative file that is passed to it
        target_file = pathlib.Path(self.workdir) / relative_file
        
        #Checking if the file exists before reading
        if not target_file.exists():
            raise FileNotFoundError(f"File {relative_file} does not exist")
        
        #Reading the file contents
        with open(target_file, 'r') as read:
            file_contents = read.read()
        
        return file_contents #Returning contents of the file
    
    #Web search method -> allows agent to search web for error troubleshooting and documentation
    def search_web(self, query: str, max_results: int = 5) -> dict:
        """Searches the web for information to help with errors or task completion"""
        logging.info(f"Searching web for: {query}")
        
        #Creating search results dictionary
        search_results = {
            'query': query,
            'results': [],
            'success': False,
            'message': ''
        }
        
        #Extracting domain for rate limiting
        domain = "html.duckduckgo.com"
        
        #Checking rate limit before making request
        try:
            self.web_rate_limiter.wait_if_needed(domain, max_wait=30.0)
        except RuntimeError as e:
            #Rate limit exceeded
            logging.error(f"Rate limit error for search: {e}")
            search_results['message'] = f"Rate limit exceeded: {str(e)}"
            return search_results
        
        try:
            #Using DuckDuckGo HTML search for web queries
            search_url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            
            #Recording the call after successful request
            self.web_rate_limiter.record_call(domain)
            
            if response.status_code == 200:
                #Parsing search results
                soup = BeautifulSoup(response.content, 'html.parser')
                result_divs = soup.find_all('div', class_='result')
                
                for idx, result in enumerate(result_divs[:max_results]):
                    #Extracting title and snippet from results
                    title_tag = result.find('a', class_='result__a')
                    snippet_tag = result.find('a', class_='result__snippet')
                    
                    if title_tag and snippet_tag:
                        search_results['results'].append({
                            'title': title_tag.get_text(strip=True),
                            'url': title_tag.get('href', ''),
                            'snippet': snippet_tag.get_text(strip=True)
                        })
                
                search_results['success'] = True
                search_results['message'] = f"Found {len(search_results['results'])} results"
                logging.info(f"Web search successful: {len(search_results['results'])} results")
            else:
                #Search request failed
                search_results['message'] = f"Search failed with status code {response.status_code}"
                logging.error(f"Web search failed: {response.status_code}")
                
        except Exception as e:
            #Exception during web search
            search_results['message'] = f"Search error: {str(e)}"
            logging.error(f"Web search error: {e}")
        
        return search_results
    
    #Fetch web page method -> retrieves full content from a URL for documentation
    def fetch_web_page(self, url: str) -> dict:
        """Fetches and extracts text content from a web page for documentation"""
        logging.info(f"Fetching web page: {url}")
        
        #Creating fetch results dictionary
        fetch_results = {
            'url': url,
            'content': '',
            'title': '',
            'success': False,
            'message': ''
        }
        
        #Extracting domain for rate limiting
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
        except Exception as e:
            logging.error(f"Invalid URL format: {e}")
            fetch_results['message'] = f"Invalid URL: {str(e)}"
            return fetch_results
        
        #Checking rate limit before making request
        try:
            self.web_rate_limiter.wait_if_needed(domain, max_wait=30.0)
        except RuntimeError as e:
            #Rate limit exceeded
            logging.error(f"Rate limit error for {domain}: {e}")
            fetch_results['message'] = f"Rate limit exceeded: {str(e)}"
            return fetch_results
        
        try:
            #Fetching the web page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=15)
            
            #Recording the call after successful request
            self.web_rate_limiter.record_call(domain)
            
            if response.status_code == 200:
                #Parsing page content
                soup = BeautifulSoup(response.content, 'html.parser')
                
                #Removing script and style elements
                for script in soup(['script', 'style', 'nav', 'footer', 'header']):
                    script.decompose()
                
                #Getting page title
                title_tag = soup.find('title')
                fetch_results['title'] = title_tag.get_text(strip=True) if title_tag else 'No title'
                
                #Extracting main text content
                text_content = soup.get_text(separator='\n', strip=True)
                #Limiting content size to prevent overflow
                fetch_results['content'] = text_content[:10000]
                
                fetch_results['success'] = True
                fetch_results['message'] = 'Page fetched successfully'
                logging.info(f"Successfully fetched page: {url}")
            else:
                #Fetch request failed
                fetch_results['message'] = f"Fetch failed with status code {response.status_code}"
                logging.error(f"Web fetch failed: {response.status_code}")
                
        except Exception as e:
            #Exception during web fetch
            fetch_results['message'] = f"Fetch error: {str(e)}"
            logging.error(f"Web fetch error: {e}")
        
        return fetch_results
    
    #Stage files method -> stages specific files to git index before commit
    def stage_files(self, file_list: list) -> dict:
        """Stages specific files to git index before committing"""
        logging.info(f"Staging files: {file_list}")
        
        #Creating staging results dictionary
        staging_results = {
            'staged_files': [],
            'failed_files': [],
            'success': False,
            'message': ''
        }
        
        try:
            for relative_file in file_list:
                #Checking if file exists before staging
                target_file = pathlib.Path(self.workdir) / relative_file
                
                if target_file.exists():
                    #Staging file to git index
                    self.repo.index.add([relative_file])
                    staging_results['staged_files'].append(relative_file)
                    logging.info(f"Staged file: {relative_file}")
                else:
                    #File doesn't exist
                    staging_results['failed_files'].append({
                        'file': relative_file,
                        'reason': 'File not found'
                    })
                    logging.warning(f"Cannot stage {relative_file}: File not found")
            
            #Setting success status
            if staging_results['staged_files']:
                staging_results['success'] = True
                staging_results['message'] = f"Successfully staged {len(staging_results['staged_files'])} file(s)"
            else:
                staging_results['message'] = 'No files were staged'
            
        except Exception as e:
            #Exception during staging
            staging_results['message'] = f"Staging error: {str(e)}"
            logging.error(f"File staging error: {e}")
        
        return staging_results 
    
    #Move file method -> moves files around inside the sandbox    
    def move_file(self, source_file: str, destination_file: str) -> str:
        """Moves files around inside sandbox directory"""
        #Creating full paths for source and destination
        src_path = pathlib.Path(self.workdir) / source_file
        dest_path = pathlib.Path(self.workdir) / destination_file
        
        #Checking if source file exists before moving
        if not src_path.exists():
            raise FileNotFoundError(f"Source file {source_file} does not exist")
        
        #Creating destination parent directories if needed
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        #Moving the file from source to destination
        shutil.move(str(src_path), str(dest_path))
        
        #Updating git index for moved files
        self.repo.index.remove([source_file], working_tree=True)
        self.repo.index.add([destination_file])
        
        return str(dest_path) #Returning the destination path

    #Delete file method -> removes files or directories from sandbox
    def delete_file(self, relative_file: str) -> bool:
        """Deletes a file or directory from sandbox"""
        #Creating full path to the target file/directory
        target_path = pathlib.Path(self.workdir) / relative_file
        
        #Checking if the target exists before deletion
        if not target_path.exists():
            return False

        #Checking if target is a directory or file
        if target_path.is_dir():
            #Removing entire directory tree if it's a directory
            shutil.rmtree(target_path)
        else:
            #Removing single file if it's not a directory
            target_path.unlink()
        
        #Removing from git index
        try:
            self.repo.index.remove([relative_file], working_tree=True)
        except Exception as e:
            logging.warning(f"Could not remove {relative_file} from git: {e}")
        
        return True #Returning success status

    #Commit changes method -> commits staged changes to git
    def commit_changes(self, message: str = "Agent automated changes"):
        """Commits all staged changes to the current branch"""
        try:
            #Checking if there are changes to commit
            if self.repo.is_dirty(untracked_files=True):
                #Adding all untracked files
                untracked = self.repo.untracked_files
                if untracked:
                    self.repo.index.add(untracked)
                
                #Committing changes
                self.repo.index.commit(message)
                logging.info(f"Committed changes: {message}")
                return True
            else:
                logging.info("No changes to commit")
                return False
        except Exception as e:
            logging.error(f"Failed to commit changes: {e}")
            raise

    #Push to remote method -> pushes branch to remote repository
    def push_to_remote(self, remote_name: str = "origin"):
        """Pushes the current branch to remote repository"""
        try:
            #Getting the remote repository
            remote = self.repo.remote(name=remote_name)
            
            #Pushing current branch to remote
            push_info = remote.push(self.branch_name)
            
            #Checking push results
            for info in push_info:
                if info.flags & info.ERROR:
                    raise Exception(f"Push failed: {info.summary}")
            
            logging.info(f"Successfully pushed branch {self.branch_name} to {remote_name}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to push to remote: {e}")
            raise

    #Cleanup method -> commits, pushes changes, stops container and removes temp directories
    def cleanup(self, commit_message: str = "Agent automated changes", push_to_remote: bool = True):
        """Commits and pushes changes, then stops the sandbox container and cleans temp directories"""
        #Committing any pending changes before cleanup
        try:
            if self.repo and self.repo.is_dirty(untracked_files=True):
                logging.info("Committing pending changes before cleanup")
                self.commit_changes(commit_message)
                
                #Pushing to remote if requested
                if push_to_remote:
                    logging.info("Pushing changes to remote repository")
                    self.push_to_remote()
        except Exception as e:
            logging.error(f"Error during git operations in cleanup: {e}")
        
        #Checking if container exists before stopping
        if hasattr(self, "container") and self.container:
            try:
                #Attempting to stop the running container
                self.container.stop()
            except Exception as stop_error:
                #Logging error if container stop fails
                logging.error("Error stopping container: %s", stop_error)
            try:
                #Attempting to remove the container
                self.container.remove(v=True, force=True)
            except Exception as remove_error:
                #Logging error if container removal fails
                logging.error("Error removing container: %s", remove_error)
            finally:
                #Setting container to None after cleanup
                self.container = None

        #Checking if temp directory exists before removal
        if hasattr(self, "temp_dir"):
            #Removing the temporary directory tree
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    #Destructor method -> ensures cleanup on object deletion
    def __del__(self):
        """Ensures cleanup when object is destroyed"""
        try:
            self.cleanup()
        except Exception as e:
            logging.error(f"Error in destructor cleanup: {e}")

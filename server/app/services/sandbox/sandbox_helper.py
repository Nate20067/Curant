#Service file to create functions to help the sandbox env 
import os
import git 
import pathlib
from urllib.parse import urlparse, urlunparse


#Creating function to read in a repo string -> check files
def read_repo(repo_object, repo_str:str, workdir, checkout_branch: str | None = None):
    """
    Function reads reads in a repo string checks files through repo 
    should check if there is a docker file for the the agent -> to be used 
    """
    #Creating the github repo as well cloning the git from the repo url  
    clone_kwargs = {}
    if checkout_branch:
        clone_kwargs["branch"] = checkout_branch

    try:
        repo_object = git.Repo.clone_from(repo_str, workdir, **clone_kwargs) #cloning repo in temp dir 
    except git.GitCommandError as clone_error:
        #If repo requires authentication attempting token injection for private repos
        authed_url = _inject_token(repo_str)
        if authed_url == repo_str:
            raise
        repo_object = git.Repo.clone_from(authed_url, workdir, **clone_kwargs)

    #Getting the latest working commit from the tree 
    tree = repo_object.head.commit.tree #latest commit tree

    #Passing the tree to retrieve the dockerfile 
    docker_path = retrieve_dockerfile(tree=tree)

    #Edge case to check if the docker file is empty -> returning none from service function
    if not docker_path:
        return (None, repo_object)

    absolute_dockerfile = pathlib.Path(workdir) / docker_path
    
    return (absolute_dockerfile, repo_object) #returns path to the docker file and the object for the repo


#Creating recursive function to iterate through the tree 
def retrieve_dockerfile(tree, dockerfile='Dockerfile'):
    #iterates the working directory 
    files_and_dir = [(entry, entry.name, entry.type) for entry in tree] #returns the tupele of the name of entry and type
    #edge case to check if the files and dir list exist 
    if not files_and_dir:
        return None #Returns from the function if the directory doesnt exist 
    
    #Iterating through the list -> tuple of entry and entry name in list 
    for root in files_and_dir:
        #Unpacking elements through from tuples in the list 
        entry, dir_name, dir_type = root
        #if all unpacked items doesnt exist skip iteration
        if not all([dir_name, entry, dir_type]):
            continue #continues to next iterable in the tuple 

        #Checking the directory type 
        #if dir type is a blob checking the file 
        if dir_type == 'blob':
            #checking the file in directory to see if matche with dockerfile
            if dir_name == dockerfile:
                return entry.path #returns path to the docker file
                
        #Iterating through the files in the directory 
        if dir_type == "tree":
            #Recursing through the tree
            located_path = retrieve_dockerfile(entry, dockerfile)
            #Checks if the located path exists 
            if located_path is not None:
                #if path exists 
                return located_path #returns the path to the docker file 
    
            
    #If no docker file exists -> return None 
    return None #docker file doesnt exist in repo


#Helper to inject github token from env for private repos
def _inject_token(repo_url: str) -> str:
    token = os.getenv("AGENT_GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN")
    if not token or not repo_url.startswith("https://"):
        return repo_url
    parsed = urlparse(repo_url)
    if not parsed.netloc or parsed.username:
        return repo_url
    authed = parsed._replace(netloc=f"{token}@{parsed.netloc}")
    return urlunparse(authed)

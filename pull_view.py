#!/usr/bin/env python
import os
import shutil
import subprocess as sp


def remove_read_only(func, path, exc_info):
    """Sometimes, Windows complains when removing .git folders"""
    import stat
    if not os.access(path, os.W_OK):
        # Is the error an access error ?
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise exc_info


def read_repo_list():
    """Return a list of lists: [dir, url, ref]"""
    repos = []
    with open("repos.list") as f:
        lines = f.readlines()
    for line in lines:
        repos.append(line.split())
    return repos


view_dir = os.path.dirname(os.getcwd())
repos = read_repo_list()
for name, url, ref in repos:
    print("\nUpdating " + name + "...")
    repo_dir = os.path.join(view_dir, name)
    
    if os.path.isdir(repo_dir):
        # check whether it has the correct URL
        old_url = sp.check_output("git config --get remote.origin.url".split(), cwd=repo_dir).strip()
        if url == old_url:
            print("URL hasn't changed")
        else:
            print("URL for " + name + " has changed.")
            print("    Old: " + old_url)
            print("    New: " + url)
            print("Deleting repository.")
            shutil.rmtree(repo_dir, onerror=remove_read_only)

    # Clone
    if not os.path.isdir(repo_dir):
        sp.check_call('git clone {} {}'.format(url, name).split(), cwd=view_dir, stdout=sp.PIPE, stderr=sp.PIPE)
    
    # Fetch
    print("Fetching...")
    sp.check_call("git fetch".split(), cwd=repo_dir, stdout=sp.PIPE, stderr=sp.PIPE)

    # Checkout
    print("Checking out " + ref + "...")
    sp.check_call("git checkout {}".format(ref).split(), cwd=repo_dir, stdout=sp.PIPE, stderr=sp.PIPE)

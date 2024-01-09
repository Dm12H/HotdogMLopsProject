import subprocess


def current_commit_id():
    pipe = subprocess.Popen(
        ['git', 'rev-parse', "HEAD"],
        stdout=subprocess.PIPE)
    out, _ = pipe.communicate()
    out_str = out.decode().strip()
    return out_str

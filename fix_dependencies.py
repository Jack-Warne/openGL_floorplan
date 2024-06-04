import subprocess

def install(package):
    subprocess.check_call(["pip", "install", package])

# Install imutils and progressbar for cvlib
install("imutils")
install("progressbar")

# Install requests_mock for conda-repo-cli
install("requests_mock")

# Install FuzzyTM for gensim
install("FuzzyTM>=0.4.0")

# Uninstall incompatible tensorflow-intel and install the correct version for tensorflow-cpu
subprocess.check_call(["pip", "uninstall", "-y", "tensorflow-intel"])
install("tensorflow-intel==2.16.1")

# Install the correct version of clyent for conda-repo-cli
install("clyent==1.2.1")

# Install the correct version of requests for conda-repo-cli
install("requests==2.31.0")

# Install the correct version of docutils for sphinx
install("docutils<0.19,>=0.14")

print("All dependencies resolved.")

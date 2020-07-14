# HOWTO: Update XMOS TensorFlow Repository Fork

The XMOS fork of the TensorFlow repository is located here:

    https://github.com/xmos/tensorflow/

Updating consists of a few concepts:

1. Keeping the `master` branch in-sync with the upstream `master` branch
2. Merging the `master` branch into `develop`
3. Building the `xcore` target port.
4. Running all unit tests

# Syncing with the upstream master branch

Clone the XMOS fork (if necessary)

    > git clone git@github.com:xmos/tensorflow.git

Add upstream remote (if necessary)

    > git remote add upstream git@github.com:tensorflow/tensorflow.git

Fetch upstream changes

    > git fetch upstream

Checkout the local `master` branch (if necessary)

    > git checkout master

Merge upstream changes

    > git merge upstream/master
    > git push

# Merging the master branch into develop

While it is possible to do this locally, it is best done via a pull request.  Visit https://github.com/xmos/tensorflow/pulls to create a new PR.

To fetch and checkout the PR

    > git fetch origin pull/ID/head:BRANCHNAME

where `ID` is the pull request id and `BRANCHNAME` is the name of the new branch that you want to create. Then,

    > git checkout BRANCHNAME

# Build & Test

To build and run the unit tests.  **Note**, this takes a long time.

    > make -f tensorflow/lite/micro/tools/make/Makefile TARGET="xcore" test

======================
De-noising model
======================

Installation
============

1. **Install Dependencies**:

.. code-block:: shell

   pip install -r requirements.txt

2. **Download Dataset**:

.. code-block:: shell

   bash download.sh

This script will download a sample of the `MS-SNSD` dataset for noise samples, and a sample of the `DNS-Challenge` dataset for clean speech. Samples are saved in the `data/` directory.

3. **Convert Dataset**:

.. code-block:: shell

   python dataset.py

Convert the downloaded datasets into training samples, and save them as `TFRecords` in `data/records/`.

Training
========

1. **Initiate Training**:

.. code-block:: shell

   python train.py

This script will train a de-noising and de-reverberation model on the prepared data, and save it as `model.h5`.

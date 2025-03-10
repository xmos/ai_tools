App Single Model on One Tile and DDR
====================================


This example demonstrates how to run a MobileNetV2 model using DDR on tile[0].
This example runs on the XCORE.AI EVALUATION KIT XK-EVK-XU316.

Setup
-----

1. Ensure you have XTC tools version 15.3.0 activated in your current terminal.
2. Install the `xmos_ai_tools` Python package in your virtual environment (venv).

Build and Run
-------------

Run the following commands in the current directory.

.. code-block:: console

    # build
    cmake -G "Unix Makefiles" -B build
    xmake -C build
    # run
    xrun --xscope bin/app_ddr.xe


Output
------

The terminal should oyutput the three top classes given that image. 
It outputs soemthing like the following:

.. code-block:: console

    Init Mobilenet DDR model
    Input size = 76800
    Output size = 1000
    Top 3 classes:
    Top 0, class:291, prob:0.86, label:'lion, king of beasts, Panthera leo'
    Top 1, class:260, prob:0.02, label:'chow, chow chow'
    Top 2, class:200, prob:0.00, label:'Tibetan terrier, chrysanthemum dog'

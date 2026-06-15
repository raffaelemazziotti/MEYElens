MEYElens Python API
====================

MEYElens 2.0 provides PyTorch pupil and eye segmentation, camera acquisition,
recording, gaze calibration, signal analysis, and offline batch processing.

Install a hardware-appropriate PyTorch 2.2+ build first, then install MEYElens:

.. code-block:: console

   pip install meyelens

MEYElens intentionally leaves PyTorch out of its package dependencies so users
can select CPU, NVIDIA CUDA, or Apple MPS support.

Launch the offline GUI with:

.. code-block:: console

   meyelens-gui

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/modules


scPoli Reproducibility Repo
=========================================================================
Repository for scripts and notebooks used for scPoli paper. All the datasets used in the work are public.
Data for the PBMC atlas can be downloaded `here <https://figshare.com/projects/scPoli_data/155018>`_.

This code is not maintained and the core code has been moved and integrate in the `scArches <https://github.com/theislab/scarches/>`_ package. A tutorial for scPoli usage is available `here <https://scarches.readthedocs.io/en/latest/scpoli_surgery_pipeline.html>`_.

Usage and installation
-------------------------------
ScVI & ScANVI:

.. code-block:: bash

   pip install scvi-tools

Scib:

.. code-block:: bash

   pip install git+https://github.com/theislab/scib

Mars
  Since the model is not packaged, we had to clone the repository in this one.
  
  
Seurat
  Install R (4.1.0)
  Install Seurat (4.0.3)
  Install SeuratDisk (0.0.0.9019)

scPoli
  Please use the `scPoli_legacy <https://github.com/theislab/scPoli_legacy>`_ repository for reproducibility. That code is not maintained. For any other use refer to `scArches <https://github.com/theislab/scarches/>`_.

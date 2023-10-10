.. _install: 

Installation
============

The below will help you quickly install meshoid.

Requirements
------------

You will need a working Python 3.x installation; we recommend installing `Anaconda <https://www.anaconda.com/download/>`_ Python version 3.x.
You will also need to install the following packages:

    * numpy

    * numba

    * scipy

Installing the latest stable release
------------------------------------

Install the latest stable release with

.. code-block:: bash

    pip install meshoid

This is the preferred way to install meshoid as it will
automatically install the necessary requirements and put meshoid
into your :code:`${PYTHONPATH}` environment variable so you can 
import it.

Install from source
-------------------

Alternatively, you can install the latest version directly from the most up-to-date version
of the source-code by cloning/forking the GitHub repository 

.. code-block:: bash

    git clone https://github.com/mikegrudic/meshoid.git


Once you have the source, you can build meshoid (and add it to your environment)
by executing

.. code-block:: bash

    python setup.py install

or

.. code-block:: bash

    pip install -e .

in the top level directory. The required Python packages will automatically be 
installed as well.

You can test your installation by looking for the meshoid 
executable built by the installation

.. code-block:: bash

    which meshoid

and by importing the meshoid Python frontend in Python

.. code-block:: python

    import meshoid

Overview
========

The :py:mod:`pactolus` python package implements efficient methods for computational identification of metabolites
based on the scoring of measured fragmentation spectra against a collection of molecular fragmentation trees.

Generating fragmentation trees
------------------------------

To achieve computational efficiency, we pre-compute fragmentation in parallel via :py:mod:`pactolus.generate_frag_dag`.
We store fragmentation trees in HDF5 files, enabling efficient sharing and reuse of fragmentation trees. This
approach also enables the flexible extension to select and add molecules of interest for scoring, simply by
selecting/adding fragmetation tree files.

Scoring spectra against trees
-----------------------------

:py:mod:`pactolus.score_frag_dag` is then used to score spectra/scans against a collection of molecular fragmentation
trees. Some typical uses of this module are as follows:

Scoring many spectra against many trees
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    1) **Define the list of fragmentation trees**: Call \
       :py:func:`pactolus.score_frag_dag.make_file_lookup_table_by_MS1_mass` to creates a \
       sorted table listing containing the **i)** .h5 file paths and **ii)** parent MS1 mass for input \
       trees from :py:mod:`pactolus.score_frag_dag`. We may save the list as a ``.npy`` numpy file for later reuse.
    2) **Score the spectra**:

        a) The function :py:func:`pactolus.score_frag_dag.score_scan_list_against_trees` \
           then allows us to score an arbitrary list of spectra against our list of trees.
        b) To score a full ``(x,y,m/z)`` cube of fragmentation spectra with a common ``m/z`` axis, we can use \
           :py:func:`pactolus.score_frag_dag.score_peakcube_against_trees`. This is useful, e.g., in this case \
           of MS2 imaging data, where we acquire the fragmentation spectra for the same precursor ``m/z`` at every \
           image location (pixel).
    3) **Compute the hit table**: To calculate a list of summary hit-tables (one table per spectrum) we then call \
       :py:func:`pactolus.score_frag_dag.make_pactolus_hit_table`. A hit table is a structured numpy array \
       that summarizes all scores for a single spectrum. Each entry in the hit-table describes the \
       `score`, `id`, `name`,  `mass`, `n_peaks`, `n_match` for a given fragmentation tree (i.e., molecule) \
       against the given spectrum (see also :py:data:`pactolus.score_frag_dag.HIT_TABLE_DTYPE` ). \
       :py:func:`pactolus.score_frag_dag.make_pactolus_hit_table` uses :py:func:`pactolus.score_frag_dag.crossref_to_db` \
       to cross-reference pactolus fragmentation trees (which are identified by inchi-keys) agains the molecular
       data base file to retrieve real molecule names. An example molecular database file is available, e.g,
       from `http://midas-omicsbio.ornl.gov/MetaCyc.mdb <http://midas-omicsbio.ornl.gov/MetaCyc.mdb>`_. The
       main important parts in this case is the second column with the molecule name and the third column with
       the inchi key.

Scoring a single spectrum against a single tree
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function :py:func:`pactolus.score_frag_dag.calculate_MIDAS_score` is used to score a single spectrum against a single tree.

Computing the matched atoms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:func:`pactolus.score_frag_dagcalculate_MIDAS_score.py` function can optionally also return the extra
matching matrix describing which peaks of the spectrum where found as which fragment (node) in the fragmentation tree.
Using the match matrix we can then locate the atoms that were matched as follows:

    1) Open the HDF5 file with the fragmentation tree and retrieve the tree array
    2) Look up the fragment in the tree---each fragment is an index in the structured numpy array using 0-based indexing
    3) Retrieve the ``atom_bool_arr`` (or first column) of the fragment entry. The ``atom_bool_arr`` is an array of \
       boolean values describing for each atim of the whole molecule whether it is part of the fragment.

Scoring spectra against spectra
-------------------------------

For many analyses it is useful to compare specta based on their L1 or L2 norm.
In practice, however, mass spectra are commonly centroided---i.e., each peak is describe by a single (``m/z``, ``intensity``)
pair---and Pactolus assumes centroided spectra for scoring. Computing the distance between centroided spectra is
complicated since the m/z values of matching peaks are often slightly shifted between spectra. The Pactolus module
:py:mod:`pactolus.score_sepctra` provides a series of functions to compute the fuzzy distance between centroided spectra,
while accounting for mass tolerance (i.e, shifts in ``m/z``) and noise (via ``intensity`` thresholds).


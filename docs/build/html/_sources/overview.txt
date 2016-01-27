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

:py:mod:`pactolus.score_frag_dag` is used to score spectra/scans against a collection of molecular fragmentation
trees.


Scoring spectra against many trees: Command Line Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:mod:`pactolus.score_frag_dag` provides a convenient command line interface that allows to easily score
spectra/scans against spectra. As file-based input we here usually require:

1) An HDF5 file with the data of all spectra provided via the ``--input`` parameter. The data format is described at :ref:`spectrum_data_format`.
2) A tree lookup file (see :ref:`_tree_file_lookup_data_format`) in either numpy or HDF5 format provided via the ``--trees`` parameter. The tree-lookup table lists all trees and their respective MS1 mass sorted by mass. The tree-lookup table is used to locate trees during the scoring. Alternatively the ``--trees`` parameter accepts also the path to a directory with all tree files to be used or a text-file listing all trees to be used (one tree per line), in which case the tree lookup table will be construcuted based on the provided information.
3) Fragmentation trees listed in the tree lookup table must be stored in the format described in :ref:`frag_tree_data_format`.

The results are typically stored in an HDF5 file indicated by the ``--save`` command line option. However, the results in ``--save`` are collected at the end, after all scores have been computed. While scores are being computed, the results are stored in temporary files. The temporary files are be default automatically generated and cleaned up. However, a user may also set the directory were temporary files should be stored via ``tempdir`` parameter. This is useful in case that we anticipate that the scoring may not complete (e.g., due to limited compute resources) and we want to be able to look at the data that has been generated so far.

An example call for the tool might look as follow:

.. code-block:: python

    python pactolus/score_frag_dag.py
        --input "local_data/my_file.h5:/my_subgroup"
        --neutralizations "[-1.00727646677,-2.0151015067699998,0.00054857990946]"  # Don't forget the quotes
        --trees=local_data/trees
        --ms1_mass_tolerance 0.025
        --ms2_mass_tolerance 0.025
        --max_depth 5
        --save local_data/test_out.h5
        --tempdir local_data/temp
        --match_matrix True
        --loglevel DEBUG

Detailed infomation about the command line options is available via the ``--help`` command line option. For example:

.. code-block:: python

    python pactolus/score_frag_dag.py --help

        usage: score_frag_dag.py [-h] --input INPUT [--save SAVE]
                                 [--precursor_mz PRECURSOR_MZ] --trees TREES
                                 --ms1_mass_tolerance MS1_MASS_TOLERANCE
                                 --ms2_mass_tolerance MS2_MASS_TOLERANCE --max_depth
                                 {0,1,2,3,4,5,6,7,8,9} --neutralizations
                                 NEUTRALIZATIONS
                                 [--schedule {DYNAMIC,STATIC_1D,STATIC}]
                                 [--collect COLLECT] [--pass_scanmeta PASS_SCANMETA]
                                 [--pass_scans PASS_SCANS]
                                 [--pass_compound_meta PASS_COMPOUND_META]
                                 [--metabolite_database METABOLITE_DATABASE]
                                 [--match_matrix MATCH_MATRIX]
                                 [--loglevel {INFO,WARNING,CRITICAL,ERROR,DEBUG,NOTSET}]
                                 [--tempdir TEMPDIR] [--clean_tempdir CLEAN_TEMPDIR]
                                 [--clean_output CLEAN_OUTPUT]

        score scan list against trees:

        optional arguments:
          -h, --help            show this help message and exit

        required analysis arguments:
          --input INPUT         Path to the HDF5 file with the input scan data
                                consisting of the<filenpath>:<grouppath> were
                                <filepath> is the path to the file and<grouppath> is
                                the path to the group within the file. E.g. a valid
                                definition may look like:
                                'test_brain_convert.h5:/entry_0/data_0. See below for
                                details on how to store the data in HDF5. (default:
                                None)
          --trees TREES         1) Path to the sub-directory with all relevent .h5
                                pactolus fragmentation trees. or 2) path to a text-
                                file with a list of names of the tree files, 3) Path
                                to the.npy file with the pactolus file lookup table,
                                or 4) Path to corresponding dataset in an HDF5 file
                                where the path is given by <filename>:<dataset_path>.
                                (default: None)
          --ms1_mass_tolerance MS1_MASS_TOLERANCE
                                float, max. diff. in Da of tree_parent_mass &
                                MS1_precursor_mass (default: 0.01)
          --ms2_mass_tolerance MS2_MASS_TOLERANCE
                                float, max. mass in Da by which two MS2/MSn peaks can
                                differ (default: 0.01)
          --max_depth {0,1,2,3,4,5,6,7,8,9}
                                Maximum depth of fragmentation pathways (default: 5)
          --neutralizations NEUTRALIZATIONS
                                List of floats, adjustments (in Da) added to data
                                peaks in order to neutralize them. Here we can use
                                standard python syntax, e.g, '[1,2,3,4]' or '[[1, 3],
                                [4, 5]]'. Remember to include the array string in
                                quotes (default: [0, 1, 2])

        optional analysis arguments:
          --save SAVE           Path to the HDF5 file where the output should be saved
                                consisting of the<filenpath>:<grouppath> were
                                <filepath> is the path to the file and<grouppath> is
                                the path to the group within the file. E.g. a valid
                                definition may look like:
                                'test_brain_convert.h5:/entry_0/data_0. See below for
                                details on how to store the data in HDF5. (default: )
          --precursor_mz PRECURSOR_MZ
                                Floating point precursor mass over charge value.
                                Default value is -1, indicating that the precursor m/z
                                should be read from file. (default: -1)
          --pass_scanmeta PASS_SCANMETA
                                Pass per-scan metadata through and include it in the
                                final output (default: True)
          --pass_scans PASS_SCANS
                                Pass the scan/spectrum data through and include it in
                                the final output (default: True)
          --pass_compound_meta PASS_COMPOUND_META
                                Compile compound metadata from the tree file and pass
                                it through to the output. (default: True)
          --metabolite_database METABOLITE_DATABASE
                                The database of metabolites from which the trees were
                                generated.Needed only if compound metadata from the
                                database should be includedin the output. (default: )
          --match_matrix MATCH_MATRIX
                                Track and record the match matrix data. Required if
                                the num_matched and match_matrix_s_c data should be
                                included in the output. (default: True)
          --loglevel {INFO,WARNING,CRITICAL,ERROR,DEBUG,NOTSET}
                                Define the level of logging to be used. (default:
                                INFO)
          --tempdir TEMPDIR     Directory where temporary data files should be stored.
                                Temporary files are created one-per-core to
                                incrementally save the results of an analysis. If not
                                given, then a tempfileswill be created automatically
                                if needed (i.e., if --save is set).Temporary files
                                will removed after completion if --save is set.
                                (default: )
          --clean_tempdir CLEAN_TEMPDIR
                                Boolean indicating whether we should automatically
                                delete conflicting data in the tempdir. (default:
                                False)
          --clean_output CLEAN_OUTPUT
                                Boolean indicating whether we should automatically
                                delete conflicting data in the output target defined
                                by --save (default: False)

        parallel execution arguments:
          --schedule {DYNAMIC,STATIC_1D,STATIC}
                                Scheduling to be used for parallel MPI runs (default:
                                DYNAMIC)
          --collect COLLECT     Collect results to the MPI root rank when running in
                                parallel (default: True)

        HDF5 input data format:
        -----------------------
        The data within the group should be stored as follows:
           (1) `peak_mz` : 1D array with all m/z values for all concatenated scans.
           (2) `peak_value` : 1D array with all intensity values for all concatenated
               scans. Must have the same length as peak_mz.
           (3) `peak_array_index` : 1D (or n-D array) where the first dimension must
                be the scan index and the last dimension  (in the case
                of n-D arrays) must contain the integer start offset where
                each scan is located in the peak_mz and peak_value arrays.
                An n-D array is sometimes used to store additional location
                That additional data will be ignored.
          (4) `mz1_mz` or `precursor_mz1` : 1D array with the MS1 precursor m/z value
              for each scan. Must be #scans long. May also be part of scan_metadata.
          (5) `scan_metadata` : Group with additional arrays for per-scan
              metadata that should be passed through. The first
              dimension of the arrays should always have the same
              length as the number of scans.
          (6) `experiment_metadata` : Group with additional arbitrary metadata
              pertaining to the experiment. This data is pass through as is.

         This command-line tool is broad to you by Pactolus. (LBNL)


Scoring many spectra against many trees
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:py:func:`pactolus.score_frag_dag.main` provides a detailed overview of the main steps. Here a rough overview of what we need to do:

    1) **Define the list of fragmentation trees**: Use :py:func:`pactolus.score_frag_dag.make_file_lookup_table_by_MS1_mass` or :py:func:`pactolus.score_frag_dag.load_file_lookup_table` to creates a sorted table containing the **i)** .h5 file paths and **ii)** parent MS1 mass for input trees from :py:mod:`pactolus.score_frag_dag`. We may save the list as a ``.npy`` numpy file for later reuse.
    2) **Define the list of scans/spectra** Here we typically load the spectrum data from file via py:func:`pactolus.score_frag_dag.load_scan_data_hdf5`. Alternatively we can also define our own ``scan_list``, which is a list of numpy ndarrays where each array has a shape of ``(num_peaks, 2)`` and the first column ([:,0]) are the m/z values and the second columd ([:,1]) are the intensity values.
    3) **Determine the precursor m/z (i.e., ms1_mz) for the scans** This is a 1D float numpy array with the precursore m/z for each spectrum. In a typical use this is included with the scan data loaded from the HDF5 file and can be retrieved from the ``scan_metadata`` returned by py:func:`pactolus.score_frag_dag.load_scan_data_hdf5` via ``scan_metadata['ms1_mz']``
    4) **Score the spectra**:  The function :py:func:`pactolus.score_frag_dag.score_scan_list_against_trees` then allows us to score an arbitrary list of spectra against our list of trees. The function supports parallel scoring via MPI. (:py:func:`pactolus.score_frag_dag.score_peakcube_against_trees` might also be useful, although less tested).

Once we have compute the base output we may want to further process the resutls via :py:func:`pactolus.score_frag_dag.make_pactolus_hit_table` or collect and save results via :py:func:`pactolus.score_frag_dag.collect_score_scan_list_results`.

Scoring a single spectrum against a single tree
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The function :py:func:`pactolus.score_frag_dag.calculate_MIDAS_score` is used to score a single spectrum against a single tree.

Computing the matched atoms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The :py:func:`pactolus.score_frag_dag.calculate_MIDAS_score.py` function can optionally also return the extra
matching matrix describing which peaks of the spectrum where found as which fragment (node) in the fragmentation tree.
Using the match matrix we can then locate the atoms that were matched as follows:

    1) Open the HDF5 file with the fragmentation tree and retrieve the tree array
    2) Look up the fragment in the tree---each fragment is an index in the structured numpy array using 0-based indexing
    3) Retrieve the ``atom_bool_arr`` (or first column) of the fragment entry. The ``atom_bool_arr`` is an array of \
       boolean values describing for each atom of the whole molecule whether it is part of the fragment.

Scoring spectra against spectra
-------------------------------

For many analyses it is useful to compare specta based on their L1 or L2 norm.
In practice, however, mass spectra are commonly centroided---i.e., each peak is describe by a single (``m/z``, ``intensity``)
pair---and Pactolus assumes centroided spectra for scoring. Computing the distance between centroided spectra is
complicated since the m/z values of matching peaks are often slightly shifted between spectra. The Pactolus module
:py:mod:`pactolus.score_sepctra` provides a series of functions to compute the fuzzy distance between centroided spectra,
while accounting for mass tolerance (i.e, shifts in ``m/z``) and noise (via ``intensity`` thresholds).


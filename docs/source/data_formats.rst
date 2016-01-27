Data Formats
============

Scan/Spectrum data format (score_frag_dag)
-----------------------------------------

The input scans are usually stored in HDF5 using the following basic data layout. The data may be stored in an
arbitrary group within the HDF5 and should contain the following groups and datasets:

    * ``peak_mz`` : 1D float array with all m/z values for all concatenated spectra.
    * ``peak_value`` : 1D float array with all intensity values for all concatenated spectra. Must have the same length as peak_mz.
    * ``peak_arrayindex`` : 1D (or n-D array) integer array where first dimension is the spectrum index and the last dimension (in the case of n-D arrays) contains the integer start offset where each spectrum is located in the peak_mz and peak_value  arrays. An n-D array is sometimes used to store additional location data (e.g., the x/y location from which a spectrum is recorded). That additional data will be ignored.
    * ``ms1_mz`` or ``precursor_mz1`` : Optional 1D array with the MS1 precursor m/z value for each spectrum. Must have the same length as the number of spectra (i.e, the length of the peak_array_index).Alternatively the ms1_mz dataset may also be stored in the spectrum_metadata group.
    * ``scan_metadata/`` : Group with additional arrays for per-spectrum metadata that should be passed through. The first dimension of the arrays should always have the same lenght as the number of spectra.
    * ``experiment_metadata/`` : Group with additional arbitrary metadata pertaining to the experiment. This data will also be pass through as is.

Scoring temporary output data format (score_frag_dag)
-----------------------------------------------------

When scoring spectra using :py:mod:`pactolus.score_frag_dag` we can optionally incrementally write the scoring results
to temporary files. The resulting output files have the following structure:


    * For each spectrum a new group ``spectrum_#s`` is created where ``#s`` is the spectrum index. Within this group,
      the following datasets are created:

         * ``score_matrix`` : The 2D score matrix with all scores. This matrix has a shape (n_scans, len(file_lookup_table)).
         * ``match_matrix_#s_#c`` where #s is the spectrum index and #c is the compound index. Each of these datasets contains the match matrix for the corresponding spectrum / compound combination. A match matrix is a 2D bool array with a shape of ``(n_peaks, n_nodes)`` where ``n_peaks` is the number of peaks in the spectrum and ``n_nodes`` is the number of nodes in the tree.

When running in parallel, one temporary output file will be generated per MPI rank (i.e, compute core).


Scoring main output data format (score_frag_dag)
------------------------------------------------

The main ouput data format is similar to the temporary output data format, but consolidates all results in a more
compact structure to describe results across a collection of spectra scored against the same set of compounds. The
output results may be stored in an arbitray user-defined group which will contain the following datasets and groups.

    * ``score_matrix`` : The 2D score matrix with all scores. This matrix has a shape (n_scans, len(file_lookup_table)) and contains floating point numbers.
    * ``score_rank_matrix`` : 2D integer matrix containing the ranking of the scores for each scan. -1 is used for scores that were not ranked because they had a value of 0 (i.e., scores that were not computed)
    * ``match_matrix_#s_#c`` where #s is the spectrum index and #c is the compound index. Each of these datasets contains the match matrix for the corresponding spectrum / compound combination. A match matrix is a 2D bool array with a shape of ``(n_peaks, n_nodes)`` where ``n_peaks`` is the number of peaks in the spectrum and ``n_nodes`` is the number of nodes in the tree. The match matrix datasets are optional.
    * ``tree_file_lookup_table`` : 1D compound dataset with the lookup table used to define the tree-files used for scoring. The dtype is defined in :py:mod:`pactolus.score_frag_dag.FILE_LOOKUP_TABLE_DTYPE` .
    * ``num_matched`` Optional dataset describing the number of peaks matched as part of a given score. If available this is a 2D integer matrix of the same shape as ``score_matrix``. Only available if the match matrix data is tracked.
    * ``scan_metadata/`` : Group with additional, optional per-spectrum metadata arrays. This group may contain arbitrary user-defined per-spectrum metadata. Here we usually assume that we have arrays where the first dimension matches the length and ordering of the scans that were scored. Usually we here add the array ``num_peaks`` indicating the number of peaks for each spectrum to help with the evaluation of the score even if the original scan data may not be easily accesible.
    * ``experiment_metadata/`` : Group with additional, optional general metadata about the experiment. This group may contain arbitrary user metadata about the experiment.
    * ``compound_metadata/`` : Group with additional metadata about the compounds. This typically includes the fields defined by the py:mod:`pactolus.score_frag_dag.METACYC_DTYPE`` ,e.g., the ``id``, ``name``, ``inchi``, ``lins``, ``inchi_key``, ``mass``.
    * ``scans/`` : Optional group with the actual scan data stored using the scan data format described above.


Tree file Lookup table data format (score_frag_dag)
---------------------------------------------------

This is usually a binary numpu ``.npy` file with a 1D array with the dtype defined in :py:mod:`pactolus.score_frag_dag.FILE_LOOKUP_TABLE_DTYPE` defining for each molecular fragmentation tree: i) the path to the HDF5 tree file and ii) the primary mass of the corresponding molecule, which is used to search for trees with a matching precusor mz. The array may also be stored in an HDF5 file in a dataset with a corresponding compound dtype.


Fragementation tree data format (generate_frag_dag)
---------------------------------------------------

Fragmentation trees are stored in HDF5 in a group where the group name is the inchi_key. Each tree-group
contains the following datasets:

    * ``FragTree_at_max_depth=#`` is the fragmentation tree dataset. The fragmentation tree is stored as a  1D compound dataset listing all fragments sorted by their mass. The dataset contains the fragments for the molecule up to the indicated fragmentation depth, i.e., we break at most ``max_depth`` bonds to generate a fragment. Each fragment in the tree is unique in that it appears only once in the fragmentation tree and we store only the shortest bond breakage path that leads to the generation of the fragment. For each fragment, the compound data type stores the following information:

        * ``atom_bool_arr`` is a bool vector consisting of ``#atoms`` values describing which atoms of the fragmented molecule are part of the fragment.
        * ``bond_bool_arr`` is a bool vector consisting of ``#bonds`` values describing the shortest bond breakage path giving rise to the fragment, i.e., which fragments do we need to break to create the fragment.
        * ``mass_vec`` is a 64bit floating point number with the mass of the fragment
        * ``parent_vec`` is a 64bit integer indicating the index of parent fragment (using 0-based indexing) in the fragmention tree.

    * In addition, the following information is stored as attributes on the group:

        * ``inchi`` : Inchi string for the molecule
        * ``num_atoms`` : Number of atoms in the molecule
        * ``num_bonds`` : Number of bonds in the molecule
        * ``num_fragments`` : Number of fragments stored in the tree
        * ``max_depth`` : The maximum fragmentation depth
        * ``time_to_build`` : The time in seconds used to build the tree.

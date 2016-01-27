#!python
"""
Score spectra/scans against a collection of molecular fragmentation trees.

"""
# TODO Update tree files to include molecule name and other metadata from the original molecular database
# TODO Update crossref_to_db to allow look-up of information from tree files
# TODO Update crossref_to_db to have a safe fallback solution when the database file is missing and the metadata is also not available in the trees
# TODO score_peakcube_against_trees test support for 'want_match_matrix' parameter
# TODO score_scan_list_against_trees test support for 'want_match_matrix' parameter
# TODO score_scan_list_against_trees and score_peakcube_against_trees allow return of a sparse matrix?
# TODO Test the workflow with the retrieval of the match matrix and allow storage of the match matrix in BASTet
# TODO In the new setup we compile the ouput as indipendent arrays (rather than as a HIT_TABLE). Should we remove the HIT_TABLE functionality?
# TODO Test that everything is working
# TODO Add description of the HDF5 format used to store fragmentation trees to the documentation

# FIXED: max_depth parameter was not passed to the scoring function in neither score_scan_list_against_trees nor score_peak_cube_against_trees
# FIXED: Updated score_scan_list_against_trees nor score_peak_cube_against_trees functions to replace the params dict with explicit input parameters
# FIXED: Allow return of match matrix
# FIXED: Allow metadata to passed through the main function
# FIXED: The created hittable conained always 0 for the number of peaks and number of matched peaks (because the match matrix was required but not returned and not set as input)
# FIXED:  calculate_MIDAS_score the calculation of the nodes to be used when max_depth was specified had a bug no.where[...] was an illegal expression
# FIXED: make_pactolus_hit_table did not fill the npeaks and nmatched endtires in the hit_table but left them as 0. If the match_matrices are available then we now fill them with the correct data values.

__authors__ = 'Curt R. Fischer, Oliver Ruebel, Benjamin P. Bowen'
__copyright__ = 'Lawrence Berkeley National Laboratory and Authors, 2015.  All rights currently reserved.'


# standard libraries
import re
import os
import sys
import time
from tempfile import NamedTemporaryFile

# numpy and scipy
import numpy as np
from scipy.stats import norm
from scipy.sparse import csc_matrix

# command line args
import argparse
try:
    from omsi.workflow.common import RawDescriptionDefaultHelpArgParseFormatter
except ImportError:
    from pactolus.third_party.argparse_helper import RawDescriptionDefaultHelpArgParseFormatter
try:
    from omsi.datastructures.analysis_data import data_dtypes
except ImportError:
    from pactolus.third_party.argparse_helper import data_dtypes

# parallel execution using mpi
try:
    import omsi.shared.mpi_helper as mpi_helper
except ImportError:
    import pactolus.third_party.mpi_helper as mpi_helper
# logging
try:
    from omsi.shared.log import log_helper
except ImportError:
    from pactolus.third_party.log import log_helper


# rdkit for finding masses and inchi keys
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt

# hdf5
import h5py

# global variables
PATH_TO_TEST_DATA = './test_data/'

HIT_TABLE_DTYPE = np.dtype({'names': ['score', 'id', 'name',  'mass', 'n_peaks', 'n_match'],
                            'formats': ['f4', 'a100', 'a100', 'f4',   'i4',       'i4']})
"""
Numpy data type (dtype) used for hit tables
"""

FILE_LOOKUP_TABLE_DTYPE = np.dtype([('filename', 'a400'),
                                    ('ms1_mass', 'f8'), ])
"""
Numpy data dtype (dtype) used for the tree file lookup table
"""

METACYC_DTYPE = np.dtype([('metacyc_id', 'a30'),
                          ('name', 'a100'),
                          ('inchi', 'a1000'),
                          ('lins', 'a400'),
                          ('inchi_key', 'a40'),
                          ('mass', float)])
"""
Numpy data type (dtype) used to store METACYC molecular databases
"""


def calculate_lambda(fragment_masses, data_masses, mass_tol):
    """
    Calculates lambda values as described in the MIDAS paper [dx.doi.org/10.1021/ac5014783]

    :param fragment_masses:    np.ndarray of floats w/ shape (n,), theoretical masses from frag tree, in Da
    :param data_masses:        np.ndarray of floats w/ shape (n,), (neutralized) masses in Da detected in MS2 scan
    :param mass_tol:           float, maximum allowed mass difference in Da between theoretical and detected peaks
    :return lambda:            np.ndarray of floats containing the lambda scores
    """
    epsilons = fragment_masses - data_masses
    return np.piecewise(epsilons,
                        condlist=np.abs(epsilons) <= mass_tol,
                        funclist=[lambda x: 2 * (1-norm.cdf(abs(x), scale=mass_tol/2)),
                                  0,
                                  ]
                        )


def bfs_plausibility_score(plaus_score, tree, matches, nodes=None,):
    """
    Modifies in place a numpy vector of plausibility scores for each fragment in a tree through breadth-first search.

    :param plaus_score:     numpy 1D vector with len same as tree, value is plausibility score
    :param tree:            numpy structured array output by FragTree.numpy_tree
    :param matches:         numpy 1D vector of ints; the rows of tree that match any data peaks
    :param nodes:           numpy array with 1 dimension; row indices of tree currently being scored
    """
    # if nodes is None, start at the root:
    if nodes is None:
        nodes = np.where(tree['parent_vec'] == -1)[0]
        plaus_score[nodes] = 1

    # find direct children of current nodes and if there are any, score them
    children = np.where(np.in1d(tree['parent_vec'], nodes))[0]
    if children.size:
        parents = tree['parent_vec'][children]
        # depth of frag i = num of bonds broken i.e. num of Trues in bond_bool_arr[i, :]
        depths = (tree[children]['bond_bool_arr']).sum(axis=1)
        parent_depths = (tree[parents]['bond_bool_arr']).sum(axis=1)
        b = depths - parent_depths

        base = np.select([np.in1d(parents, matches)], [0.5], default=0.1,)
        plaus_score[children] = np.power(base, b) * plaus_score[parents]

        # continue recursive bfs by looking for children of these children
        bfs_plausibility_score(plaus_score, tree, matches, nodes=children)

    else:
        return


def find_matching_fragments(data_masses, tree, mass_tol):
    """
    Find node sets in a tree whose mass is within mass_tol of a data_mz value

    :param data_masses: numpy 1D array, float, *neutralized* m/z values of data from an MS2 or MSn scan
    :param tree: numpy structured array as output by FragDag
    :param mass_tol: precursore m/z mass tolerance

    :return: matching_frag_sets, list of lists; len is same as data_mzs; sublists are idxs to rows of tree that match
    :return: unique_matching_frags, numpy 1d array, a flattened numpy version of unique idxs in matching_frag_sets
    """
    # start_idx is element for which inserting data_mz directly ahead of it maintains sort order
    start_idxs = np.searchsorted(tree['mass_vec'], data_masses-mass_tol)

    # end_idx is element for which inserting data_mz directly after it maintains sort order
    #  found by searching negative reversed list since np.searchshorted requires increasing order
    length = len(tree)
    end_idxs = length - np.searchsorted(-tree['mass_vec'][::-1], -(data_masses+mass_tol))

    # if the start and end idx is the same, the peak is too far away in mass from the data and will be empty
    matching_frag_sets = [range(start, end) for start, end in zip(start_idxs, end_idxs)]  # if range(start, end)]

    # flattening the list
    unique_matching_frags = np.unique(np.concatenate(matching_frag_sets))

    # Returning both the flat index array and the sets of arrays is good:
    #       matching_frag_sets makes maximizing the MIDAS score easy
    #       unique_matching_frags makes calculating the plausibility score easy
    return matching_frag_sets, unique_matching_frags


def normalize_intensities(mz_intensity_arr, order=1):
    """
    Normalizes intensities in a 2D numpy array of m/z values & intensities.
    Designed to be used non-neutralized data.

    :param mz_intensity_arr: numpy float with shape (num_peaks, 2)
    :param order: int, if 1 then L1 norm is returned; otherwise L2 norm is computed
    :return: out, a normalized version of mz_intensity_arr with intensities that sum to 1

    Note: this function assumes intensities can never be negative.
    """
    out = mz_intensity_arr
    norm_ = out[:, 1].sum() if order == 1 else np.linalg.norm(out[:, 1])
    out[:, 1] = out[:, 1] / norm_
    return out


def neutralize_mzs(data_mz_intensity_arr, neutralizations):
    """
    Converts data m/z values to possible data neutral mass values (in Da) by combinatorial subtraction from ionizations.

    :param data_mz_intensity_arr: numpy float with shape (num_peaks, 2), assumed normalized i.e. x[:, 1].sum() = 1
    :param neutralizations: masses: in Da of singly charged ionic species whose removal results in a neutral mass.
    :return: neutralized_mass_intensity_arr
    """
    num_peaks, _ = data_mz_intensity_arr.shape
    mzs, intensities = data_mz_intensity_arr[:, 0], data_mz_intensity_arr[:, 1]
    shifted_arrays = tuple(np.array([mzs+shift, intensities]).T for shift in neutralizations)
    return np.vstack(shifted_arrays)


def calculate_MIDAS_score(mz_intensity_arr, tree, mass_tol, neutralizations, max_depth=None, want_match_matrix=False):
    """
    Score the the plausibility that a given MS2 (or MSn) scan arose from a given compound.

    :param mz_intensity_arr:    ndarray w/ shape (n_peaks, 2).  m/z values in col 0, intensities in col 1
    :param tree:                numpy structured array for a frag. tree of a molecule, usually from FragTree.numpy_tree
    :param mass_tol:            maximum mass in Da by which two MS2 (or MSn) peaks can differ
    :param neutralizations:     list of floats, adjustments (in Da) added to data peaks in order to neutralize them
    :param max_depth:           int, truncates tree, keeping only nodes <= max_depth bond breakages away from root
    :param want_match_matrix:   bool, if True then tuple of (score, match_matrix) is returned, else return score only
    :return:                    score, float, a MIDAS score
    :return:                    match_matrix, a bool matrix with n_peaks columns and n_nodes rows.
                                              elements are True if given peak matches given node of frag_dag
    """

    # attempt to truncate tree if max_depth is supplied
    if max_depth:
        node_depths = tree['bond_bool_arr'].sum(axis=1)
        nodes_to_keep = np.where(node_depths <= max_depth)[0]
        tree = tree[nodes_to_keep]

    # normalize and neutralize data
    mz_rel_intensity_arr = normalize_intensities(mz_intensity_arr)
    mass_rel_intensity_arr = neutralize_mzs(mz_rel_intensity_arr, neutralizations)
    data_masses, data_rel_intensities = mass_rel_intensity_arr[:, 0], mass_rel_intensity_arr[:, 1]

    # find matching fragments
    matching_frag_sets, unique_frag_arr = find_matching_fragments(data_masses, tree, mass_tol)

    # if there are no matching fragments, the score is 0
    if unique_frag_arr.size == 0:
        return 0

    # initialize and modify in place the plaus_score array:
    plaus_score = np.zeros(len(tree))
    bfs_plausibility_score(plaus_score, tree, matches=unique_frag_arr)

    # construct match_matrix to matrix (csr format) where columns are data peaks, rows are tree nodes.
    #  matrix is constructed as sparse but converted to dense.
    #  slight speedup probably possible for full sparse implementation.
    inds = np.concatenate(matching_frag_sets)  # row indices
    indptr = np.append(0, np.cumsum(([len(el) for el in matching_frag_sets])))
    data = np.ones(shape=inds.shape, dtype=bool)
    match_matrix = csc_matrix((data, inds, indptr), dtype=bool).toarray()

    # loop over rows of match_matrix to calculate score
    score = 0
    rows_with_matches = np.where(match_matrix.any(axis=1))[0]
    for row_idx in rows_with_matches:
        col_idx = np.where(match_matrix[row_idx, :])[0]
        lambdas = calculate_lambda(tree[row_idx]['mass_vec'], data_masses[col_idx], mass_tol)
        subscore = (plaus_score[row_idx] * data_rel_intensities[col_idx] * lambdas).max()
        score += subscore

    if want_match_matrix:
        return score, match_matrix
    else:
        return score


def make_file_lookup_table_by_MS1_mass(tree_files=None, path=None, save_result=None):
    """
    Creates a sorted table listing of .h5 file paths and (i) parent MS1 mass for input trees from generate_frag_dag.

    :param tree_files:  list of filenames (must be full-path!) containing trees to use for scoring
    :param path:        path to subdirectory containing all relevant .h5 files
    :param save_result: if not None, the name of a file to which to save the numpy table as a
    :return:            tree_files_by_ms1_mass, a numpy structured array with columns (i) filename and (ii) MS1 mass.
                        The numpy dtype is type = [('filename', 'a400'),('ms1_mass', 'f8'), ]
    """
    global FILE_LOOKUP_TABLE_DTYPE
    # check arguments and decide to read from supplied path for from tree_file list
    if tree_files is None and path is None:
        raise ValueError('tree_files and path cannot both be None.')

    if tree_files is None:
        import glob
        tree_files = glob.glob(os.path.join(path, '*.h5'))

    # initialize result table
    num_files = len(tree_files)
    dtype = FILE_LOOKUP_TABLE_DTYPE
    tree_files_by_ms1_mass = np.zeros(shape=num_files, dtype=dtype)

    # loop over all files
    for idx, filename in enumerate(tree_files):
        # try to get MS1 mass from file
        try:
            file_reader = h5py.File(filename)
            group_key = file_reader.keys()[0]
            data_key = file_reader[group_key].keys()[0]
            ms1_mass = file_reader[group_key][data_key]['mass_vec'][-1]  # parent mass is at bottom of table
            tree_files_by_ms1_mass[idx] = (filename, ms1_mass)
        except (TypeError, ValueError, IOError):
            tree_files_by_ms1_mass[idx] = (filename, -1)  # a "score" when file IO problems occur is flagged as -1
            print "Warning: Could not read MS1 mass from h5 file %s" % filename

    # sort, save results if desired, and return
    order = np.argsort(tree_files_by_ms1_mass['ms1_mass'])
    if save_result:
        np.save(os.path.join(path, save_result), tree_files_by_ms1_mass[order])
    return tree_files_by_ms1_mass[order]


def score_peakcube_against_trees(peakcube,
                                 peakmzs,
                                 ms1_mz,
                                 file_lookup_table,
                                 neutralizations,
                                 ms1_mass_tol,
                                 ms2_mass_tol,
                                 max_depth,
                                 want_match_matrix=False):
    """
    Create a score cube of MIDAS scores for each

    :param peakcube:          numpy ndarray of floats   Stores peak intensities.  Has shape (nx, ..., n_peaks).
                                                        i.e., the last axis is a peak index, and prior axes are
                                                        an arbitrary number of spatial or time coordinates

    :param peakmzs:           numpy array of floats     Stores fragment peak m/z values.  Has shape (n_peaks,)

    :param ms1_mz:            float                     the (unneutralized) MS1 mass of the precursor ions.

    :param file_lookup_table: full path to a .npy file having a numpy structured array with columns
                              (i) filename and (ii) MS1 mass. Alternatively, this may also be the numpy array directly.

    :param ms1_mass_tol: float, max. diff. in Da of tree_parent_mass & MS1_precursor_mass

    :param ms2_mass_tol: float, max. mass in Da by which two MS2/MSn peaks can differ

    :param neutralizations:   list of floats, adjustments (in Da) added to data peaks in order to neutralize them

    :param max_depth:         int, optional.  For restricting scoring to lower max depths than
                              are present in the supplied tree

    :param want_match_matrix:   bool, if True then tuple of (score, match_matrix) is returned, else return score only.
                                Default value is False.

    :return: score_cube       a numpy ndarray of shape (nx, ..., len(file_lookup_table))
    :return: match_matrix     Optional output that is only returned if want_match_matrix is set to True.
                              List of lists of match matrices (one matrix per scan). Each entry is a bool matrix
                              with n_peaks columns and n_nodes rows. Elements are True if given peak matches given
                              node of frag_dag. An entry will be None in case that the hit-score was 0. The return
                              value will be None if want_match_matrix is set to False (Default)


    Unlike score_scan_list_against_trees, this function is designed to work on numpy arrays of scans.  It is more
     appropriate for imaging data where:
        1. an common MS1 precursor has been fragmented and scanned many times/places, as long as
        2. a global peak finder has been run on all scans, so peak intensities exist for every peak at every pixel
    """
    # load file_lookup_table if needed
    if isinstance(file_lookup_table, basestring):
        file_lookup_table = np.load(file_lookup_table)
    elif isinstance(file_lookup_table, np.ndarray):
        pass
    else:
        raise ValueError('Invalid file_lookup_table specified')

    # unravel peakcube into flat 2d array
    spatial_dimensions = peakcube.shape[:-1]
    n_coordinates = np.product(spatial_dimensions)
    n_peaks = peakcube.shape[-1]

    peaklist = peakcube.reshape(n_coordinates, n_peaks)
    n_compounds = len(file_lookup_table)

    # initialize score_cube
    score_cube = np.zeros(shape=(n_coordinates, n_compounds), dtype=float)

    # Initalize the match matrix
    match_matrix = None if not want_match_matrix else ([[None] * n_compounds] * n_coordinates)

    # figure files to score
    file_idxs = []
    for ion_loss in neutralizations:
        ms1_mass = ms1_mz + ion_loss
        mass_difference = np.abs(file_lookup_table['ms1_mass'] - ms1_mass)
        new_indices = np.where(mass_difference <= ms1_mass_tol)
        file_idxs = np.append(file_idxs, new_indices)

    # score selected files against every scan in data
    for idx in file_idxs:
        filename = file_lookup_table[idx]['filename']
        file_reader = h5py.File(filename)
        group_key = file_reader.keys()[0]
        data_key = file_reader[group_key].keys()[0]
        tree = file_reader[group_key][data_key][:]

        for i in xrange(n_coordinates):
            mzs = peakmzs
            intensities = peaklist[i, :]
            if intensities.max() == 0:
                continue  # if all peaks are 0 intensity (e.g. masked data) skip scoring algo and keep score at 0
            mz_intensity_arr = np.array([mzs, intensities])
            if want_match_matrix:
                score_cube[i, idx], match_matrix[i][idx] = calculate_MIDAS_score(mz_intensity_arr,
                                                                                 tree,
                                                                                 mass_tol=ms2_mass_tol,
                                                                                 neutralizations=neutralizations,
                                                                                 max_depth=max_depth,
                                                                                 want_match_matrix=True)
            else:
                score_cube[i, idx] = calculate_MIDAS_score(mz_intensity_arr,
                                                           tree,
                                                           mass_tol=ms2_mass_tol,
                                                           neutralizations=neutralizations,
                                                           max_depth=max_depth,
                                                           want_match_matrix=False)

    # Compute the shape the score cube should have when we return it
    # Alternatively one could e.g. do tuple(list(spatial_dimensions) + [n_compounds,])
    score_cube_shape = sum((spatial_dimensions, (n_compounds,)), ())

    if want_match_matrix:
        return score_cube.reshape(score_cube_shape), match_matrix.reshape(score_cube_shape)
    else:
        return score_cube.reshape(score_cube_shape), None


def score_scan_list_against_trees_serial(scan_list,
                                         ms1_mz,
                                         file_lookup_table,
                                         neutralizations,
                                         ms1_mass_tol,
                                         ms2_mass_tol,
                                         max_depth,
                                         want_match_matrix=False,
                                         scan_indexes=False,
                                         temp_out_group=None,
                                         mpi_root=None,
                                         mpi_comm=None):
    """
    Create a score cube of MIDAS scores for each

    :param scan_list:         list of numpy ndarrays    Stores peak mzs and intensities.  Each list el has shape of
                                                        (n_peaks, 2), with mzs in column 1 an intensities in col 2.
                                                        Length of list is n_scans long
    :param ms1_mz:            numpy array of floats            The (unneutralized) MS1 mz/s of the precursor ions.
                                                        Must have same length as scan_list, i.e. len(ms1_ms) = num_scans

    :param file_lookup_table: full path to a .npy file having a numpy structured array with columns
                              (i) filename and (ii) MS1 mass. Alternatively, this may also be the numpy array directly.

    :param ms1_mass_tol: float, max. diff. in Da of tree_parent_mass & MS1_precursor_mass

    :param ms2_mass_tol: float, max. mass in Da by which two MS2/MSn peaks can differ

    :param neutralizations:   list of floats, adjustments (in Da) added to data peaks in order to neutralize them

    :param max_depth:         int, optional.  For restricting scoring to lower max depths than
                              are present in the supplied tree

    :param want_match_matrix:   bool, if True then tuple of (score, match_matrix) is returned, else return score only.
                                Default value is False.

    :param scan_indexes: Optional 1D array or list of integer indices for the scans. Used to identify scans
        for logging. Default value is None.

    :param temp_out_group: HDF5 group where the results from this run should temporarily be stored. If given, then
        the results for each scan will be written to a group 'scan_#' which in turn contains the following
        datasets

         * `score_matrix` : The 2D score matrix
         * `match_matrix_#s_#c` where #s is the scan index and #c is the compound index. This dataset contains \
            the match matrix for the corresponding scan / compound combindation

    :param mpi_root: The root MPI task. Only needed for consistent logging when running in parallel. Set to NONE
        to do logging from all cores.

    :param mpi_comm: The mpi communicator to be used. Only needed for consistent logging when running in parallel.

    :return: score_matrix     a numpy ndarray of shape (n_scans, len(file_lookup_table))
    :return: match_matrix     Optional output that is only returned if want_match_matrix is set to True.
                              List of lists of match matrices (one matrix per scan and compound combination).
                              Each entry is a bool matrix with n_peaks columns and n_nodes rows. Elements are True
                              if given peak matches given node of frag_dag. An entry will be None in case that
                              the hit-score was 0. The return value will be None if want_match_matrix is set
                              to False (Default)

    Unlike score_peakcube_against_trees, this function is designed to work on _lists_ of scans.  It is more
     appropriate for scans directly extracted from mzML files or for centroided data.  This function does NOT
     assume that each scan in the list has the same precursors.
    """
    log_helper.debug(__name__, 'Processing scans', comm=mpi_comm, root=mpi_root)
    # load file_lookup_table if needed
    if isinstance(file_lookup_table, basestring):
        file_lookup_table = np.load(file_lookup_table)
    elif isinstance(file_lookup_table, np.ndarray):
        pass
    else:
        raise ValueError('Invalid file_lookup_table specified')

    # Check that the temporary output group is valid
    if temp_out_group is not None:
        if not (isinstance(temp_out_group, h5py.Group) or isinstance(temp_out_group, h5py.File)):
            raise ValueError("The temporary output parameter must be a HDF5 Group, HDF5 File or None")
        if isinstance(temp_out_group, h5py.File):
            temp_out_group = temp_out_group['/']

    # size input variables
    n_scans = len(scan_list)
    n_compounds = len(file_lookup_table)

    # Define the scan indexes if necessary
    if scan_indexes is None:
        scan_indexes = range(n_scans)

    # initialize output variable score_matrix
    score_matrix = np.zeros(shape=(n_scans, n_compounds), dtype=float)
    match_matrix = None if not want_match_matrix else ([[None] * n_compounds] * n_scans)

    # cannot assume common parent for all scans; must loop over scans first
    # if this part is slow it can be improved by grouping/clustering scans by common precursor first
    for i, scan in enumerate(scan_list):

        # check size of mz_intensity_arr
        if scan.shape[1] != 2:
            raise TypeError('Scans must be numpy arrays with a row for each peak, and *only* two columns')

        # figure files to score
        file_idxs = []
        for ion_loss in neutralizations:
            ms1_mass = ms1_mz[i] + ion_loss
            mass_difference = np.abs(file_lookup_table['ms1_mass'] - ms1_mass)
            new_indices = np.where(mass_difference <= ms1_mass_tol)
            file_idxs = np.append(file_idxs, new_indices)

        # score selected files against every scan in data
        start_time = time.time()
        for idx in file_idxs:
            filename = file_lookup_table[idx]['filename']
            file_reader = h5py.File(filename)
            group_key = file_reader.keys()[0]
            data_key = file_reader[group_key].keys()[0]
            tree = file_reader[group_key][data_key][:]
            if want_match_matrix:
                score_matrix[i, idx], match_matrix[i][idx] = calculate_MIDAS_score(scan,
                                                                                   tree,
                                                                                   mass_tol=ms2_mass_tol,
                                                                                   neutralizations=neutralizations,
                                                                                   max_depth=max_depth,
                                                                                   want_match_matrix=True)
            else:
                score_matrix[i, idx] = calculate_MIDAS_score(scan,
                                                             tree,
                                                             mass_tol=ms2_mass_tol,
                                                             neutralizations=neutralizations,
                                                             max_depth=max_depth,
                                                             want_match_matrix=False)
        number_of_hits = (score_matrix[i, idx] > 0).sum()

        end_time = time.time()
        execution_time = end_time - start_time
        time_str = "rank : " + str(mpi_helper.get_rank()) + " : index : " + \
                   str(scan_indexes[i]) + " : time in s : " + str(execution_time)
        time_str += " : num hits : " + str(number_of_hits)
        log_helper.info(__name__, time_str, comm=mpi_comm, root=None)
        sys.stdout.flush()
        sys.stderr.flush()

        # Save the score_matrix and match_matrix data if requested
        if temp_out_group is not None:
            score_data_group = temp_out_group.require_group('scan_'+str(scan_indexes[i]))
            score_data_group['score_matrix'] = score_matrix
            if want_match_matrix:
                for compound_index in range(n_compounds):
                    if match_matrix[i][compound_index] is not None:
                        match_matrix_dataset_name = 'match_matrix_%i_%i' % (scan_indexes[i], compound_index)
                        score_data_group[match_matrix_dataset_name] = match_matrix[i][compound_index]
            temp_out_group.file.flush()

    if want_match_matrix:
        return score_matrix, match_matrix
    else:
        return score_matrix, None


def make_pactolus_hit_table(pactolus_results, table_file, original_db, match_matrix=None):
    """
    Makes a hit table in the same format as lbl-midas for comparison of the two algorithms

    :param pactolus_results: ndarray,    n_compounds by n_scans matrix of pactolus scores
    :param table_file:      string,     full path to .npy file containing tree file names and parent masses that
                                            was used to generate pactolus results.  Or the numpy array directly.
    :param original_db:     string,     full path to flat text file containing the molecule DB used to generate the
                        fragmentation trees. The primary use for this is to enable lookup of molecule names.
    :param match_matrix:     Optional input. List of lists of match matrices (one matrix per scan and hit).
                              Each entry is a bool matrix with n_peaks columns and n_nodes rows. Elements are True
                              if given peak matches given node of frag_dag. An entry will be None in case that
                              the hit-score was 0. This is required to fill the number of peaks and matched peaks
                              entries in the able. If not available then the values will be set to 0.

    :return: hit_table_list, a list of hit_tables
    """
    # transform pactolus results into hit table
    global HIT_TABLE_DTYPE
    db_arr = crossref_to_db(table_file, original_db)

    # return a list of hit_tables when pactolus_results is a score_list
    hit_table_list = []
    for scan_index, scores in enumerate(pactolus_results):
        num_nonzero = np.count_nonzero(scores)
        hit_table = np.zeros(shape=(num_nonzero), dtype=HIT_TABLE_DTYPE)
        order = scores.argsort()[::-1][:num_nonzero]
        for idx, hit_index in enumerate(order):
            # Determine the number of peaks and number of matched peaks
            npeaks = 0
            nmatched_peaks = 0
            if match_matrix is not None:
                matches = match_matrix[scan_index][hit_index]
                # Since we only look at non-zero scores we should always have a match matrix but just be sure we check
                if matches is not None:
                    npeaks = matches.size
                    nmatched_peaks = matches.sum()
            # Compule the hittable entry
            hit_table[idx] = (scores[hit_index],
                              db_arr['metacyc_id'][hit_index],
                              db_arr['name'][hit_index],
                              db_arr['mass'][hit_index],
                              npeaks,
                              nmatched_peaks,
                              )
        hit_table_list.append(hit_table)
    assert len(hit_table_list) == pactolus_results.shape[0]
    return hit_table_list


def crossref_to_db(table_file, original_db):
    """
    Cross-references pactolus results to name, structure,  etc. data in a flat text file

    :param table_file:      string,     full path to .npy file containing tree file names and parent masses that
                                            was used to generate pactolus results. Or the numpy array directly.
    :param original_db:     string,     full path to flat text file containing DB
    :return db_arr:         numpy structured array re-ordered to match table_file. The numpy dtype is
                            score_frag_dag.METACYC_DTYPE  which has the fields:
                            'metacyc_id', 'name', 'inchi', 'lins', 'inchi_key'
    """
    # load flat-text DB and get inchi keys for each compound
    global METACYC_DTYPE
    with open(original_db, 'r+') as io:
        db_length = sum(1 for _ in io.readlines()) - 1  # subtract 1 because of header line
        db_arr = np.zeros(shape=(db_length, ), dtype=METACYC_DTYPE)
        io.seek(0)
        for idx, line in enumerate(io.readlines()):
            if idx == 0:
                continue  # ignore header
            fields = [el.strip() for el in line.split('\t')]
            inchi_key = Chem.inchi.InchiToInchiKey(fields[2])
            try:
                mol = Chem.MolFromInchi(fields[2])
                mass = CalcExactMolWt(mol)
            except:
                print fields[2], idx, line
                raise TypeError
            fields.append(inchi_key)
            fields.append(mass)
            db_arr[idx-1] = tuple(fields)

    # load .npy file & extract inchi keys from the filename
    if isinstance(table_file, basestring):
        table = np.load(table_file)
    elif isinstance(table_file, np.ndarray):
        table = table_file
    else:
        raise ValueError('Invalid table-file given')

    inchi_key_regex = re.compile('[A-Z-]+(?=[.])')
    table_inchi_keys = [re.findall(inchi_key_regex, filename)[0]
                        for filename in table['filename']]

    # look up where each key in table matches a db key
    matches = np.zeros(len(table_inchi_keys), dtype=int)
    db_inchi_keys = db_arr['inchi_key'].tolist()
    for idx, key in enumerate(table_inchi_keys):
        matches[idx] = db_inchi_keys.index(key)

    # return
    return db_arr[matches]


##################################################################
#    Implement the main function                                 #
##################################################################
def load_scan_data_hdf5(filepath,
                            grouppath=None):
    """
    Load the scans/spectra to be scored from file

    Expected HDF5 data layout within the group:

    * `peak_mz` : 1D array with all m/z values for all concatenated scans.
    * `peak_value` : 1D array with all intensity values for all concatenated scans. Must have the same \
            length as peak_mz.
    * `peak_arrayindex : 1D (or n-D array) where first dimension is the scan index and the last dimension \
            (in the case of n-D arrays) contains the integer start offset where each scan is located in \
            the peak_mz and peak_value  arrays. An n-D array is sometimes used to store additional location data \
            (e.g., the x/y location from which a scan is recorded). The additional data will be returned as \
            part of the scan_metadata in the form of peak_arrayindex_# = peak_arrayindex[:, #].
    * `ms1_mz` or `precursor_mz1` : Optional 1D array with the MS1 precursor m/z value for each scan. \
            Must have the same length as the number of scans (i.e, the length of the peak_array_index). \
            Alternatively the ms1_mz dataset may also be stored in the scan_metadata group. This will be \
            returned as part of the scan_metadata as well as as a separate return value.
    * `scan_metadata` : Group with additional arrays for per-scan metadata that should be passed through. \
            The first dimension of the arrays should always have the same lenght as the number of scans.
    * `experiment_metadata` : Group with additional arbitrary metadata pertaining to the experiment. This data \
            will also be pass through as is.


    :param filepath: The path to the file
    :param grouppath: The group object within the file

    :return:  scan_list  :  list of numpy ndarrays  with mzs and intensities.  Each list el has shape of
                            (n_peaks, 2), with mzs in column 1 an intensities in col 2.
                            Length of list is n_scans long
    :return:  scan_metadata : A dictionary of 1D numpy arrays with the same length as the scan_list where the
                           keys are the names of the metadata field and the arrays contain the per-scan
                           metadata. This also includes the ``ms1_mz`` data if available, which is a numpy array
                           of floats with the (unneutralized) MS1 mz/s of the precursor ions.
    :return:  experiment_metadata : A dictionary with additional arbitrary metadata about the collection of scans
                           as a whole

    """
    # Open the file
    input_file = h5py.File(filepath, 'r')
    input_group = input_file[grouppath]

    # Read the mz1_mz data with the precursor_mz values
    if 'precursor_mz' in input_group.keys():
        ms1_mz = input_group['precursor_mz'][:]
    elif 'ms1_mz' in input_group.keys():
        ms1_mz = input_group['ms1_mz'][:]
    elif 'scan_metadata' in input_group.keys():
        scan_metadata_group = input_group['scan_metadata']
        if 'precursor_mz' in scan_metadata_group.keys():
            ms1_mz = scan_metadata_group['precursor_mz'][:]
        elif 'ms1_mz' in scan_metadata_group.keys():
            ms1_mz = scan_metadata_group['ms1_mz'][:]
    else:
        ms1_mz = None

    # Compile the scan_list
    peak_mz = input_group['peak_mz']
    peak_value = input_group['peak_value']
    peak_arrayindex = input_group['peak_arrayindex']
    peak_arrayindex_is_2D = len(peak_arrayindex.shape) == 2
    num_scans = peak_arrayindex.shape[0]
    scan_list = []
    for scan_index in range(num_scans):
        # Determine the start/stop index of the current scan
        if peak_arrayindex_is_2D:
            start = int(peak_arrayindex[scan_index, -1])
            stop = int(peak_arrayindex[(scan_index+1), -1] if scan_index < (num_scans-1) else peak_value.size)
        else:
            start = int(peak_arrayindex[scan_index])
            stop = int(peak_arrayindex[(scan_index+1)] if scan_index < (num_scans-1) else peak_value.size)
        scan_length = stop - start
        # Load and compile the m/z and intensity values
        current_peaks_list = np.zeros(shape=(scan_length, 2), dtype=float)
        current_peaks_list[:, 0] = peak_mz[start:stop]
        current_peaks_list[:, 1] = peak_value[start:stop]
        # Add the scan to the scan_list
        scan_list.append(current_peaks_list)

    # Get the scan metadata
    scan_metadata = {}
    if ms1_mz is not None:
        scan_metadata = {'ms1_mz': ms1_mz}
    if 'scan_metadata' in input_group.keys():
        scan_metadata_group = input_group['scan_metadata']
        for dataset_name in scan_metadata_group.keys():
            metadata_dataset = scan_metadata_group[dataset_name][:]
            if metadata_dataset.ndim < 1 or metadata_dataset.shape[0] != num_scans:
                log_helper.warning(__name__, dataset_name + ' dimensions do not match number of scans. Data ignored.')
            else:
                scan_metadata[dataset_name] = metadata_dataset
    # Get scan metadata from the peak_arrayindex
    if peak_arrayindex_is_2D and peak_arrayindex.shape[1] > 1:
        for i in range(peak_arrayindex.shape[1]-1):
            scan_metadata['peak_arrayindex_%i' % i] = peak_arrayindex[:, i]

    # Check if we have any experiment metadata
    experiment_metadata = {}
    if 'experiment_metdata' in input_group.keys():
        experiment_metadata_group = input_group['experiment_metadata']
        for dataset_name in experiment_metadata_group.keys():
            metadata_dataset = experiment_metadata_group[dataset_name][:]
            experiment_metadata[dataset_name] = metadata_dataset

    # Close the input file
    input_file.close()

    # Return the data
    return scan_list, ms1_mz, scan_metadata, experiment_metadata


def load_file_lookup_table(path):
    """
    Load or generate a file lookup table based on the given path.

    :param path: This may be either the path to: 1) the .npy file with the file lookup table prepared via
                 make_file_lookup_table_by_MS1_mass, 2) a path to an HDF5 file ending in '.h5'
                 and dataset defined via <filename>:<datasetpath
                 3) a textfile where each line a path to a tree
                 4) the path to a directory with all trees that should be used. In the latter 2 cases the
                 make_file_lookup_table_by_MS1_mass function is called to compile the lookup table, whereas
                 in the first case the table is simply restored from file.

    :return: tree_files_by_ms1_mass, a numpy structured array with columns (i) filename and (ii) MS1 mass.
             The numpy dtype is type = [('filename', 'a400'),('ms1_mass', 'f8'), ]

    """
    if os.path.isfile(path):
        if path.endswith('.npy'):
            file_lookup_table = np.load(path)
        else:
            in_treefile = open(path, 'r')
            tree_files = [line.rstrip('\n') for line in in_treefile]
            in_treefile.close()
            file_lookup_table = make_file_lookup_table_by_MS1_mass(tree_files=tree_files)
            in_treefile.close()
    elif os.path.isdir(path):
        file_lookup_table = make_file_lookup_table_by_MS1_mass(path=path)
    else:
        split_path = path.split(':')
        if len(split_path) == 2 and split_path[0].endswith('.h5'):
            hdf_file = h5py.File(split_path[0], 'r')
            file_lookup_table = hdf_file[split_path[1]][:]
        else:
            file_lookup_table = None

    return file_lookup_table


def parse_command_line_args():
    """
    Define and parse all command line arguments

    :return: Dictionary with the parsed command line arguments. See --help for a list of all command line arguments
             and their names. The basic keys in the dict are:

             * `input` : The full input path provided by the user
             * `input_filepath` : The path to the input file
             * `input_grouppath` : The path to the group within the input file
             * `output` : The full path to the output file
             * `output_filepath` : The path to the ouput file
             * `ouput_grouppath` : The path to to the group within the output file
             * `precursor_mz` : Optional precursor_mz floating point value (-1 by default, i.e., read from file)
             * `metabolite_database` : Optional path to the metabolite database
             * `trees` : The file or dir with the list of trees to be used for scoring
             * `ms1_mass_tolerance` : The ms1 mass tolerance floating point value
             * `ms2_mass_tolerance` : The ms2 mass tolerance floating point value
             * `max_depth` : The maximum search depth in the trees integer value
             * `neutralizations` : Numpy array with floating point neutralization values
             * `pass_meta`: Boolean indicating whether we should pass additional metadata through to the ouptut
             * `pass_scans`: Boolean indicating whether the actual scan/spectra data should be passed through \
                            to the output.
             * `schedule` : The scheduling scheme to be used
             * `collect` : Boolean defining whether the results should be collected to rank 0 in parallel
             * `loglevel` : String indicating the logging level to be used
             * `temppath` : Path basename where temporary data files should be stored. Temporary files are created \
                           one-per-core to incrementally save the results of an analysis.

    """
    dtypes = data_dtypes.get_dtypes()

    # Create the argparse parser
    parser_description = "score scan list against trees:\n"
    parser_epilog = "HDF5 input data format: \n" + \
                    "----------------------- \n" + \
                    "The data within the group should be stored as follows: \n" + \
                    "   (1) `peak_mz` : 1D array with all m/z values for all concatenated scans.\n" + \
                    "   (2) `peak_value` : 1D array with all intensity values for all concatenated \n" + \
                    "       scans. Must have the same length as peak_mz. \n" + \
                    "   (3) `peak_array_index` : 1D (or n-D array) where the first dimension must \n" + \
                    "        be the scan index and the last dimension  (in the case \n" + \
                    "        of n-D arrays) must contain the integer start offset where \n" + \
                    "        each scan is located in the peak_mz and peak_value arrays.\n" + \
                    "        An n-D array is sometimes used to store additional location \n" + \
                    "        That additional data will be ignored.\n" + \
                    "  (4) `mz1_mz` or `precursor_mz1` : 1D array with the MS1 precursor m/z value  \n" + \
                    "      for each scan. Must be #scans long. \n" + \
                    "  (5) `scan_metadata` : Group with additional arrays for per-scan  \n" + \
                    "      metadata that should be passed through. The first  \n" + \
                    "      dimension of the arrays should always have the same  \n" + \
                    "      length as the number of scans.  \n" + \
                    "  (6) `experiment_metadata` : Group with additional arbitrary metadata  \n" + \
                    "      pertaining to the experiment. This data is pass through as is. " + \
                    "\n\n This command-line tool is broad to you by Pactolus. (LBNL)"
    parser = argparse.ArgumentParser(description=parser_description,
                                     epilog=parser_epilog,
                                     formatter_class=RawDescriptionDefaultHelpArgParseFormatter,
                                     add_help=True)
    required_argument_group = parser.add_argument_group(title="required analysis arguments")
    optional_argument_group = parser.add_argument_group(title="optional analysis arguments")
    parallel_argument_group = parser.add_argument_group(title="parallel execution arguments")

    # Add all command line arguments
    required_argument_group.add_argument('--input',
                                         type=dtypes['str'],
                                         required=True,
                                         action="store",
                                         help="Path to the HDF5 file with the input scan data consisting of the" +
                                              "<filenpath>:<grouppath> were <filepath> is the path to the file and" +
                                              "<grouppath> is the path to the group within the file. E.g. a valid  " +
                                              " definition may look like: 'test_brain_convert.h5:/entry_0/data_0. "
                                              " See below for details on how to store the data in HDF5.",
                                         dest="input")
    optional_argument_group.add_argument('--save',
                                         type=dtypes['str'],
                                         required=False,
                                         default='',
                                         action="store",
                                         help="Path to the HDF5 file where the output should be saved consisting of the" +
                                              "<filenpath>:<grouppath> were <filepath> is the path to the file and" +
                                              "<grouppath> is the path to the group within the file. E.g. a valid  " +
                                              " definition may look like: 'test_brain_convert.h5:/entry_0/data_0. "
                                              " See below for details on how to store the data in HDF5.",
                                         dest="save")
    optional_argument_group.add_argument("--precursor_mz",
                                         help="Floating point precursor mass over charge value. Default value is -1, " +
                                              "indicating that the precursor m/z should be read from file.",
                                         action="store",
                                         type=dtypes["float"],
                                         required=False,
                                         default=-1,
                                         dest="precursor_mz")
    optional_argument_group.add_argument("--metabolite_database",
                                         help="The database of metabolites from which the trees were generated." +
                                              "Needed only if compound metadata from the database should be included" +
                                              "in the output.",
                                         action="store",
                                         type=dtypes["str"],
                                         default="",
                                         required=False,
                                         dest="metabolite_database")
    required_argument_group.add_argument("--trees",
                                         help="1) Path to the sub-directory with all relevent .h5 pactolus fragmentation trees. or " +
                                              "2) path to a text-file with a list of names of the tree files, 3) Path to the" +
                                              ".npy file with the pactolus file lookup table, or 4) Path to corresponding " +
                                              "dataset in an HDF5 file where the path is given by <filename>:<dataset_path>. ",
                                         action="store",
                                         type=dtypes["str"],
                                         required=True,
                                         dest="trees")
    required_argument_group.add_argument("--ms1_mass_tolerance",
                                         help="float, max. diff. in Da of tree_parent_mass & MS1_precursor_mass",
                                         action="store",
                                         type=dtypes["float"],
                                         default=0.01,
                                         required=True,
                                         dest="ms1_mass_tolerance")
    required_argument_group.add_argument("--ms2_mass_tolerance",
                                         help="float, max. mass in Da by which two MS2/MSn peaks can differ",
                                         action="store",
                                         type=dtypes["float"],
                                         default=0.01,
                                         required=True,
                                         dest="ms2_mass_tolerance")
    required_argument_group.add_argument("--max_depth",
                                         help="Maximum depth of fragmentation pathways",
                                         action="store",
                                         type=dtypes["int"],
                                         choices=range(10),
                                         default=5,
                                         required=True,
                                         dest="max_depth")
    required_argument_group.add_argument("--neutralizations",
                                         help="List of floats, adjustments (in Da) added to data peaks in order to " +
                                              "neutralize them. Here we can use standard python syntax, e.g, " +
                                              "'[1,2,3,4]' or '[[1, 3], [4, 5]]'. Remember to include the array " +
                                              "string in quotes",
                                         action="store",
                                         required=True,
                                         type=dtypes["ndarray"],
                                         default=[0, 1, 2],
                                         dest="neutralizations")
    parallel_argument_group.add_argument("--schedule",
                                         help="Scheduling to be used for parallel MPI runs",
                                         action="store",
                                         type=dtypes["str"],
                                         required=False,
                                         choices=mpi_helper.parallel_over_axes.SCHEDULES.values(),
                                         default=mpi_helper.parallel_over_axes.SCHEDULES["DYNAMIC"],
                                         dest="schedule")
    parallel_argument_group.add_argument("--collect",
                                         help="Collect results to the MPI root rank when running in parallel",
                                         action="store",
                                         type=dtypes["bool"],
                                         required=False,
                                         default=True,
                                         dest="collect")
    optional_argument_group.add_argument("--pass_meta",
                                         help="Pass per-scan metadata through and include it in the final output",
                                         action="store",
                                         type=dtypes["bool"],
                                         required=False,
                                         default=True,
                                         dest="pass_meta")
    optional_argument_group.add_argument("--pass_scans",
                                         help="Pass the scan/spectrum data through and include it in the final output",
                                         action="store",
                                         type=dtypes["bool"],
                                         required=False,
                                         default=True,
                                         dest="pass_scans")
    optional_argument_group.add_argument("--match_matrix",
                                         help="Track and record the match matrix data. Required if the num_matched " +
                                              "and match_matrix_s_c data should be included in the output.",
                                         action="store",
                                         type=dtypes["bool"],
                                         required=False,
                                         default=True,
                                         dest="match_matrix")
    optional_argument_group.add_argument("--loglevel",
                                         help="Define the level of logging to be used.",
                                         type=dtypes['str'],
                                         action="store",
                                         default="INFO",
                                         required=False,
                                         choices=log_helper.log_levels.keys())
    optional_argument_group.add_argument("--tempath",
                                         help="Path basename where temporary data files should be stored. " +
                                              "Temporary files are created one-per-core to incrementally " +
                                              "save the results of an analysis. If not given, then a temp-dir" +
                                              "will be created automatically.",
                                         type=dtypes['str'],
                                         action="store",
                                         default="",
                                         required=False,
                                         choices=log_helper.log_levels.keys())

    # Parse the command line arguments
    command_line_args = vars(parser.parse_known_args()[0])

    # Add the filepath and grouppath values based on the the --input argument value
    input_path = command_line_args['input'].split(':')
    if len(input_path) == 2:
        input_filepath = input_path[0]
        input_grouppath = input_path[1]
    elif len(input_path) == 1:
        input_filepath = input_path[0]
        input_grouppath = '/'
    else:
        raise ValueError('Invalid input path: ' + input_path)
    command_line_args['input_filepath'] = input_filepath
    command_line_args['input_grouppath'] = input_grouppath

    # Add the filepath and grouppath values based on the the --save argument value
    output_path = command_line_args['save'].split(':"')
    if len(output_path) == 2:
        output_filepath = output_path[0]
        output_grouppath = output_path[1]
    elif len(output_path) == 1 and len(output_path[0]) > 0:
        output_filepath = output_path[0]
        output_grouppath = '/'
    else:
        output_filepath = None
        output_grouppath = None
    command_line_args['output_filepath'] = output_filepath
    command_line_args['output_grouppath'] = output_grouppath

    # Correcte metabolite database
    if len(command_line_args['metabolite_database']) == 0:
        command_line_args['metabolite_database'] = None

    # Returned the parsed command line data
    return command_line_args


def score_scan_list_against_trees(scan_list,
                                  ms1_mz,
                                  file_lookup_table,
                                  neutralizations,
                                  ms1_mass_tol,
                                  ms2_mass_tol,
                                  max_depth,
                                  want_match_matrix=False,
                                  temp_out_group=None,
                                  collect=False,
                                  schedule=mpi_helper.parallel_over_axes.SCHEDULES['DYNAMIC'],
                                  mpi_comm=None,
                                  mpi_root=0):
    """
    TODO Add documentation

    :param scan_list:
    :param ms1_mz:
    :param file_lookup_table:
    :param neutralizations:
    :param ms1_mass_tol:
    :param ms2_mass_tol:
    :param max_depth:
    :param want_match_matrix:
    :param temp_out_group:
    :param schedule:
    :param mpi_comm:
    :param mpi_root:
    :return:
    """
    if not mpi_helper.MPI_AVAILABLE or mpi_helper.get_size() == 1:
        result = score_scan_list_against_trees_serial(
                scan_list=scan_list,
                ms1_mz=ms1_mz,
                file_lookup_table=file_lookup_table,
                neutralizations=neutralizations,
                ms1_mass_tol=ms1_mass_tol,
                ms2_mass_tol=ms2_mass_tol,
                max_depth=max_depth,
                want_match_matrix=want_match_matrix,
                temp_out_group=temp_out_group)
    else:
        result = score_scan_list_against_trees_parallel(
                scan_list=scan_list,
                ms1_mz=ms1_mz,
                file_lookup_table=file_lookup_table,
                neutralizations=neutralizations,
                ms1_mass_tol=ms1_mass_tol,
                ms2_mass_tol=ms2_mass_tol,
                max_depth=max_depth,
                want_match_matrix=want_match_matrix,
                temp_out_group=temp_out_group,
                scan_indexes=None,
                schedule=schedule,
                mpi_comm=mpi_comm if mpi_comm is not None else mpi_helper.get_comm_world(),
                mpi_root=mpi_root)

    return result

def score_scan_list_against_trees_parallel(scan_list,
                                           ms1_mz,
                                           file_lookup_table,
                                           neutralizations,
                                           ms1_mass_tol,
                                           ms2_mass_tol,
                                           max_depth,
                                           want_match_matrix=False,
                                           temp_out_group=None,
                                           scan_indexes=False,
                                           schedule=mpi_helper.parallel_over_axes.SCHEDULES['DYNAMIC'],
                                           mpi_comm=None,
                                           mpi_root=0):
    """
    TODO Add documentation

    :param scan_indexes:
    :param scan_list:
    :param ms1_mz:
    :param file_lookup_table:
    :param neutralizations:
    :param ms2_mass_tol:
    :param ms1_mass_tol:
    :param max_depth:
    :param metabolite_database:
    :param temp_out_group:
    :param want_match_matrix:
    :param schedule:  (None means serial)
    :param mpi_comm:
    :param mpi_root:

    :return: score_matrix     a numpy ndarray of shape (n_scans, len(file_lookup_table))
    :return: match_matrix     Optional output that is only returned if want_match_matrix is set to True.
                              List of lists of match matrices (one matrix per scan). Each entry is a bool matrix
                              with n_peaks columns and n_nodes rows. Elements are True if given peak matches given
                              node of frag_dag. An entry will be None in case that the hit-score was 0. The return
                              value will be None if want_match_matrix is set to False (Default)
    :return: scan_index   Numpy array of integer indices indicating the subset of scans for procesing was
                              performed by the given core (and or collected to the core in case of the root rank
                              with collect set to True).

    """
    if mpi_comm is None:
        mpi_comm = mpi_helper.get_comm_world()

    # size input variables
    num_scans = len(scan_list)
    num_compounds = len(file_lookup_table)
    # Define the list of scan indexes
    if scan_indexes is None:
        # Get the complete peak array index data
        scan_indexes = np.arange(0, num_scans)
        enable_parallel = True
    else:
        if isinstance(scan_indexes, int):
            scan_indexes = np.asarray([scan_indexes, ])
        enable_parallel = False

    #############################################################
    # Parallel execution using MPI
    #############################################################
    # We have more than a single core AND we have multiple scans to process
    if mpi_helper.get_size() > 1 and len(scan_indexes) > 1 and schedule is not None:
        # We were not asked to process a specific data subblock from a parallel process
        # but we need to initiate the parallel processing.
        if enable_parallel:
            log_helper.debug(__name__, 'Preparing parallel execution', comm=mpi_comm, root=mpi_root)
            # Setup the parallel processing using mpi_helper.parallel_over_axes
            split_axis = [0, ]
            scheduler = mpi_helper.parallel_over_axes(
                task_function=score_scan_list_against_trees_parallel,   # Execute this function
                task_function_params={'file_lookup_table': file_lookup_table,
                                      'scan_list': scan_list,
                                      'ms1_mz': ms1_mz,
                                      'neutralizations': neutralizations,
                                      'ms1_mass_tol': ms1_mass_tol,
                                      'ms2_mass_tol': ms2_mass_tol,
                                      'max_depth': max_depth,
                                      'want_match_matrix': want_match_matrix,
                                      'temp_out_group': temp_out_group,
                                      'schedule': None,
                                      'mpi_comm': mpi_comm,
                                      'mpi_root': mpi_root},  # Reuse the input data
                main_data=scan_indexes,                             # Process the scans independently
                split_axes=split_axis,                              # Split along axes
                main_data_param_name='scan_indexes',                # data input param
                root=mpi_root,                                      # The root MPI task
                schedule=schedule,                                  # Parallel scheduling scheme
                comm=mpi_comm)                                      # MPI communicator

            # Execute the analysis in parallel
            results, block_index = scheduler.run()

            # If we processed scans then compile the results
            scan_index = np.empty(dtype='int')
            score_matrix = np.zeros(shape=(0, num_compounds), dtype=float)
            match_matrix = [] if want_match_matrix else None
            if len(results) > 0:
                 # Compile the result from the current core
                score_matrix_list = [ri[0] for ri in results]
                match_matrix_list = [ri[1] for ri in results]
                scan_index_list = [ri[2] for ri in results]
                # Compile the list of scan indexes
                if len(scan_index_list) > 0:
                    scan_index = np.concatenate(tuple(scan_index_list), axis=0)
                    num_processed_scans = scan_index.size
                else:
                    num_processed_scans = 0
                # Compile the score matrix
                if len(score_matrix_list) > 0:
                    score_matrix = np.concatenate(tuple(score_matrix_list), axis=0)
                # Compile the match matrix list
                if want_match_matrix and len(match_matrix_list) > 0:
                    for mm in match_matrix_list:
                        match_matrix += mm

            return score_matrix, match_matrix, scan_index

    #############################################################
    # Serial processing of the current data block
    #############################################################
    # Using DYNAMIC scheduling scan_indexes will be a list with a single integer index while using
    # STATIC scheduling multiple scans may be processed at once.
    my_scans = [scan_list[i] for i in scan_indexes]
    my_ms1_mz = [ms1_mz[i] for i in scan_indexes]
    score_matrix, match_matrix  = score_scan_list_against_trees_serial(
            scan_list=my_scans,
            ms1_mz=my_ms1_mz,
            file_lookup_table=file_lookup_table,
            neutralizations=neutralizations,
            ms1_mass_tol=ms1_mass_tol,
            ms2_mass_tol=ms2_mass_tol,
            max_depth=max_depth,
            want_match_matrix=want_match_matrix,
            temp_out_group=temp_out_group,
            mpi_root=None, # We want to log the results from all scans
            mpi_comm=mpi_comm)
    return score_matrix, match_matrix, scan_indexes


def collect_score_scan_list_results(temp_filename_lists,
                                    output_filepath,
                                    output_grouppath,
                                    num_scans,
                                    num_compounds,
                                    scan_list,
                                    save_scan_list=False,
                                    scan_metadata=None,
                                    experiment_metadata=None,
                                    file_lookup_table=None,
                                    compound_metadata=None,
                                    mpi_comm=None,
                                    mpi_root=0):
    """
    Consolidate the results from the incrementally stored scoring results into
    a single file with a consolidated data layout

    :param temp_filename_lists: A list of temporary files with data
    :param output_filepath: The path to the output file. The file will be created if it does not exists or appended
            if the file does exist.
    :param output_grouppath: The path to the group in the output file. This may be a hierarchy of groups
            given by a standard "/" separated path. Any groups missing in the hierarchy will be created.
    :param num_scans: The number of scans (i.e, scans) that were scored.
    :param num_compounds: The number of compounds included in the scoring.
    :param scan_list:         list of numpy ndarrays    Stores peak mzs and intensities.  Each list el has shape of
                                                        (n_peaks, 2), with mzs in column 1 an intensities in col 2.
                                                        Length of list is n_scans long
    :param save_scan_list: Boolean indicating whether we should include the scan_list in the output file
    :param scan_meta: Dictionary of scan metadata. Keys are dataset names and values are numpy data arrays.
    :param experiment_meta: Dictionary of the experiment metadata. Keys are dataset names and values are
                numpy data arrays.
    :param compound_metadata: Dictionary of compound metadata. Keys are dataset names and values are numpy data
                arrays where the first dimension should have the same lenght and order as the compounds/tree
                dimension in the scores array.
    :param file_lookup_table: The numpy array with the file look-up table for the tree files used for scoring
    :param mpi_comm: The MPI communicator to be used
    :param mpi_root: The MPI root rank to be used for writing

    """
    if mpi_helper.get_rank(comm=mpi_comm) != mpi_root:
        return

    # Create/open the main output file
    output_file = h5py.File(output_filepath, 'a')
    # Get the target group or create the required group(s) if necessary
    try:
        output_group = output_file[output_grouppath]
    except KeyError:
        current_group = output_file['/']   # Split the path into the hiearchy of subgroups to be used.
        # Creat all subgroups
        for subgroup in output_grouppath.split('/'):
            current_group = current_group.require_group(subgroup)
        output_group = current_group


    score_dataset = output_group.create_dataset(name='score_matrix',
                                                shape=(num_scans, num_compounds),
                                                dtype='float',
                                                chunks=True,
                                                fillvalue=0,
                                                compression='gzip',
                                                compression_opts=2)
    num_matched = np.zeros(shape=(num_scans, num_compounds), dtype='int')

    # Open the temporary output files one-by-one and add the data to the output file
    for temp_output_filename in temp_filename_lists:
        temp_output_file=h5py.File(temp_output_filename, 'r')
        scan_group_names= [g for g in temp_output_file.keys() if g.startswith('scan_') ]
        # Iterate over all scan scoring results in the temporary output file
        for scan_group_name in scan_group_names:
            scan_group = temp_output_file[scan_group_name]
            scan_index = int(scan_group_name.split('_')[-1])
            match_matrix_dataset_names = [d for d in scan_group.keys() if d.startswith('match_matrix_')]
            # Copy the scores for the scan into the main output file
            score_dataset[scan_index, :] = scan_group['score_matrix'][:]
            # Copy all match-matrices (if any) to the main output file
            for match_matrix_name in match_matrix_dataset_names:
                compound_index = int(match_matrix_name.split('_')[-1])
                match_matrix = scan_group[match_matrix_name][:]
                num_matched[scan_index, compound_index] = match_matrix.sum()
                output_group[match_matrix_name] = match_matrix
                num_matched = scan_group[match_matrix_name][:].sum()

    # Save the number of matched peaks array if anything is available
    if num_matched.sum() > 0:
        num_matched_dataset = output_group.create_dataset(name='num_matched',
                                                          data=num_matched,
                                                          chunks=True,
                                                          compression='gzip',
                                                          compression_opts=2)
        output_file.flush()
    del num_matched

    # Compute the ranking of the scores
    rank_dataset = output_group.create_dataset(name='score_rank_matrix',
                                               shape=(num_scans, num_compounds),
                                               dtype='int',
                                               fillvalue=-1,
                                               chunks=True,
                                               compression='gzip',
                                               compression_opts=2)
    for i in range(num_scans):
        scores = score_dataset[i,:]
        nonzero_scores = np.where(scores > 0)[0]
        num_nonzero = nonzero_scores.size
        ranking = scores.argsort()[::-1][:num_nonzero]
        rank_dataset[nonzero_scores] = ranking

    # Write the compound metadata
    compound_metadata_group = output_group.require_group('compound_metadata')
    if compound_metadata is not None:
        for key, value in compound_metadata.iteritems():
            compound_metadata_group[key] = value
            output_file.flush()

    # Write the additional per-scan metdata
    scan_metadata_group = output_group.require_group('scan_metadata')
    if scan_metadata is not None:
        for key, value in scan_metadata.iteritems():
            scan_metadata_group [key] = value
            output_file.flush()

    # Compile and add the n_peaks data
    num_peaks = np.asarray([scan.shape[0] for scan in scan_list])
    scan_metadata_group['num_peaks'] = num_peaks

    # Write the experiment metadata
    experiment_metadata_group = output_group.require_group('experiment_metadata')
    if experiment_metadata is not None:
        for key, value in experiment_metadata.iteritems():
            experiment_metadata_group[key] = value
            output_file.flush()

    # Write the file look-up table if necessary
    if file_lookup_table:
        output_group['tree_file_lookup_table'] = file_lookup_table

    # Write the scan_list data if necessary
    if scan_list is not None:
        scan_group = output_group.require_group('scans')
        scan_group['peak_mz'] = np.concatenate(tuple([ri[0] for ri in scan_list]), axis=-1)
        scan_group['peak_value'] = np.concatenate(tuple([ri[1] for ri in scan_list]), axis=-1)
        scan_group['peak_arrayindex'] = np.cumsum([0] + [ ri[0].shape[0] for ri in scan_list ])[:-1]
        # Create a link to the experiment and scan metadata
        scan_group['experiment_metadata'] = experiment_metadata_group
        scan_group['scan_metadata'] = scan_metadata_group



    # Compile the matrix describing the number of peaks that were matched


    # Compile the compound metadata if available
    # HIT_TABLE_DTYPE = np.dtype({'names': ['score', 'id', 'name',  'mass', 'n_peaks', 'n_match'],
    #                             'formats': ['f4', 'a100', 'a100', 'f4',   'i4',       'i4']})

    # def make_pactolus_hit_table(pactolus_results, table_file, original_db, match_matrix=None):
    # # transform pactolus results into hit table
    # global HIT_TABLE_DTYPE
    # db_arr = crossref_to_db(table_file, original_db)
    #
    # # return a list of hit_tables when pactolus_results is a score_list
    # hit_table_list = []
    # for scan_index, scores in enumerate(pactolus_results):
    #     num_nonzero = np.count_nonzero(scores)
    #     hit_table = np.zeros(shape=(num_nonzero), dtype=HIT_TABLE_DTYPE)
    #     order = scores.argsort()[::-1][:num_nonzero]
    #     for idx, hit_index in enumerate(order):
    #         # Determine the number of peaks and number of matched peaks
    #         npeaks = 0
    #         nmatched_peaks = 0
    #         if match_matrix is not None:
    #             matches = match_matrix[scan_index][hit_index]
    #             # Since we only look at non-zero scores we should always have a match matrix but just be sure we check
    #             if matches is not None:
    #                 npeaks = matches.size
    #                 nmatched_peaks = matches.sum()
    #         # Compule the hittable entry
    #         hit_table[idx] = (scores[hit_index],
    #                           db_arr['metacyc_id'][hit_index],
    #                           db_arr['name'][hit_index],
    #                           db_arr['mass'][hit_index],
    #                           npeaks,
    #                           nmatched_peaks,
    #                           )
    #     hit_table_list.append(hit_table)
    # assert len(hit_table_list) == pactolus_results.shape[0]
    # return hit_table_list


def main(use_command_line=True, **kwargs):
    """
    Main function used to execute the scoring of scans

    :param use_command_line: Retrieve analysis settings from the command line
    :param kwargs: Optional keyword arguments used to define all (or overwrite some) settings usually defined
        via the command line. Optional keyqord arguments are:

         * `input` : The full input path provided by the user
         * `input_filepath` : The path to the input file
         * `input_grouppath` : The path to the group within the input file
         * `output` : The full path to the output file
         * `output_filepath` : The path to the ouput file
         * `ouput_grouppath` : The path to to the group within the output file
         * `precursor_mz` : Optional precursor_mz floating point value (-1 by default, i.e., read from file)
         * `metabolite_database` : Optional path to the metabolite database
         * `trees` : The file or dir with the list of trees to be used for scoring
         * `ms1_mass_tolerance` : The ms1 mass tolerance floating point value
         * `ms2_mass_tolerance` : The ms2 mass tolerance floating point value
         * `max_depth` : The maximum search depth in the trees integer value
         * `neutralizations` : Numpy array with floating point neutralization values
         * `pass_scan_meta`: Boolean indicating whether we should pass additional metadata
         * `schedule` : The scheduling scheme to be used
         * `loglevel` : String indicating the logging level to be used
         * `temppath` : Path basename where temporary data files should be stored. Temporary files are created \
                       one-per-core to incrementally save the results of an analysis.

    """
    global METACYC_DTYPE

    mpi_root = 0
    mpi_comm = mpi_helper.get_comm_world()

    # Parse the command line arguments if necessary
    if use_command_line:
        log_helper.debug(__name__, 'Parsing command line arguments', comm=mpi_comm, root=mpi_root)
        command_line_args = parse_command_line_args()
        # Overwrite command line arguments if necessary
        if len(kwargs) > 0:
            log_helper.debug(__name__, 'Overwrite the following command line arguments ' + str(kwargs), comm=mpi_comm, root=mpi_root)
            command_line_args.update(kwargs)
    else:
         command_line_args = kwargs

    # Get the command-line arguments
    ms1_mass_tol = command_line_args['ms1_mass_tolerance']
    ms2_mass_tol = command_line_args['ms2_mass_tolerance']
    max_depth = command_line_args['max_depth']
    neutralizations = command_line_args['neutralizations']
    schedule = command_line_args['schedule']
    trees = command_line_args['trees']
    metabolite_database = command_line_args['metabolite_database']
    input_filepath = command_line_args['input_filepath']
    input_grouppath = command_line_args['input_grouppath']
    output_filepath = command_line_args['output_filepath']
    output_grouppath = command_line_args['output_grouppath']
    match_matrix = command_line_args['match_matrix']
    cl_precursor_mz = command_line_args['precursor_mz']
    pass_meta = command_line_args['pass_meta']
    pass_scans = command_line_args['pass_scans']
    loglevel = command_line_args['loglevel']
    temppath = command_line_args['temppath']

    # Set the log level
    if loglevel in log_helper.log_levels.keys():
        log_helper.set_log_level(level=log_helper.log_levels[loglevel])
    else:
        log_helper.error(module_name=__name__, message="Invalid log level specified")

    # Read the scan data from file
    # TODO: Possibly optimize data load by reading only on the mpi root and then sending the data to all ranks via MPI
    log_helper.debug(__name__, 'Reading the input scans for scoring', comm=mpi_comm, root=mpi_root)

    scan_list, scan_metadata, experiment_metadata = load_scan_data_hdf5(filepath=input_filepath,
                                                                       grouppath=input_grouppath)
    ms1_mz = scan_metadata['ms1_mz'] if 'ms1_mz' in scan_metadata else None

    if cl_precursor_mz >= 0:
        if ms1_mz is not None:
            log_helper.info(__name__,
                            'Precursor m/z values stored in the input file overwritten by command-line arg value.',
                            mpi_root=mpi_root, mpi_helper=mpi_helper)
        mz1_mz = cl_precursor_mz

    # Load the file lookup table
    # TODO: Possibly optimize lookup table generaltion by performing only on mpi root sending to other ranks via MPI
    log_helper.debug(__name__, 'Preparing file lookup table', comm=mpi_comm, root=mpi_root)
    file_lookup_table = load_file_lookup_table(path=trees)

    # size input variables
    num_scans = len(scan_list)
    num_compounds = len(file_lookup_table)

    # Initalize the temporary output file
    temp_out_group=None
    if len(temppath) > 0:
        if os.path.isdir(temppath):
            temp_out_file = h5py.File(os.path.join(temppath, 'pactolus_rankfile_' + str(mpi_helper.get_rank()) + ".h5"))
            temp_out_group = temp_out_file['/']
    elif output_filepath is not None:
        # Create a NamedTempory file that will be deleted automaticaly at the end of the run
        local_tempfile = NamedTemporaryFile()
        temp_out_file = h5py.File(local_tempfile.name)
        temp_out_group = temp_out_file['/']

    # Score the scans
    log_helper.debug(__name__, "Starting the scoring process", comm=mpi_comm, root=mpi_root)
    results = score_scan_list_against_trees(scan_list=scan_list,
                                            ms1_mz=ms1_mz,
                                            file_lookup_table=file_lookup_table,
                                            neutralizations=neutralizations,
                                            ms2_mass_tol=ms2_mass_tol,
                                            ms1_mass_tol=ms1_mass_tol,
                                            max_depth=max_depth,
                                            metabolite_database=metabolite_database,
                                            temp_out_group=temp_out_group,
                                            want_match_matrix=match_matrix,
                                            schedule=schedule,
                                            mpi_comm=mpi_comm,
                                            mpi_root=mpi_root)

    # Compile the compound metadata
    compound_metadata = {}
    if metabolite_database:
        try:
            compound_db = crossref_to_db(table_file=file_lookup_table,
                                         original_db=metabolite_database)
            for field_name in METACYC_DTYPE.names:
                key = field_name if field_name != 'metacyc_id' else 'id'
                compound_metadata[field_name] = compound_db[field_name]
        except:
            log_helper.error(__name__, "Crossreferencing with metabolite database failed." + str(sys.exc_info()),
                             root=mpi_root, comm=mpi_comm)

    # Compile the results from all output files to the output file if requested
    cleanup_temporary_files = False
    if output_filepath is not None:
        log_helper.debug(__name__, 'Collecting results from the temporary files', comm=mpi_comm, root=mpi_root)
        # Collect the list of temporary files we need to read
        all_tempfile_names = mpi_helper.gather(temp_out_group.file.name, comm=mpi_comm, root=mpi_root)
        # Compile the scan data if we should pass it through as well
        #scan_data = {}
        #if pass_sepctra:
        #    input_file = h5py.File(input_filepath, 'r')
        #    input_group = input_file[input_grouppath]
        #    scan_data['peak_mz'] = input_group['peak_mz'][:]
        #    scan_data['peak_value'] = input_group['peak_value'][:]
        #    scan_data['peak_arrayindex'] = input_group['peak_arrayindex'][:]

        # collect_score_scan_list_results takes care of the parallel case, i.e., its safe to call it from all cores
        collect_score_scan_list_results(temp_filename_lists=all_tempfile_names,
                                        output_filepath=output_filepath,
                                        output_grouppath=output_grouppath,
                                        num_scans=num_scans,
                                        num_compounds=num_compounds,
                                        scan_list=scan_list,
                                        save_scan_list=pass_scans,
                                        scan_metadata=scan_metadata if pass_meta else {},
                                        experiment_metadata=experiment_metadata if pass_meta else {},
                                        compound_metadata=compound_metadata,
                                        file_lookup_table=file_lookup_table,
                                        mpi_comm=mpi_comm,
                                        mpi_root=mpi_root)
        cleanup_temporary_files = True

    # Close the output file for the current core. We do this after the consolidation of the ouptut data
    # to avoid that Python might clean up the NamedTemporaryFile when we close it.
    if temp_out_group is not None:
        tempfile_name = temp_out_group.file.name()
        temp_out_group.file.close()
        if cleanup_temporary_files:
            os.remove(tempfile_name)


if __name__ == "__main__":
    main(use_command_line=True)


# Save the data to file if necessary
# if output_filepath is not None and mpi_helper.get_rank() == mpi_root:
#     # Compile the output data for storage to HDF5 and sort the values to match the input order
#     output_data = {'scan_index': results[0]}
#     output_data['score'] = results[1][output_data['scan_index']]
#     output_data['id_data'] = results[2][output_data['scan_index']]
#     output_data['name'] = results[3][output_data['scan_index']]
#     output_data['mass'] = results[4][output_data['scan_index']]
#     output_data['n_peaks'] = results[5][output_data['scan_index']]
#     output_data['n_match'] = results[6][output_data['scan_index']]
#     output_data['id_data'] = results[2][output_data['scan_index']]
#     if not pass_meta:
#         scan_metadata = {}
#         experiment_metadata = {}
#
#     # Write the data to HDF5
#     write_results_hdf5(filepath=output_filepath,
#                        groupath=output_grouppath,
#                        output_data=output_data,
#                        scan_metadata=scan_metadata,
#                        experiment_metadata=experiment_metadata)

# def write_results_hdf5(filepath,
#                        groupath,
#                        output_data,
#                        scan_meta=None,
#                        experiment_meta=None):
#     """
#     Write the results to file
#
#     :param filepath: The path to the HDF5 file ot be used
#     :param groupath: The path to the group to be used
#     :param output_data: Dictionary of the output data. Keys are dataset names and values are numpy data arrays.
#     :param scan_meta: Dictionary of scan metadata. Keys are dataset names and values are numpy data arrays.
#     :param experiment_meta: Dictionary of the experiment metadata. Keys are dataset names and values are
#            numpy data arrays.
#
#     """
#     # Open the file
#     infile = h5py.File(filepath, 'a')
#     # Get the target group or create the required group(s) if necessary
#     try:
#         target = infile[groupath]
#     except KeyError:
#         current_group = infile['/']   # Split the path into the hiearchy of subgroups to be used.
#         # Creat all subgroups
#         for subgroup in groupath.split('/'):
#             current_group = current_group.require_group(subgroup)
#         target = current_group
#     # Write all data arrays, one after the other
#     for key, value in output_data.iteritems():
#         target[key] = value
#         infile.flush()
#     if scan_meta is not None:
#         scan_group = target.require_group('scan_metadata')
#         for key, value in scan_meta.iteritems():
#             scan_group[key] = value
#             infile.flush()
#     if experiment_meta is not None:
#         experiment_group = target.require_group('experiment_metadata')
#         for key, value in experiment_meta.iteritems():
#             experiment_group[key] = value
#             infile.flush()
#
#     infile.close()


#!python
"""
Score spectra/scans against a collection of molecular fragmentation trees.

"""
__authors__ = 'Curt R. Fischer, Oliver Ruebel, Benjamin P. Bowen'
__copyright__ = 'Lawrence Berkeley National Laboratory and Authors, 2015.  All rights currently reserved.'


# standard libraries
import re
import os

# numpy and scipy
import numpy as np
from scipy.stats import norm
from scipy.sparse import csc_matrix

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
    Score the the plausibility that a given MS2 (or MSn) spectrum arose from a given compound.

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
        nodes_to_keep = np.where[node_depths <= max_depth][0]
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
    :return:            tree_files_by_ms1_mass, a numpy structured array with columns (i) filename and (ii) MS1 mass
    """
    # check arguments and decide to read from supplied path for from tree_file list
    if tree_files is None and path is None:
        raise ValueError('tree_files and path cannot both be None.')

    if tree_files is None:
        import glob
        tree_files = glob.glob(os.path.join(path, '*.h5'))

    # initialize result table
    num_files = len(tree_files)
    dtype = [('filename', 'a400'),
             ('ms1_mass', 'f8'), ]
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


def score_peakcube_against_trees(peakcube, peakmzs, ms1_mz, params):
    """
    Create a score cube of MIDAS scores for each

    :param peakcube:          numpy ndarray of floats   Stores peak intensities.  Has shape (nx, ..., n_peaks).
                                                        i.e., the last axis is a peak index, and prior axes are
                                                        an arbitrary number of spatial or time coordinates

    :param peakmzs:           numpy array of floats     Stores fragment peak m/z values.  Has shape (n_peaks,)
    :param ms1_mz:            float                     the (unneutralized) MS1 mass of the precursor ions.
    :param params: dictionary containing other required parameters

            * ``file_lookup_table``: full path to a .npy file having a numpy structured array with columns \
                                     (i) filename and (ii) MS1 mass. Alternatively this may also be the numpy \
                                     array directly.
            * ``ms1_mass_tol``       float, max. diff. in Da of tree_parent_mass & MS1_precursor_mass
            * ``ms2_mass_tol``       float, max. mass in Da by which two MS2/MSn peaks can differ
            * ``neutralizations``:   list of floats, adjustments (in Da) added to data peaks in order to neutralize them
            * ``max_depth``          int, optional.  For restricting scoring to lower max depths than \
                                    are present in the supplied tree

    :return: score_cube       a numpy ndarray of shape (nx, ..., len(file_lookup_table))

    Unlike score_scan_list_against_trees, this function is designed to work on numpy arrays of spectra.  It is more
     appropriate for imaging data where:
        1. an common MS1 precursor has been fragmented and scanned many times/places, as long as
        2. a global peak finder has been run on all spectra, so peak intensities exist for every peak at every pixel
    """
    # TODO: add support for want_match_matrix
    # unpack parameters
    neutralizations = params['neutralizations']
    ms2_mass_tol = params['ms2_mass_tol']
    ms1_mass_tol = params['ms1_mass_tol']
    if isinstance(params['file_lookup_table'], basestring):
        file_lookup_table = np.load(params['file_lookup_table'])
    elif isinstance(params['file_lookup_table'], np.ndarray):
        file_lookup_table = params['file_lookup_table']
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

    # figure files to score
    file_idxs = []
    for ion_loss in neutralizations:
        ms1_mass = ms1_mz + ion_loss
        mass_difference = np.abs(file_lookup_table['ms1_mass'] - ms1_mass)
        new_indices = np.where(mass_difference <= ms1_mass_tol)
        file_idxs = np.append(file_idxs, new_indices)

    # score selected files against every spectrum in data
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
            score_cube[i, idx] = calculate_MIDAS_score(mz_intensity_arr,
                                                       tree,
                                                       mass_tol=ms2_mass_tol,
                                                       neutralizations=neutralizations)

    # reshape output and return
    tuple_of_tuples = (spatial_dimensions, (n_compounds,))
    score_cube_shape = sum(tuple_of_tuples, ())  # http://stackoverflow.com/a/3205524/4480692

    return score_cube.reshape(score_cube_shape)


def score_scan_list_against_trees(scan_list, ms1_mz, params):
    """
    Create a score cube of MIDAS scores for each

    :param scan_list:         list of numpy ndarrays    Stores peak mzs and intensities.  Each list el has shape of
                                                        (n_peaks, 2), with mzs in column 1 an intensities in col 2.
                                                        Length of list is n_scans long
    :param ms1_mz:            list of floats            The (unneutralized) MS1 mz/s of the precursor ions.
                                                        Must have same length as scan_list, i.e. len(ms1_ms) = num_scans

    :param params: dictionary containing other required parameters

            * ``file_lookup_table``: full path to a .npy file having a numpy structured array with columns \
                                     (i) filename and (ii) MS1 mass. Alternatively, this may also be the \
                                     numpy array directly.
            * ``ms1_mass_tol``       float, max. diff. in Da of tree_parent_mass & MS1_precursor_mass
            * ``ms2_mass_tol``       float, max. mass in Da by which two MS2/MSn peaks can differ
            * ``neutralizations``:   list of floats, adjustments (in Da) added to data peaks in order to neutralize them
            * ``max_depth``          int, optional.  For restricting scoring to lower max depths than \
                                    are present in the supplied tree

    :return: score_matrix     a numpy ndarray of shape (n_scans, len(file_lookup_table))

    Unlike score_peakcube_against_trees, this function is designed to work on _lists_ of spectra.  It is more
     appropriate for spectra directly extracted from mzML files or for centroided data.  This function does NOT
     assume that each scan in the list has the same precursors.
    """
    # TODO: add support for 'want_match_matrix' parameter
    # TODO: return a sparse matrix?
    # unpack parameters
    neutralizations = params['neutralizations']
    ms2_mass_tol = params['ms2_mass_tol']
    ms1_mass_tol = params['ms1_mass_tol']
    if isinstance(params['file_lookup_table'], basestring):
        file_lookup_table = np.load(params['file_lookup_table'])
    elif isinstance(params['file_lookup_table'], np.ndarray):
        file_lookup_table = params['file_lookup_table']
    else:
        raise ValueError('Invalid file_lookup_table specified')

    # size input variables
    n_scans = len(scan_list)
    n_compounds = len(file_lookup_table)

    # initialize output variable score_matrix
    score_matrix = np.zeros(shape=(n_scans, n_compounds), dtype=float)

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

        # score selected files against every spectrum in data
        for idx in file_idxs:
            filename = file_lookup_table[idx]['filename']
            file_reader = h5py.File(filename)
            group_key = file_reader.keys()[0]
            data_key = file_reader[group_key].keys()[0]
            tree = file_reader[group_key][data_key][:]

            score_matrix[i, idx] = calculate_MIDAS_score(scan,
                                                         tree,
                                                         mass_tol=ms2_mass_tol,
                                                         neutralizations=neutralizations)
    return score_matrix


def make_pactolus_hit_table(pactolus_results, table_file, original_db, return_list=True):
    """
    Makes a hit table in the same format as lbl-midas for comparison of the two algorithms

    :param pactolus_results: ndarray,    n_compounds by n_spectra matrix of pactolus scores
    :param table_file:      string,     full path to .npy file containing tree file names and parent masses that
                                            was used to generate pactolus results.  Or the numpy array directly.
    :param original_db:     string,     full path to flat text file containing the molecule DB used to generate the
                        fragmentation trees. The primary use for this is to enable lookup of molecule names.
    :param return_list:     bool, whether to return a list of hit tables
    :return: hit_table_list, a list of hit_tables
    """
    # transform pactolus results into hit table
    global HIT_TABLE_DTYPE
    db_arr = crossref_to_db(table_file, original_db)

    # return a list of hit_tables when pactolus_results is a score_list
    if return_list:
        hit_table_list = []
        for scores in pactolus_results:
            num_nonzero = np.count_nonzero(scores)
            hit_table = np.zeros(shape=(num_nonzero), dtype=HIT_TABLE_DTYPE)
            order = scores.argsort()[::-1][:num_nonzero]
            for idx, hit_index in enumerate(order):
                hit_table[idx] = (scores[hit_index],
                                  db_arr['metacyc_id'][hit_index],
                                  db_arr['name'][hit_index],
                                  db_arr['mass'][hit_index],
                                  0,  # TODO is the number of peaks not set?
                                  0,  # TODO is the number of matched peaks not set?
                                  )
            hit_table_list.append(hit_table)
        assert len(hit_table_list) == pactolus_results.shape[0]
        return hit_table_list
    if not return_list:
        raise NotImplementedError
        # TODO: implement


def crossref_to_db(table_file, original_db):
    """
    Cross-references pactolus results to name, structure,  etc. data in a flat text file

    :param table_file:      string,     full path to .npy file containing tree file names and parent masses that
                                            was used to generate pactolus results. Or the numpy array directly.
    :param original_db:     string,     full path to flat text file containing DB
    :return db_arr:         numpy structured array re-ordered to match table_file, with fields:
                                                    'metacyc_id',
                                                    'name'
                                                    'inchi'
                                                    'lins',
                                                    'inchi_key'
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
                print fields[2], mol, idx, line
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

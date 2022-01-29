# -*- coding: utf-8 -*-

import sys
import tqdm
from multiprocessing import cpu_count, Pool
from pathlib import PurePath
from typing import Callable, Dict, List, Union, Tuple
import numpy as np

from PIL import Image
from duplib.search.bktree import BKTree
from duplib.search.brute_force import BruteForce
from duplib.search.brute_force_cython import BruteForceCython

IMG_FORMATS = ['JPEG', 'PNG', 'BMP', 'MPO', 'PPM', 'TIFF', 'GIF', 'WEBP']

def get_files_to_remove(duplicates: Dict[str, List]) -> List:
    """
    Get a list of files to remove.
    Args:
        duplicates: A dictionary with file name as key and a list of duplicate file names as value.
    Returns:
        A list of files that should be removed.
    """
    # iterate over dict_ret keys, get value for the key and delete the dict keys that are in the value list
    files_to_remove = set()

    for k, v in duplicates.items():
        tmp = [
            i[0] if isinstance(i, tuple) else i for i in v
        ]  # handle tuples (image_id, score)

        if k not in files_to_remove:
            files_to_remove.update(tmp)

    return list(files_to_remove)
    
def parallelise(function: Callable, data: List) -> List:
    pool = Pool(processes=cpu_count())
    results = list(
        pool.imap(function, data, 100)
    )
    pool.close()
    pool.join()
    return results
    
    
def preprocess_image(
    image, target_size: Tuple[int, int] = None, grayscale: bool = False
) -> np.ndarray:
    """
    Take as input an image as numpy array or Pillow format. Returns an array version of optionally resized and grayed
    image.
    Args:
        image: numpy array or a pillow image.
        target_size: Size to resize the input image to.
        grayscale: A boolean indicating whether to grayscale the image.
    Returns:
        A numpy array of the processed image.
    """
    if isinstance(image, np.ndarray):
        image = image.astype('uint8')
        image_pil = Image.fromarray(image)

    elif isinstance(image, Image.Image):
        image_pil = image
    else:
        raise ValueError('Input is expected to be a numpy array or a pillow object!')

    if target_size:
        image_pil = image_pil.resize(target_size, Image.ANTIALIAS)

    if grayscale:
        image_pil = image_pil.convert('L')

    return np.array(image_pil).astype('uint8')
    
def check_image_array_hash(image_arr: np.ndarray) -> None:
    """
    Checks the sanity of the input image numpy array for hashing functions.
    Args:
        image_arr: Image array.
    """
    image_arr_shape = image_arr.shape
    if len(image_arr_shape) == 3:
        _check_3_dim(image_arr_shape)
    elif len(image_arr_shape) > 3 or len(image_arr_shape) < 2:
        _raise_wrong_dim_value_error(image_arr_shape)
    
def load_image(
    image_file: Union[PurePath, str],
    target_size: Tuple[int, int] = None,
    grayscale: bool = False,
    img_formats: List[str] = IMG_FORMATS,
) -> np.ndarray:
    """
    Load an image given its path. Returns an array version of optionally resized and grayed image. Only allows images
    of types described by img_formats argument.
    Args:
        image_file: Path to the image file.
        target_size: Size to resize the input image to.
        grayscale: A boolean indicating whether to grayscale the image.
        img_formats: List of allowed image formats that can be loaded.
    """
    try:
        img = Image.open(image_file)

        # validate image format
        if img.format not in img_formats:
            print(f'Invalid image format {img.format}!')
            return None

        else:
            if img.mode != 'RGB':
                # convert to RGBA first to avoid warning
                # we ignore alpha channel if available
                img = img.convert('RGBA').convert('RGB')

            img = preprocess_image(img, target_size=target_size, grayscale=grayscale)

            return img

    except Exception as e:
        print(f'Invalid image file {image_file}:\n{e}')
        return None
        
class HashEval:
    def __init__(
        self,
        test: Dict,
        queries: Dict,
        distance_function: Callable,
        verbose: bool = True,
        threshold: int = 5,
        search_method: str = 'brute_force_cython' if not sys.platform == 'win32' else 'bktree',
    ) -> None:
        """
        Initialize a HashEval object which offers an interface to control hashing and search methods for desired
        dataset. Compute a map of duplicate images in the document space given certain input control parameters.
        """
        self.test = test  # database
        self.queries = queries
        self.distance_invoker = distance_function
        self.verbose = verbose
        self.threshold = threshold
        self.query_results_map = None

        if search_method == 'bktree':
            self._fetch_nearest_neighbors_bktree()
        elif search_method == 'brute_force':
            self._fetch_nearest_neighbors_brute_force()
        else:
            self._fetch_nearest_neighbors_brute_force_cython()

    def _searcher(self, data_tuple) -> None:
        """
        Perform search on a query passed in by _get_query_results multiprocessing part.
        Args:
            data_tuple: Tuple of (query_key, query_val, search_method_object, thresh)
        Returns:
           List of retrieved duplicate files and corresponding hamming distance for the query file.
        """
        query_key, query_val, search_method_object, thresh = data_tuple
        res = search_method_object.search(query=query_val, tol=thresh)
        res = [i for i in res if i[0] != query_key]  # to avoid self retrieval
        return res

    def _get_query_results(
        self, search_method_object: Union[BruteForce, BKTree]
    ) -> None:
        """
        Get result for the query using specified search object. Populate the global query_results_map.
        Args:
            search_method_object: BruteForce or BKTree object to get results for the query.
        """
        args = list(
            zip(
                list(self.queries.keys()),
                list(self.queries.values()),
                [search_method_object] * len(self.queries),
                [self.threshold] * len(self.queries),
            )
        )
        result_map_list = parallelise(self._searcher, args)
        result_map = dict(zip(list(self.queries.keys()), result_map_list))

        self.query_results_map = {
            k: [i for i in sorted(v, key=lambda tup: tup[1], reverse=False)]
            for k, v in result_map.items()
        }  # {'filename.jpg': [('dup1.jpg', 3)], 'filename2.jpg': [('dup2.jpg', 10)]}

    def _fetch_nearest_neighbors_brute_force(self) -> None:
        """
        Wrapper function to retrieve results for all queries in dataset using brute-force search.
        """
        #print('Start: Retrieving duplicates using Brute force algorithm')
        brute_force = BruteForce(self.test, self.distance_invoker)
        self._get_query_results(brute_force)
        #print('End: Retrieving duplicates using Brute force algorithm')

    def _fetch_nearest_neighbors_brute_force_cython(self) -> None:
        """
        Wrapper function to retrieve results for all queries in dataset using brute-force search.
        """
        #print('Start: Retrieving duplicates using Cython Brute force algorithm')
        brute_force_cython = BruteForceCython(self.test, self.distance_invoker)
        self._get_query_results(brute_force_cython)
        #print('End: Retrieving duplicates using Cython Brute force algorithm')

    def _fetch_nearest_neighbors_bktree(self) -> None:
        """
        Wrapper function to retrieve results for all queries in dataset using a BKTree search.
        """
        #print('Start: Retrieving duplicates using BKTree algorithm')
        built_tree = BKTree(self.test, self.distance_invoker)  # construct bktree
        self._get_query_results(built_tree)
        #print('End: Retrieving duplicates using BKTree algorithm')

    def retrieve_results(self, scores: bool = False) -> Dict:
        """
        Return results with or without scores.
        Args:
            scores: Boolean indicating whether results are to eb returned with or without scores.
        Returns:
            if scores is True, then a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
            if scores is False, then a dictionary of the form {'image1.jpg': ['image1_duplicate1.jpg',
            'image1_duplicate2.jpg'], 'image2.jpg':['image1_duplicate1.jpg',..], ..}
        """
        if scores:
            return self.query_results_map
        else:
            return {k: [i[0] for i in v] for k, v in self.query_results_map.items()}
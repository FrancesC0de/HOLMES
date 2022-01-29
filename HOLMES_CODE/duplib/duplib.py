# -*- coding: utf-8 -*-

import os
import sys
from pathlib import PurePath, Path
from typing import Dict, List, Optional
import numpy as np
from scipy.fftpack import dct
from duplib.utils import *

class Hashing:
    def __init__(self) -> None:
        """
        Initialize hashing class.
        """
        self.target_size = (8, 8)  # resizing to dims

    @staticmethod
    def hamming_distance(hash1: str, hash2: str) -> float:
        """
        Calculate the hamming distance between two hashes. If length of hashes is not 64 bits, then pads the length
        to be 64 for each hash and then calculates the hamming distance.
        Args:
            hash1: hash string
            hash2: hash string
        Returns:
            hamming_distance: Hamming distance between the two hashes.
        """
        hash1_bin = bin(int(hash1, 16))[2:].zfill(
            64
        )  # zfill ensures that len of hash is 64 and pads MSB if it is < A
        hash2_bin = bin(int(hash2, 16))[2:].zfill(64)
        return np.sum([i != j for i, j in zip(hash1_bin, hash2_bin)])

    @staticmethod
    def _array_to_hash(hash_mat: np.ndarray) -> str:
        """
        Convert a matrix of binary numerals to 64 character hash.
        Args:
            hash_mat: A numpy array consisting of 0/1 values.
        Returns:
            An hexadecimal hash string.
        """
        return ''.join('%0.2x' % x for x in np.packbits(hash_mat))

    def encode_image(
        self, image_file=None, image_array: Optional[np.ndarray] = None
    ) -> str:
        """
        Generate hash for a single image.
        Args:
            image_file: Path to the image file.
            image_array: Optional, used instead of image_file. Image typecast to numpy array.
        Returns:
            hash: A 16 character hexadecimal string hash for the image.
        """
        try:
            if image_file and os.path.exists(image_file):
                image_file = Path(image_file)
                image_pp = load_image(
                    image_file=image_file, target_size=self.target_size, grayscale=True
                )

            elif isinstance(image_array, np.ndarray):
                check_image_array_hash(image_array)  # Do sanity checks on array
                image_pp = preprocess_image(
                    image=image_array, target_size=self.target_size, grayscale=True
                )
            else:
                raise ValueError
        except (ValueError, TypeError):
            raise ValueError('Please provide either image file path or image array!')

        return self._hash_func(image_pp) if isinstance(image_pp, np.ndarray) else None

    def encode_images(self, image_dir=None):
        """
        Generate hashes for all images in a given directory of images.
        Args:
            image_dir: Path to the image directory.
        Returns:
            dictionary: A dictionary that contains a mapping of filenames and corresponding 64 character hash string
                        such as {'Image1.jpg': 'hash_string1', 'Image2.jpg': 'hash_string2', ...}
        """
        if not os.path.isdir(image_dir):
            raise ValueError('Please provide a valid directory path!')

        image_dir = Path(image_dir)
        
        files = [
            i.absolute() for i in image_dir.glob('*') if not i.name.startswith('.')
        ]  # ignore hidden files

        print(f'Start: Calculating hashes...')

        hashes = parallelise(self.encode_image, files)
        hash_initial_dict = dict(zip([f.name for f in files], hashes))
        hash_dict = {
            k: v for k, v in hash_initial_dict.items() if v
        }  # To ignore None (returned if some probelm with image file)

        print(f'End: Calculating hashes!')
        return hash_dict

    def _hash_algo(self, image_array: np.ndarray):
        pass

    def _hash_func(self, image_array: np.ndarray):
        hash_mat = self._hash_algo(image_array)
        return self._array_to_hash(hash_mat)

    # search part

    @staticmethod
    def _check_hamming_distance_bounds(thresh: int) -> None:
        """
        Check if provided threshold is valid. Raises TypeError if wrong threshold variable type is passed or a
        ValueError if an out of range value is supplied.
        Args:
            thresh: Threshold value (must be int between 0 and 64)
        Raises:
            TypeError: If wrong variable type is provided.
            ValueError: If invalid value is provided.
        """
        if not isinstance(thresh, int):
            raise TypeError('Threshold must be an int between 0 and 64')
        elif thresh < 0 or thresh > 64:
            raise ValueError('Threshold must be an int between 0 and 64')
        else:
            return None

    def _find_duplicates_dict(
        self,
        encoding_map: Dict[str, str],
        max_distance_threshold: int = 10,
        search_method: str = 'brute_force_cython' if not sys.platform == 'win32' else 'bktree',
    ) -> Dict:
        """
        Take in dictionary {filename: encoded image}, detects duplicates below the given hamming distance threshold
        and returns a dictionary containing key as filename and value as a list of duplicate filenames.
        Args:
            encoding_map: Dictionary with keys as file names and values as encoded images (hashes).
            max_distance_threshold: Hamming distance between two images below which retrieved duplicates are valid.
            search_method: Algorithm used to retrieve duplicates. Default is brute_force_cython for Unix else bktree.
        Returns:
            a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
        """
        print('Start: Evaluating hamming distances for getting duplicates')
        result_set = HashEval(
            test=encoding_map,
            queries=encoding_map,
            distance_function=self.hamming_distance,
            threshold=max_distance_threshold,
            search_method=search_method,
        )

        print('End: Evaluating hamming distances for getting duplicates')

        self.results = result_set.retrieve_results()
        return self.results

    def _find_duplicates_dir(
        self,
        image_dir: PurePath,
        max_distance_threshold: int = 10,
        search_method: str = 'brute_force_cython' if not sys.platform == 'win32' else 'bktree',
    ) -> Dict:
        """
        Take in path of the directory in which duplicates are to be detected below the given hamming distance
        threshold. Returns dictionary containing key as filename and value as a list of duplicate file names.
        Args:
            image_dir: Path to the directory containing all the images.
            max_distance_threshold: Hamming distance between two images below which retrieved duplicates are valid.
            search_method: Algorithm used to retrieve duplicates. Default is brute_force_cython for Unix else bktree.
        Returns:
            a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
            score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}
        """
        encoding_map = self.encode_images(image_dir)
        results = self._find_duplicates_dict(
            encoding_map=encoding_map,
            max_distance_threshold=max_distance_threshold,
            search_method=search_method,
        )
        return results

    def find_duplicates(
        self,
        image_dir: PurePath = None,
        max_distance_threshold: int = 10,
        search_method: str = 'brute_force_cython' if not sys.platform == 'win32' else 'bktree',
    ) -> Dict:
        """
        Find duplicates for each file. Takes in path of the directory or encoding dictionary in which duplicates are to
        be detected. All images with hamming distance less than or equal to the max_distance_threshold are regarded as
        duplicates. Returns dictionary containing key as filename and value as a list of duplicate file names.
        Optionally, the below the given hamming distance could be returned instead of just duplicate filenames for each
        query file.
        Args:
            image_dir: Path to the directory containing all the images or dictionary with keys as file names
                       and values as hash strings for the key image file.
            max_distance_threshold: Optional, hamming distance between two images below which retrieved duplicates are
                                    valid. (must be an int between 0 and 64). Default is 10.
            search_method: Algorithm used to retrieve duplicates. Default is brute_force_cython for Unix else bktree.
        Returns:
            duplicates dictionary: a dictionary of the form {'image1.jpg': [('image1_duplicate1.jpg',
                        score), ('image1_duplicate2.jpg', score)], 'image2.jpg': [] ..}.
        """
        self._check_hamming_distance_bounds(thresh=max_distance_threshold)
        result = self._find_duplicates_dir(
            image_dir=image_dir,
            max_distance_threshold=max_distance_threshold,
            search_method=search_method,
        )
        return result

    def find_duplicates_to_remove(
        self,
        image_dir: PurePath = None,
        max_distance_threshold: int = 10,
    ) -> List:
        """
        Give out a list of image file names to remove based on the hamming distance threshold threshold. Does not
        remove the mentioned files.
        Args:
            image_dir: Path to the directory containing all the images or dictionary with keys as file names
                       and values as hash strings for the key image file.
            max_distance_threshold: Optional, hamming distance between two images below which retrieved duplicates are
                                    valid. (must be an int between 0 and 64). Default is 10.
        Returns:
            duplicates: List of image file names that are found to be duplicate of me other file in the directory.
        """
        result = self.find_duplicates(
            image_dir=image_dir,
            max_distance_threshold=max_distance_threshold,
        )
        files_to_remove = get_files_to_remove(result)
        return files_to_remove


class PHash(Hashing):
    def __init__(self) -> None:
        """
        Initialize perceptual hashing class.
        """
        self.__coefficient_extract = (8, 8)
        self.target_size = (32, 32)

    def _hash_algo(self, image_array):
        """
        Get perceptual hash of the input image.
        Args:
            image_array: numpy array that corresponds to the image.
        Returns:
            A string representing the perceptual hash of the image.
        """
        dct_coef = dct(dct(image_array, axis=0), axis=1)

        # retain top left 8 by 8 dct coefficients
        dct_reduced_coef = dct_coef[
            : self.__coefficient_extract[0], : self.__coefficient_extract[1]
        ]

        # median of coefficients excluding the DC term (0th term)
        median_coef_val = np.median(np.ndarray.flatten(dct_reduced_coef)[1:])

        # return mask of all coefficients greater than mean of coefficients
        hash_mat = dct_reduced_coef >= median_coef_val
        return hash_mat

def delete_duplicates(image_dir, hamming_th):
  phasher = PHash()
  duplicates_list = phasher.find_duplicates_to_remove(image_dir=image_dir, max_distance_threshold=hamming_th)
  for duplicate in duplicates_list:
    dup = os.path.join(image_dir, duplicate)
    if os.path.exists(dup):
      os.remove(dup)
    else:
      raise Exception("Duplicate " + duplicate + " not found.")
  return duplicates_list
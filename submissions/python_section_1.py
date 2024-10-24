from typing import Dict, List, Any
import math
import pandas as pd
from itertools import permutations
import re
import polyline


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:

    n_lst = []
    index = 0
    
    # Iterating over the list in chunks of size n
    for j in range(math.ceil(len(lst) / n)):
        # Taking the current chunk (up to n elements) manually
        chunk = []
        chunk_size = min(n, len(lst) - index)  # Handling the case where remaining elements are fewer than n
        
        # Manually adding elements to chunk
        for i in range(chunk_size):
            chunk.append(lst[index + i])
        
        # Manually reverse the chunk and add to the final list
        for i in range(chunk_size - 1, -1, -1):
            n_lst.append(chunk[i])
        
        # Moving the index forward by the size of the processed chunk
        index += chunk_size
    
    lst = n_lst
    return lst


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    new_dict = {}  # Initializing an empty dictionary to hold the groups

    # Iterating over each string in the input list
    for string in lst:
        length = len(string)  # Get the length of the current string

        # If the length is not in the dictionary, initializing it with an empty list
        if length not in new_dict:
            new_dict[length] = []

        # Appending the string to the corresponding length's list
        new_dict[length].append(string)

    # dictionary sorted by keys (lengths)
    return dict(sorted(new_dict.items()))


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    items = {}  # Initializing an empty dictionary to hold the flattened items

    def flatten(current_dict: Dict[str, Any], parent_key: str = ''):
        for key, value in current_dict.items():
            # Creating a new key by concatenating the parent key and the current key
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            
            if isinstance(value, dict):
                # Recursively flatten the dictionary
                flatten(value, new_key)
            elif isinstance(value, list):
                # Handling lists by enumerating their items
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        # If the list item is a dict, flatten it recursively
                        flatten(item, f"{new_key}[{i}]")
                    else:
                        # Otherwise, add the item directly
                        items[f"{new_key}[{i}]"] = item
            else:
                # If the value is neither a dict nor a list, add it to the items
                items[new_key] = value

    flatten(nested_dict)  # Start the flattening process
    return items


def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    return list(set(permutations(nums)))


def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    
    # Define the regex pattern for 'dd-mm-yyyy', 'mm/dd/yyyy', and 'yyyy.mm.dd'
    pattern = r"\b(\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2})\b"
    
    # Use re.findall() to find all matches for the date pattern
    dates = re.findall(pattern, text)
    
    return dates



def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points
    on the Earth's surface given their latitude and longitude using the Haversine formula.
    
    Args:
        lat1, lon1 (float): Latitude and longitude of the first point.
        lat2, lon2 (float): Latitude and longitude of the second point.
    
    Returns:
        float: Distance between two points in meters.
    """
    R = 6371000  # Earth radius in meters
    
    # latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Difference in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Distance in meters
    return R * c

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    # Step 1: Decoding the polyline string into a list of (latitude, longitude) tuples
    coordinates = polyline.decode(polyline_str)
    
    # Step 2: Creating a DataFrame from the coordinates list
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Step 3: Add a 'distance' column initialized to 0.0 for now
    df['distance'] = 0.0
    
    # Step 4: Calculating distances for successive rows using the Haversine formula
    for i in range(1, len(df)):
        # Extracting latitude and longitude for the current and previous points
        lat1, lon1 = df.loc[i - 1, ['latitude', 'longitude']]
        lat2, lon2 = df.loc[i, ['latitude', 'longitude']]
        
        # Calculating the distance between successive points and store it in the DataFrame
        df.loc[i, 'distance'] = haversine(lat1, lon1, lat2, lon2)
    
    # Step 5: DataFrame with latitude, longitude, and distance columns
    return df



def rotate_and_sum_matrix(matrix: List[List[int]]) -> List[List[int]]:
    n = len(matrix)
    
    # Step 1: Rotating the matrix by 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - i - 1] = matrix[i][j]
    
    # Step 2: Replacing each element with the sum of all elements in the same row and column, excluding itself
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum
    
    return final_matrix


def time_check(df: pd.DataFrame) -> pd.Series:
    # List of all days of the week
    all_days = {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'}
    
    # Initializing an empty result series to store results for each (id, id_2) pair
    result = pd.Series(dtype=bool)
    
    # Group the data by 'id' and 'id_2'
    grouped = df.groupby(['id', 'id_2'])
    
    # Loop through each group of unique (id, id_2) pairs
    for (id_val, id_2_val), group in grouped:
        # Extracting the set of days covered by the startDay and endDay
        days_covered = set(group['startDay']).union(set(group['endDay']))
        
        # Convert 'startTime' and 'endTime' to datetime objects for comparison
        group['startTime'] = pd.to_datetime(group['startTime'], format='%H:%M:%S').dt.time
        group['endTime'] = pd.to_datetime(group['endTime'], format='%H:%M:%S').dt.time
        
        # Check if the group covers all 7 days and the full 24-hour time range
        full_day_coverage = (
            group['startTime'].min() == pd.to_datetime('00:00:00', format='%H:%M:%S').time() and
            group['endTime'].max() == pd.to_datetime('23:59:59', format='%H:%M:%S').time()
        )
        
        # Check if the days cover the entire week
        full_week_coverage = days_covered == all_days
        
        # If both time and day coverage are complete, mark as False (correct), otherwise True (incomplete)
        result[(id_val, id_2_val)] = not (full_day_coverage and full_week_coverage)
    
    return result





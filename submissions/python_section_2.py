import pandas as pd
import numpy as np
from datetime import time

def calculate_distance_matrix(df) -> pd.DataFrame:
    # Get a list of all unique toll location IDs
    toll_locations = pd.unique(df[['id_start', 'id_end']].values.ravel('K'))

    # Initializing an empty distance matrix filled with infinity
    distance_matrix = pd.DataFrame(np.inf, index=toll_locations, columns=toll_locations)

    # Fill in the distances for the direct routes
    for _, row in df.iterrows():
        distance_matrix.loc[row['id_start'], row['id_end']] = row['distance']
        distance_matrix.loc[row['id_end'], row['id_start']] = row['distance']  # Ensure symmetry

    # Set the diagonal to 0 (distance from a location to itself)
    np.fill_diagonal(distance_matrix.values, 0)

    # Applying Floyd-Warshall algorithm to compute shortest paths
    for k in toll_locations:
        for i in toll_locations:
            for j in toll_locations:
                distance_matrix.loc[i, j] = min(distance_matrix.loc[i, j],
                                                distance_matrix.loc[i, k] + distance_matrix.loc[k, j])

    return distance_matrix


def unroll_distance_matrix(df)->pd.DataFrame:
    # Unstack the matrix into long form (id_start, id_end, distance)
    unrolled_df = df.stack().reset_index()
    unrolled_df.columns = ['id_start', 'id_end', 'distance']
    
    # Filter out rows where id_start equals id_end
    unrolled_df = unrolled_df[unrolled_df['id_start'] != unrolled_df['id_end']]

    return unrolled_df



def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame:
    # Filter the DataFrame for the reference id_start
    reference_distances = df[df['id_start'] == reference_id]['distance']
    
    # Calculating the average distance for the reference id_start
    reference_avg = reference_distances.mean()
    
    # Calculate 10% threshold (lower and upper bounds)
    lower_bound = reference_avg * 0.9
    upper_bound = reference_avg * 1.1
    
    # Initializing an empty DataFrame to store results
    result_df = pd.DataFrame(columns=df.columns)
    
    # Loop through unique id_start values and check their averages
    for id_start in df['id_start'].unique():
        id_start_avg = df[df['id_start'] == id_start]['distance'].mean()
        
        if lower_bound <= id_start_avg <= upper_bound:
            # Append the matching rows for this id_start to the result DataFrame
            result_df = pd.concat([result_df, df[df['id_start'] == id_start]], ignore_index=True)
    
    return result_df


def calculate_toll_rate(df: pd.DataFrame) -> pd.DataFrame:
    
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle_type, rate in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate

    return df



def calculate_time_based_toll_rates(df) -> pd.DataFrame:
    # discount factors
    discount_factors = {
        'weekday': {
            (time(0, 0), time(10, 0)): 0.8,
            (time(10, 0), time(18, 0)): 1.2,
            (time(18, 0), time(23, 59, 59)): 0.8
        },
        'weekend': 0.7
    }
    
    # days of the week
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    # list to hold the new rows
    new_rows = []
    
    # Iterate through each unique (id_start, id_end) pair
    for _, row in df.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        
        for day in days_of_week:
            # For each day, creating time entries for the full 24 hours
            for hour in range(24):
                current_time = time(hour, 0)
                
                # Determine the discount factor based on the day and time
                if day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                    # Weekday logic
                    for time_range, factor in discount_factors['weekday'].items():
                        if time_range[0] <= current_time < time_range[1]:
                            discount = factor
                            break
                    else:
                        discount = 1  # Default to no discount if not in specified ranges
                else:
                    # Weekend logic
                    discount = discount_factors['weekend']
                
                # Create a new row with adjusted values
                new_row = {
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': row['distance'] * discount,
                    'moto': row['moto'] * discount,
                    'car': row['car'] * discount,
                    'rv': row['rv'] * discount,
                    'bus': row['bus'] * discount,
                    'truck': row['truck'] * discount,
                    'start_day': day,
                    'start_time': current_time,
                    'end_day': day,
                    'end_time': current_time
                }
                
                new_rows.append(new_row)
    
    # Creating a new DataFrame from the new rows
    new_df = pd.DataFrame(new_rows)
    
    return new_df
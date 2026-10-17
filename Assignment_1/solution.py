SURVEY_YEARS: list[int] = [2016, 2019, 2020, 2022, 2024]
POVERTY_YEARS: list[int] = [2019, 2020, 2022, 2024]
STATES: list[str] = [
    "Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan", "Pahang",
    "Perak", "Perlis", "Pulau Pinang", "Sabah", "Sarawak", "Selangor",
    "Terengganu", "W.P. Kuala Lumpur", "W.P. Labuan", "W.P. Putrajaya"
]
REGIONS: dict[str, list[str]] = {
    "Northern": ["Perlis", "Kedah", "Pulau Pinang", "Perak"],
    "Central": ["Selangor", "W.P. Kuala Lumpur", "W.P. Putrajaya", "Negeri Sembilan"],
    "Southern": ["Melaka", "Johor"],
    "East Coast": ["Kelantan", "Terengganu", "Pahang"],
    "East Malaysia": ["Sabah", "Sarawak", "W.P. Labuan"],
}

def get_group_states(matric_numbers: list[int], states: list[str]):
    ## Owner: Integration and test lead (role 5). Reviewed by: Data steward (role 1).
    if not states or len(states) != 16:
        raise ValueError("The states list must contain exactly 16 states.")
    
    # 1. Pastikan senarai negeri disisih mengikut abjad
    sorted_states = sorted(states)
    
    assigned_states: list[str] = []
    
    # 2. Proses nombor matrik mengikut urutan menaik
    for matric in sorted(matric_numbers):
        # Kira indeks baki pembahagian modulo 16
        idx = int(matric) % len(sorted_states)
        
        # 3. Sekiranya negeri sudah diambil, cari negeri seterusnya (wrap around)
        while sorted_states[idx] in assigned_states:
            idx = (idx + 1) % len(sorted_states)
            
        assigned_states.append(sorted_states[idx])
        
    return assigned_states

def validate_records(records: list[dict], required_keys: tuple[str, ...]) -> None:
    """
    Raise an informative exception if records are not a usable list of keyed records.

    Owner: Data steward (role 1). Reviewed by: Structures engineer (role 2).
    """
    if not isinstance(records, list):
        raise TypeError(f"Expected records to be a list, got {type(records).__name__}.")
    if not records:
        raise ValueError("Records list is empty.")

    seen_keys: set[tuple[str, str]] = set()

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise TypeError(f"Record at index {idx} is not a dictionary.")

        for req_key in required_keys:
            if req_key not in rec:
                raise KeyError(f"Record at index {idx} is missing required key: '{req_key}'.")

        state = rec.get("state")
        date_str = rec.get("date")
        if state is not None and date_str is not None:
            composite_key = (str(state), str(date_str))
            if composite_key in seen_keys:
                raise ValueError(f"Duplicate record found for state '{state}' and date '{date_str}'.")
            seen_keys.add(composite_key)

import numpy as np
from typing import Any

def index_by_key(records: list[dict]) -> dict[tuple[str, int], dict]:
    """
    Return a dictionary keyed by (state, year). Raise exception on duplicate key.

    Owner: Structures engineer (role 2). Reviewed by: Data steward (role 1).
    """
    indexed: dict[tuple[str, int], dict] = {}
    for rec in records:
        state = rec["state"]
        year = int(str(rec["date"])[:4])
        key = (state, year)
        if key in indexed:
            raise ValueError(f"Duplicate record key detected for: {key}")
        indexed[key] = rec
    return indexed


def join_records(
    income_records: list[dict], 
    poverty_records: list[dict]
) -> tuple[dict[tuple[str, int], dict], set[tuple[str, int]], set[tuple[str, int]]]:
    """
    Matches records on (state, year) and returns joined records,
    income-only keys, and poverty-only keys.

    Owner: Structures engineer (role 2). Reviewed by: Data steward (role 1).
    """
    inc_dict = index_by_key(income_records)
    pov_dict = index_by_key(poverty_records)
    
    inc_keys = set(inc_dict.keys())
    pov_keys = set(pov_dict.keys())
    
    common_keys = inc_keys & pov_keys
    income_only = inc_keys - pov_keys
    poverty_only = pov_keys - inc_keys
    
    joined: dict[tuple[str, int], dict] = {}
    for k in common_keys:
        joined[k] = {**inc_dict[k], **pov_dict[k]}
        
    return joined, income_only, poverty_only


def to_float(value: Any) -> float:
    """
    Return np.nan for None or empty string, and float otherwise.

    Owner: Structures engineer (role 2). Reviewed by: Data steward (role 1).
    """
    if value is None or value == "" or (isinstance(value, float) and np.isnan(value)):
        return float(np.nan)
    return float(value)


def build_matrices(
    income_records: list[dict],
    poverty_records: list[dict],
    states: list[str],
    survey_years: list[int],
    poverty_years: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build MEDIAN (16, 5), MEAN (16, 5), and POVERTY (16, 4) float matrices.

    Owner: Structures engineer (role 2). Reviewed by: Data steward (role 1).
    """
    inc_indexed = index_by_key(income_records)
    pov_indexed = index_by_key(poverty_records)
    
    # Bina matriks menggunakan list comprehension & np.array
    median_list = [
        [to_float(inc_indexed.get((st, yr), {}).get("income_median")) for yr in survey_years]
        for st in states
    ]
    mean_list = [
        [to_float(inc_indexed.get((st, yr), {}).get("income_mean")) for yr in survey_years]
        for st in states
    ]
    poverty_list = [
        [to_float(pov_indexed.get((st, yr), {}).get("poverty_absolute")) for yr in poverty_years]
        for st in states
    ]
    
    return np.array(median_list, dtype=float), np.array(mean_list, dtype=float), np.array(poverty_list, dtype=float)

import numpy as np

def map_states_to_regions(states: list[str], regions_dict: dict[str, list[str]]) -> dict[str, str]:
    """
    Validate that every state belongs to exactly one region and return a state -> region map.

    Owner: Matrix analyst B (role 4). Reviewed by: Matrix analyst A (role 3).
    """
    state_to_region: dict[str, str] = {}
    assigned_count: dict[str, int] = {st: 0 for st in states}
    
    for region, reg_states in regions_dict.items():
        for st in reg_states:
            if st not in assigned_count:
                raise ValueError(f"State '{st}' in REGIONS is not in the valid STATES list.")
            assigned_count[st] += 1
            state_to_region[st] = region
            
    for st, count in assigned_count.items():
        if count == 0:
            raise ValueError(f"State '{st}' is not assigned to any region.")
        elif count > 1:
            raise ValueError(f"State '{st}' is assigned to multiple regions ({count} times).")
            
    return state_to_region


def calculate_relative_income_index(median_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate relative income index using matrix broadcasting across all years.

    Owner: Matrix analyst A (role 3). Reviewed by: Matrix analyst B (role 4).
    """
    # Vector of medians across the 16 states for each year: shape (5,)
    col_medians = np.median(median_matrix, axis=0) 
    # Broadcasting across rows: shape (16, 5) / (5,) -> (16, 5)
    return median_matrix / col_medians


def calculate_regional_means(
    states: list[str],
    median_matrix: np.ndarray,
    regions_dict: dict[str, list[str]]
) -> dict[str, np.ndarray]:
    """
    Calculate unweighted mean of median income for each region across survey years using Boolean masks.

    Owner: Matrix analyst B (role 4). Reviewed by: Matrix analyst A (role 3).
    """
    states_arr = np.array(states)
    regional_means: dict[str, np.ndarray] = {}
    
    for region, reg_states in regions_dict.items():
        # Mask matching region states
        mask = np.isin(states_arr, reg_states)
        regional_means[region] = np.mean(median_matrix[mask, :], axis=0)
        
    return regional_means


def flag_income_poverty_divergence(
    median_matrix: np.ndarray, 
    poverty_matrix: np.ndarray
) -> np.ndarray:
    """
    Return Boolean matrix of shape (16, 3) for periods 2019-2020, 2020-2022, and 2022-2024
    where True indicates median income rose (> 0) but absolute poverty did not fall (>= 0).

    Owner: Matrix analyst B (role 4). Reviewed by: Matrix analyst A (role 3).
    """
    # Sub-slice MEDIAN for years [2019, 2020, 2022, 2024] (Indices 1 to 4)
    med_sub = median_matrix[:, 1:]
    
    med_diff = med_sub[:, 1:] - med_sub[:, :-1]          # shape (16, 3)
    pov_diff = poverty_matrix[:, 1:] - poverty_matrix[:, :-1]  # shape (16, 3)
    
    # Income rose AND poverty did not fall
    return (med_diff > 0) & (pov_diff >= 0)
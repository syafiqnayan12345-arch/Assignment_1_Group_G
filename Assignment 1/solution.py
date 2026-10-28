from typing import Any
import numpy as np

# ==========================================
# FIXED DEFINITIONS & CONSTANTS
# ==========================================

SURVEY_YEARS: list[int] = [2016, 2019, 2020, 2022, 2024]
POVERTY_YEARS: list[int] = [2019, 2020, 2022, 2024]

STATES: list[str] = sorted(
    [
        "Johor",
        "Kedah",
        "Kelantan",
        "Melaka",
        "Negeri Sembilan",
        "Pahang",
        "Perak",
        "Perlis",
        "Pulau Pinang",
        "Sabah",
        "Sarawak",
        "Selangor",
        "Terengganu",
        "W.P. Kuala Lumpur",
        "W.P. Labuan",
        "W.P. Putrajaya",
    ]
)

REGIONS: dict[str, list[str]] = {
    "Northern": ["Perlis", "Kedah", "Pulau Pinang", "Perak"],
    "Central": [
        "Selangor",
        "W.P. Kuala Lumpur",
        "W.P. Putrajaya",
        "Negeri Sembilan",
    ],
    "Southern": ["Melaka", "Johor"],
    "East Coast": ["Kelantan", "Terengganu", "Pahang"],
    "East Malaysia": ["Sabah", "Sarawak", "W.P. Labuan"],
}

def get_group_state(matric_numbers: list[int]) -> list[str]:
    """Calculate the focus state for each group member based on matric numbers.

    Owner: Data steward Athirah
    Reviewed by: Integration and test lead Zulaikha
    """
    if not matric_numbers:
        raise ValueError("Matric numbers list cannot be empty.")

    sorted_matrics = sorted(matric_numbers)
    group_states: list[str] = []

    for m in sorted_matrics:
        rem = m % 16
        state = STATES[rem]
        # Resolve collisions using next available state in circular order
        while state in group_states:
            rem = (rem + 1) % 16
            state = STATES[rem]
        group_states.append(state)
    return group_states 

def validate_records(records, required_keys):

	"""Raise an informative exception if records are not a usable list of keyed records."""
	if not isinstance(records, list):
		raise TypeError("records must be a list")

	if not records:
		raise ValueError("records cannot be empty")

	seen = set()

	for record in records:
		if not isinstance(record, dict):
			raise TypeError("all items must be dictionaries")

		for key in required_keys:
			if key not in record:
				raise KeyError(f"Missing key: {key}")

		pair = (record["state"], record["date"])
		if pair in seen:
			raise ValueError(f"Duplicate (state,date): {pair}")

		seen.add(pair)

# ==========================================
# TASK 2: DATA STRUCTURE CONVERSIONS
# ==========================================


def index_by_key(records: list[dict]) -> dict[tuple[str, int], dict]:
    """Index records by (state, year) tuple and raise an Exception on duplicate keys.

    Owner: Structures engineer Aina
    Reviewed by: Integration and test lead Zulaikha
    """
    indexed: dict[tuple[str, int], dict] = {}
    for rec in records:
        year = int(str(rec["date"])[:4])
        key = (rec["state"], year)
        if key in indexed:
            raise ValueError(f"Duplicate key found in records: {key}")
        indexed[key] = rec
    return indexed


def join_records(income_records: list[dict], poverty_records: list[dict]):
    """Match records on (state, year) and return joined records, income-only keys, and poverty-only keys.

    Owner: Structures engineer Aina
    Reviewed by: Integration and test lead Zulaikha
    """
    inc_indexed = index_by_key(income_records)
    pov_indexed = index_by_key(poverty_records)

    inc_keys = set(inc_indexed.keys())
    pov_keys = set(pov_indexed.keys())

    common_keys = inc_keys.intersection(pov_keys)
    income_only = inc_keys - pov_keys
    poverty_only = pov_keys - inc_keys

    joined: dict[tuple[str, int], dict] = {}
    for key in common_keys:
        joined[key] = {**inc_indexed[key], **pov_indexed[key]}

    return joined, income_only, poverty_only


def to_float(value: Any) -> float:
    """Return np.nan for None and a float otherwise.

    Owner: Structures engineer Aina
    Reviewed by: Integration and test lead Zulaikha
    """
    if value is None:
        return np.nan
    return float(value)


def build_matrices(
    income_records: list[dict], poverty_records: list[dict]
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Construct STATES, MEDIAN (16,5), MEAN (16,5), and POVERTY (16,4) arrays.

    Owner: Structures engineer Aina
    Reviewed by: Integration and test lead Zulaikha
    """
    inc_indexed = index_by_key(income_records)
    pov_indexed = index_by_key(poverty_records)

    states = STATES
    median_mat = np.zeros((len(states), len(SURVEY_YEARS)), dtype=float)
    mean_mat = np.zeros((len(states), len(SURVEY_YEARS)), dtype=float)
    poverty_mat = np.zeros((len(states), len(POVERTY_YEARS)), dtype=float)

    for i, state in enumerate(states):
        for j, year in enumerate(SURVEY_YEARS):
            rec = inc_indexed.get((state, year))
            val_med = rec.get("income_median") if rec else None
            val_mean = rec.get("income_mean") if rec else None
            median_mat[i, j] = to_float(val_med)
            mean_mat[i, j] = to_float(val_mean)

        for k, year in enumerate(POVERTY_YEARS):
            rec = pov_indexed.get((state, year))
            val_pov = rec.get("poverty_absolute") if rec else None
            poverty_mat[i, k] = to_float(val_pov)

    return states, median_mat, mean_mat, poverty_mat


# ==========================================
# TASK 4: REGION MAPPER HELPER
# ==========================================


def validate_and_map_regions(
    regions: dict[str, list[str]], states: list[str]
) -> dict[str, str]:
    """Map states to regions and raise an Exception if any state is missing or assigned multiple times.

    Owner: Matrix analyst B Syafiq
    Reviewed by: Integration and test lead Zulaikha
    """
    state_to_region: dict[str, str] = {}
    for region, state_list in regions.items():
        for state in state_list:
            if state in state_to_region:
                raise ValueError(
                    f"State '{state}' is assigned to multiple regions: '{state_to_region[state]}' and '{region}'"
                )
            state_to_region[state] = region

    for state in states:
        if state not in state_to_region:
            raise ValueError(
                f"State '{state}' is not assigned to any region."
            )

    return state_to_region


# ==========================================
# TASK 5.1: PROFILE BUILDER FUNCTION
# ==========================================


def build_state_profile(
    state: str,
    states: list[str],
    years: list[int],
    median: np.ndarray,
    mean: np.ndarray,
    poverty: np.ndarray,
    poverty_years: list[int],
    regions: dict[str, list[str]],
) -> dict:
    """Return a JSON-serializable profile of one state without using global variables.

    Owner: Matrix analyst B Syafiq
    Reviewed by: Integration and test lead Zulaikha
    """
    if state not in states:
        raise ValueError(f"Unknown state: '{state}'")

    state_idx = states.index(state)

    # Resolve region
    state_region = None
    for reg, st_list in regions.items():
        if state in st_list:
            state_region = reg
            break
    if state_region is None:
        raise ValueError(f"State '{state}' not found in any region mapping.")

    # Income dictionary by year
    income_by_year = {}
    for j, y in enumerate(years):
        income_by_year[str(y)] = {
            "median": float(median[state_idx, j]),
            "mean": float(mean[state_idx, j]),
        }

    # Growth 2016-2024
    med_first = float(median[state_idx, 0])
    med_last = float(median[state_idx, -1])
    growth_pct = float(((med_last - med_first) / med_first) * 100)

    # Relative income index & rank in latest year
    latest_medians = median[:, -1]
    med_of_medians_latest = float(np.median(latest_medians))
    rel_index_latest = float(med_last / med_of_medians_latest)

    # Rank (1 = highest median income)
    sorted_indices = np.argsort(-latest_medians)
    rank = int(np.where(sorted_indices == state_idx)[0][0]) + 1

    # Regional mean median in latest year
    reg_states = regions[state_region]
    reg_mask = np.array([s in reg_states for s in states])
    reg_mean_latest = float(np.mean(median[reg_mask, -1]))
    is_above_region_mean = bool(med_last > reg_mean_latest)

    # Absolute poverty by year
    poverty_by_year = {}
    for k, py in enumerate(poverty_years):
        poverty_by_year[str(py)] = float(poverty[state_idx, k])

    # Flagged periods (Task 4.7): Median rose but poverty did not fall
    periods = ["2019-2020", "2020-2022", "2022-2024"]
    flagged_periods = []
    for p_idx, p_name in enumerate(periods):
        med_rose = median[state_idx, p_idx + 2] > median[state_idx, p_idx + 1]
        pov_not_fell = poverty[state_idx, p_idx + 1] >= poverty[state_idx, p_idx]
        if med_rose and pov_not_fell:
            flagged_periods.append(p_name)

    profile = {
        "state": state,
        "region": state_region,
        "income_by_year": income_by_year,
        "growth_2016_2024_pct": round(growth_pct, 2),
        "relative_income_index_2024": round(rel_index_latest, 4),
        "rank_2024": rank,
        "region_mean_median_2024": round(reg_mean_latest, 2),
        "above_region_mean": is_above_region_mean,
        "poverty_by_year": poverty_by_year,
        "flagged_periods": flagged_periods,
    }

    return profile
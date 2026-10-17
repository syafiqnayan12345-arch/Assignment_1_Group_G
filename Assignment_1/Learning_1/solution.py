STATE : list[str] = [ "Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan", "Pahang",
    "Perak", "Perlis", "Pulau Pinang", "Sabah", "Sarawak", "Selangor",
    "Terengganu", "W.P. Kuala Lumpur", "W.P. Labuan", "W.P. Putrajaya"]

SURVEY_YEARS = [2016, 2019, 2020, 2022, 2024]   # income analysis
POVERTY_YEARS = [2019, 2020, 2022, 2024]         # poverty analysis (see Task 3)

REGIONS = {
    "Northern": ["Perlis", "Kedah", "Pulau Pinang", "Perak"],
    "Central": ["Selangor", "W.P. Kuala Lumpur", "W.P. Putrajaya", "Negeri Sembilan"],
    "Southern": ["Melaka", "Johor"],
    "East Coast": ["Kelantan", "Terengganu", "Pahang"],
    "East Malaysia": ["Sabah", "Sarawak", "W.P. Labuan"],
}


def get_member_state(matrix_num : list[int], state : list[str]):
     ## Owner: Integration and test lead (role 5). Reviewed by: Data steward (role 1).
     ##1.Organize the state alphabeticaly
     sorted_state = sorted(state)

     ##2.Organize matrix number to ascending order
     sorted_matrix = sorted(matrix_num)

     ##3.Get the state for the each matrix number
     group_state : list[str] = []
     for i in sorted_matrix:
            ## Get state index using formula matrix_num modulo 16 state
            index = i % len(state)
            ## If already have same state get the next state that do not have in group_state
            while sorted_state[index] in group_state:
               index = (i+1) % len(state)
            group_state.append(sorted_state[index])

     return group_state

def validate_records(records: list[dict], required_keys: tuple[str, ...]):
    ##Owner: Data steward (role 1). Reviewed by: Structures engineer (role 2).
    if not isinstance(records, list):
         raise TypeError(f"Record should be list, got {type(record).__name__}.")
    if not records:
         raise ValueError("Records is empty")

    process_key = set[tuple[str,str]] = set()

    for index, rec in enumerate(records):
        if not isinstance(rec, dict):
              raise TypeError(f"Record at index {index} is not a dictionary.")
        for req_key in required_keys:
              if req_key not in rec:
                   raise KeyError(f"Record at index {index} is missing required key: '{req_key}'.")
              
        state = rec.get("state")
        date_str = rec.get("date")

        if state is not None and date_str is not None:
             key = (state, date_str)
             if key in process_key():
                  raise ValueError(f"Duplicate record found for state '{state}' and date '{date_str}'.")
             process_key.add(key)
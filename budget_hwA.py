import numpy as np

def compute_budget(total_budget: float, citizen_votes: list[list[float]]) -> list[float]:
    """
    Calculates the budget using the Generalized Median Algorithm with linear rising functions.
    
    Args:
    total_budget: The total amount of money to distribute (C).
    citizen_votes: A list of lists, where each inner list represents a citizen's votes for all subjects.
                Rows are citizens, columns are subjects.
                   
    Returns:
    A list of floats representing the allocated budget for each subject.

    Note: I limited binary search for efficiency. numerical result may vary slightly due to floating 
        point precision. This is fixed by a final scaling step to ensure the total budget matches exactly.  
    """
    
    # Convert to numpy array for easier column handling
    votes = np.array(citizen_votes)
    num_citizens, num_subjects = votes.shape # (rows n, columns m) = (number of citizens, number of subjects)
    
    # Number of phantom voters is n - 1 to ensure pareto efficiency
    num_phantoms = num_citizens - 1
    

    # Helper function to calculate the budget for a *specific t*
    def calculate_proposed_budget(t):
        current_budget = [] # in python [] means we create an empty list

        # For each subject, insert *real* and *phantom* votes to the list of votes on subject_j
        for subject_j in range(num_subjects): 
            # 1. insert *real* votes
            subject_votes = list(votes[:, subject_j])
            
            # 2. insert *phantom* votes
            # Add phantom votes based on t
            # The function: f_k(t) = C * min(1, k * t)
            for k in range(1, num_phantoms + 1):
                phantom_val = total_budget * min(1.0, k * t)
                subject_votes.append(phantom_val) 
            
            # 3. Calculate median of citizen + phantom votes
            median_val = np.median(subject_votes) 
            current_budget.append(median_val) # each element is stored at the end of the dynamic list
        return current_budget

    # Binary search for the optimal t such that sum(budget) == total_budget
    low = 0.0
    high = 1.0  # in order to keep total budget within C
    
    # A small epsilon for floating point comparison
    epsilon = 1e-7
    
    final_budget = []
    
    # Binary search loop
    # using low and high to look each time at narrowed down range
    for _ in range(100): # 100 iterations is plenty for high precision
        mid = (low + high) / 2
        proposed_budget = calculate_proposed_budget(mid) # calculate budget for t=mid
        current_sum = sum(proposed_budget) # total proposed budget sum
        
        if abs(current_sum - total_budget) < epsilon: # if sum close enough to budget cap
            final_budget = proposed_budget # t found!
            break
        elif current_sum < total_budget:
            # Need to increase budget -> increase t (by increasing low and subsequently increasing mid)
            low = mid
        else:
            # Need to decrease budget -> decrease t
            high = mid
            
    # If loop finishes without exact break, use last computed budget
    if not final_budget:
        final_budget = calculate_proposed_budget(mid)
        
    # tiny numeric fix so sum is exactly total_budget (only roundoff-level)
    scaling_factor = total_budget / sum(final_budget)
    final_budget = [x * scaling_factor for x in final_budget]

    return [float(x) for x in final_budget]

# Run example like required
# Each citizen vote must be 100% on one subject in order to promise group fairness
if __name__ == "__main__":
    print(compute_budget(100, [[100, 0, 0], [0, 0, 100]]))

    # must be 100% on one subject
    print(compute_budget(30, [
        [30, 0, 0],
        [0, 30, 0],
        [0, 0, 30]
    ]))

    print(compute_budget(100, [
        [100, 0],
        [100, 0],
        [0, 100]
    ]))
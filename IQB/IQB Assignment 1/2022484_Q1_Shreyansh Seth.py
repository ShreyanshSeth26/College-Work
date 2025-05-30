import numpy as np
import pandas as pd

def perform_global_alignment(seq1, seq2, match_score=2, mismatch_penalty=-3, gap_penalty=-1):
    """
    Performs global alignment of two sequences.
    
    Parameters:
    - seq1: First sequence
    - seq2: Second sequence
    - match_score: Score for character match
    - mismatch_penalty: Penalty for character mismatch
    - gap_penalty: Penalty for introducing a gap
    
    Outputs:
    - Prints the scoring matrix, best score, and all optimal alignments.
    """
    
    # Determine the lengths of both sequences
    len_seq1 = len(seq1)
    len_seq2 = len(seq2)
    
    # Initialize the scoring matrix with zeros. The matrix dimensions will be len_seq1+1 x len_seq2+1.
    scoring_matrix = np.zeros((len_seq1 + 1, len_seq2 + 1))
    
    # Initialize the first column and first row of the scoring matrix with gap penalties.
    for i in range(1, len_seq1 + 1):
        scoring_matrix[i, 0] = gap_penalty * i
    for j in range(1, len_seq2 + 1):
        scoring_matrix[0, j] = gap_penalty * j
    
    # Populate the scoring matrix based on dynamic programming.
    # This loop calculates scores for all cells in the matrix based on matches, mismatches, and gaps.
    for i in range(1, len_seq1 + 1):
        for j in range(1, len_seq2 + 1):
            # Calculate potential scores from three scenarios: diagonal (match/mismatch), up (gap in seq1), left (gap in seq2)
            match_mismatch_score = scoring_matrix[i-1, j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_penalty)
            gap_in_seq1_score = scoring_matrix[i-1, j] + gap_penalty
            gap_in_seq2_score = scoring_matrix[i, j-1] + gap_penalty
            # Assign the max score to the current cell
            scoring_matrix[i, j] = max(match_mismatch_score, gap_in_seq1_score, gap_in_seq2_score)
    
    # Recursive function to trace back through the scoring matrix to find all optimal alignments.
    def find_traceback_paths(i, j, aligned_seq1='', aligned_seq2=''):
        if i == 0 and j == 0:
            # Base case: reached the start of the matrix, return current alignment
            return [(aligned_seq1, aligned_seq2)]
        
        traceback_paths = []
        # Check for paths that could have led to the current cell's score and recursively find alignments for those paths.
        if i > 0 and j > 0 and scoring_matrix[i, j] == scoring_matrix[i-1, j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_penalty):
            traceback_paths += find_traceback_paths(i-1, j-1, seq1[i-1] + aligned_seq1, seq2[j-1] + aligned_seq2)
        if i > 0 and scoring_matrix[i, j] == scoring_matrix[i-1, j] + gap_penalty:
            traceback_paths += find_traceback_paths(i-1, j, seq1[i-1] + aligned_seq1, '-' + aligned_seq2)
        if j > 0 and scoring_matrix[i, j] == scoring_matrix[i, j-1] + gap_penalty:
            traceback_paths += find_traceback_paths(i, j-1, '-' + aligned_seq1, seq2[j-1] + aligned_seq2)
        
        return traceback_paths

    # Retrieve all optimal alignments by starting traceback from the bottom-right corner of the matrix.
    all_optimal_alignments = find_traceback_paths(len_seq1, len_seq2)
    
    # Display the scoring matrix using pandas DataFrame for better visualization.
    print("\nScoring Matrix:")
    print(pd.DataFrame(scoring_matrix, index=['-'] + list(seq1), columns=['-'] + list(seq2)))
    
    # Display the best score, which is found at the bottom-right corner of the scoring matrix.
    optimal_score = scoring_matrix[len_seq1, len_seq2]
    print("\nBest Score:", optimal_score)
    
    # Display all optimal alignments found, numbered for clarity.
    print("\nOptimal Alignments:")
    
    for index, alignment_pair in enumerate(all_optimal_alignments, start=1):
        print(f"Alignment {index}:\n{alignment_pair[0]}\n{alignment_pair[1]}\n")

# Sequences to be aligned
sequence1 = "GATGCGCAG"
sequence2 = "GGCAGTA"

# Execute the global alignment function to find all optimal alignments between the two sequences
perform_global_alignment(sequence1, sequence2)
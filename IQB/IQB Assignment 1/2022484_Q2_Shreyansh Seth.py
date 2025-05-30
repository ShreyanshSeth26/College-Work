import numpy as np
import pandas as pd

def perform_local_sequence_alignment(sequence1, sequence2, match_score=2, mismatch_penalty=-1, gap_penalty=-3):
    """
    Performs local sequence alignment of two sequences.
    
    Parameters:
    - sequence1: First sequence.
    - sequence2: Second sequence.
    - match_score: Score for character match.
    - mismatch_penalty: Penalty for character mismatch.
    - gap_penalty: Penalty for introducing a gap.
    
    Outputs:
    - Prints the scoring matrix, best score, and all optimal local alignments, numbered.
    """
    
    # Initialize the length of sequences and scoring matrix
    sequence_length1 = len(sequence1)
    sequence_length2 = len(sequence2)
    score_matrix = np.zeros((sequence_length1 + 1, sequence_length2 + 1))
    
    highest_score = 0  # Track the highest score found
    highest_score_positions = []  # Track positions of highest scores for traceback
    
    # Fill the scoring matrix
    for row in range(1, sequence_length1 + 1):
        for col in range(1, sequence_length2 + 1):
            # Calculate scores from three scenarios: match/mismatch, gap in sequence1, and gap in sequence2
            score_from_diagonal = score_matrix[row-1, col-1] + (match_score if sequence1[row-1] == sequence2[col-1] else mismatch_penalty)
            score_from_top = score_matrix[row-1, col] + gap_penalty
            score_from_left = score_matrix[row, col-1] + gap_penalty
            # Select the best score (must be non-negative)
            current_score = max(0, score_from_diagonal, score_from_top, score_from_left)
            score_matrix[row, col] = current_score
            
            # Update highest score and positions
            if current_score > highest_score:
                highest_score = current_score
                highest_score_positions = [(row, col)]
            elif current_score == highest_score:
                highest_score_positions.append((row, col))
    
    # Function to traceback from a given position to find all optimal alignments
    def find_alignments_from_score(row, col, aligned_sequence1='', aligned_sequence2=''):
        # Base case: score of 0 indicates the start of an alignment
        if score_matrix[row, col] == 0:
            return [(aligned_sequence1, aligned_sequence2)]
        
        paths = []
        # Traceback paths: diagonal (match/mismatch), up (gap in seq1), left (gap in seq2)
        if row > 0 and col > 0 and score_matrix[row, col] == score_matrix[row-1, col-1] + (match_score if sequence1[row-1] == sequence2[col-1] else mismatch_penalty):
            paths.extend(find_alignments_from_score(row-1, col-1, sequence1[row-1] + aligned_sequence1, sequence2[col-1] + aligned_sequence2))
        if row > 0 and score_matrix[row, col] == score_matrix[row-1, col] + gap_penalty:
            paths.extend(find_alignments_from_score(row-1, col, sequence1[row-1] + aligned_sequence1, '-' + aligned_sequence2))
        if col > 0 and score_matrix[row, col] == score_matrix[row, col-1] + gap_penalty:
            paths.extend(find_alignments_from_score(row, col-1, '-' + aligned_sequence1, sequence2[col-1] + aligned_sequence2))
        
        return paths
    
    # Compile all optimal alignments starting from highest score positions
    all_optimal_alignments = []
    for position in highest_score_positions:
        row, col = position
        all_optimal_alignments.extend(find_alignments_from_score(row, col))
    
    # Remove duplicate alignments
    unique_alignments = list(set(all_optimal_alignments))
    
    # Display results: scoring matrix, highest score, and all unique optimal alignments
    print("\nScoring Matrix:")
    print(pd.DataFrame(score_matrix, index=['-'] + list(sequence1), columns=['-'] + list(sequence2)))
    
    print("\nHighest Score:", highest_score)
    print("\nOptimal Local Alignments:")
    for index, alignment in enumerate(unique_alignments, start=1):
        print(f"Alignment {index}:\n{alignment[0]}\n{alignment[1]}\n")

# Sequences to be aligned
sequence1 = "GCGTAGCCTGC"
sequence2 = "GTGAC"

# Execute the local alignment function
perform_local_sequence_alignment(sequence1, sequence2)
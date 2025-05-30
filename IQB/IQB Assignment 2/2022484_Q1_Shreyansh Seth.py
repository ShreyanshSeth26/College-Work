# Dictionary of propensity values for amino acids to form alpha-helices
pa = {'A': 1.45, 'C': 0.77, 'D': 0.98, 'E': 1.53, 'F': 1.12, 'G': 0.53, 'H': 1.24,
      'I': 1.00, 'K': 1.07, 'L': 1.34, 'M': 1.20, 'N': 0.73, 'P': 0.59, 'Q': 1.17,
      'R': 0.79, 'S': 0.79, 'T': 0.82, 'V': 1.14, 'W': 1.14, 'Y': 0.61}

# Dictionary of propensity values for amino acids to form beta-strands
pb = {'A': 0.97, 'C': 1.30, 'D': 0.80, 'E': 0.26, 'F': 1.28, 'G': 0.81, 'H': 0.71,
      'I': 1.60, 'K': 0.74, 'L': 1.22, 'M': 1.67, 'N': 0.65, 'P': 0.62, 'Q': 1.23,
      'R': 0.90, 'S': 0.72, 'T': 1.20, 'V': 1.65, 'W': 1.19, 'Y': 1.29}

# Assignment amino acid sequence
amino_acid_sequence = "MNASSEGESFAGSVQIPGGTTVLVELTPDIHICGICKQQFNNLDAFVAHKQSGCQLTGTSAAAPSTVQFVSEETVPATQTQTTTRTITSETQTITVSAPEFVFEHGYQTYLPTESNENQTATVISLPAKSRTKKPTTPPAQKRLNCCYPGCQFKTAYGMKDMERHLKIHTGDKPHKCEVCGKCFSRKDKLKTHMRCHTGVKPYKCKTCDYAAADSSSLNKHLRIHSDERPFKCQICPYASRNSSQLTVHLRSHTASELDDDVPKANCLSTESTDTPKAPVITLPSEAREQMATLGERTFNCCYPGCHFKTVHGMKDLDRHLRIHTGDKPHKCEFCDKCFSRKDNLTMHMRCHTSVKPHKCHLCDYAAVDSSSLKKHLRIHSDERPYKCQLCPYASRNSSQLTVHLRSHTGDTPFQCWLCSAKFKISSDLKRHMIVHSGEKPFKCEFCDVRCTMKANLKSHIRIKHTFKCLHCAFQGRDRADLLEHSRLHQADHPEKCPECSYSCSSAAALRVHSRVHCKDRPFKCDFCSFDTKRPSSLAKHVDKVHRDEAKTENRAPLGKEGLREGSSQHVAKIVTQRAFRCETCGASFVRDDSLRCHKKQHSDQSENKNSDLVTFPPESGASGQLSTLVSVGQLEAPLEPSQDL"

# Function to detect regions likely to form helices
def detect_helix_regions(sequence, propensities):
    # Initialize a list to store cumulative propensity scores for helix formation.
    cumulative_helix_score = [0]
    # Initialize a list to count the number of residues favorable for helix formation.
    count_helix_favorable = [0]
    # Iterate over each amino acid in the sequence.
    for aa in sequence:
        # Calculate the running total of helix propensity scores.
        cumulative_helix_score.append(cumulative_helix_score[-1] + propensities[aa])
        # Count how many residues up to this point have a propensity score >= 1, indicating favorable helix formation.
        count_helix_favorable.append(count_helix_favorable[-1] + (1 if propensities[aa] >= 1 else 0))
    # Initialize a list to assign ' ' (space) or 'H' to each residue indicating non-helix or helix.
    helix_assignment = [' '] * len(sequence)
    # Loop through the sequence allowing for a minimum helix length of 6 residues.
    for i in range(len(sequence) - 6 + 1):  # Ensures we have enough room for a 6-residue segment.
        # Determine if the segment from i to i+5 has at least 4 residues favorable for helix formation.
        helix_favorable_count = count_helix_favorable[i + 6] - count_helix_favorable[i]
        if helix_favorable_count >= 4:
            # Assign 'H' to each residue in this segment.
            for j in range(i, i + 6):
                helix_assignment[j] = "H"
            # Extend the helix backward if conditions are met using the expand_backward function.
            expand_backward(i, cumulative_helix_score, helix_assignment)
            # Extend the helix forward from the end of the initial segment using the expand_forward function.
            expand_forward(i + 6, cumulative_helix_score, helix_assignment, len(sequence))
    # Return the list of helix assignments for each residue in the sequence.
    return helix_assignment

# Function to detect regions likely to form beta strands
def detect_beta_regions(sequence, propensities):
    # Initialize lists to store cumulative propensity scores and the count of beta-favorable residues.
    cumulative_beta_score = [0]
    count_beta_favorable = [0]
    # Iterate over each amino acid in the sequence.
    for aa in sequence:
        # Update the cumulative score for beta strand formation propensity.
        cumulative_beta_score.append(cumulative_beta_score[-1] + propensities[aa])
        # Count residues favorable for beta strand formation, with a propensity score >= 1.
        count_beta_favorable.append(count_beta_favorable[-1] + (1 if propensities[aa] >= 1 else 0))
    # Initialize a list to store beta strand assignments (' ' for none, 'S' for beta strand).
    beta_assignment = [' '] * len(sequence)
    # Loop through the sequence to check for potential beta strand segments of minimum length 5 residues.
    for i in range(len(sequence)-5+1):  # Ensure we have enough residues to form a strand.
        # Calculate the count of favorable residues in the current segment of length 5.
        beta_favorable_count = count_beta_favorable[i+5] - count_beta_favorable[i]
        # Check if the segment has at least 3 favorable residues.
        if beta_favorable_count >= 3:
            # Assign 'S' to each residue in this segment indicating it's part of a beta strand.
            for j in range(i, i+5):
                beta_assignment[j] = "S"
            # Optionally extend the beta strand backward to include more residues.
            expand_backward(i, cumulative_beta_score, beta_assignment)
            # Extend the beta strand forward to include additional contiguous favorable residues.
            expand_forward(i+5, cumulative_beta_score, beta_assignment, len(sequence))
    # Return the list of beta strand assignments for each residue.
    return beta_assignment

# Helper functions to expand helix and strand assignments
def expand_backward(start, cumulative_scores, assignment):
    # Initialize the index `k` to one position before the start of the current segment.
    k = start - 1
    # Continue to check backwards as long as `k` remains within the bounds of the sequence (k >= 0).
    while k >= 0:
        # Check if the cumulative score difference between the positions `k+4` and `k` is less than 4.
        # The window of 5 residues (`k+4 - k`) is evaluated to ensure that extending the structure is justified.
        if cumulative_scores[k+4] - cumulative_scores[k] < 4:
            # If the score difference is less than 4, it indicates that extending the structure backward
            # may not be supported by sufficient propensity scores, so break out of the loop.
            break
        # If the condition is not met (i.e., the score difference is 4 or more), assign the structure type
        # (helix or strand, as indicated by the `assignment[start]`) to the current position `k`.
        assignment[k] = assignment[start]
        # Move one residue further backward to continue checking.
        k -= 1
        
# Helper functions to expand helix and strand assignments
def expand_forward(start, cumulative_scores, assignment, seq_length):
    # Initialize the index `k` to the start of the segment where forward extension is considered.
    k = start
    # Loop through the sequence starting from `k` until the end of the sequence (`seq_length`).
    while k < seq_length:
        # Check the cumulative propensity score difference over a window that spans four residues backward.
        # The window `k-3 to k+1` (five residues total) checks if continuing the structure is supported.
        if cumulative_scores[k+1] - cumulative_scores[k-3] < 4:
            # If the score difference is less than 4, it indicates insufficient support to continue
            # extending the structure forward, so break out of the loop.
            break
        # If the condition is not met (i.e., the score difference is 4 or more), assign the structure type
        # (helix or strand, as indicated by the `assignment[start - 1]`) to the current position `k`.
        assignment[k] = assignment[start - 1]
        # Increment `k` to move to the next residue forward in the sequence.
        k += 1

# Function to combine helix and beta strand assignments while resolving conflicts
def combine_structures(sequence, helix, beta, pa, pb):
    # Initialize the final structure list to store the resolved secondary structure.
    final_structure = []
    # Initialize conflict points to track positions where both helix and beta strand predictions overlap.
    conflict_points = [" "] * len(sequence)
    # Iterate over each residue in the sequence.
    i = 0
    while i < len(sequence):
        # Check for conflicts where a residue is predicted to be both a helix and a beta strand.
        if helix[i] == "H" and beta[i] == "S":
            # Mark the starting point of the conflict.
            decision_point = i + 1
            # Continue to mark conflict points as long as the conflict persists.
            while decision_point < len(sequence) and helix[decision_point] == "H" and beta[decision_point] == "S":
                conflict_points[decision_point] = "C"
                decision_point += 1
            conflict_points[i] = "C"
            # Calculate the total propensity scores for forming a helix and a beta strand over the conflicted region.
            helix_score = sum(pa[sequence[j]] for j in range(i, decision_point))
            beta_score = sum(pb[sequence[j]] for j in range(i, decision_point))
            # Resolve the conflict by selecting the structure type (helix or strand) with the higher score.
            structure_type = "H" if helix_score > beta_score else "S"
            # Extend the chosen structure type over the length of the conflicted region.
            final_structure.extend([structure_type] * (decision_point - i))
            # Move the index to the end of the conflicted region.
            i = decision_point - 1
        # Assign 'H' for helix regions without conflict.
        elif helix[i] == "H":
            final_structure.append("H")
        # Assign 'S' for beta strand regions without conflict.
        elif beta[i] == "S":
            final_structure.append("S")
        # Assign '_' for regions not predicted to be either helix or beta strand.
        else:
            final_structure.append("_")
        # Move to the next residue.
        i += 1
    # Return the final secondary structure and conflict points as strings.
    return "".join(final_structure), "".join(conflict_points).replace(" ", "_")

# Detect regions for helices and beta strands
helix_regions = detect_helix_regions(amino_acid_sequence, pa)
beta_regions = detect_beta_regions(amino_acid_sequence, pb)
# Combine structures and resolve conflicts
final_secondary_structure, conflict_resolution = combine_structures(amino_acid_sequence, helix_regions, beta_regions, pa, pb)

# Print the header and the protein sequence under investigation.
print("IQB ASSIGNMENT 2\n\n" + "For Sequence:\n" + amino_acid_sequence)
# Print the locations within the sequence where alpha-helices are predicted.
print("\nA.) Helix Nucleation Sites:\n" + "".join(helix_regions).replace(" ", "_"))
# Print the locations within the sequence where beta-strands are predicted.
print("\nB.) Beta Nucleation Sites:\n" + "".join(beta_regions).replace(" ", "_"))
# Print the regions of the sequence where there were conflicts between helix and strand predictions.
print("\nC.) Conflicting Regions:\n" + conflict_resolution)
# Print the final secondary structure assignment for the sequence after resolving all conflicts.
print("\nD.) Final Secondary Structure:\n" + final_secondary_structure)
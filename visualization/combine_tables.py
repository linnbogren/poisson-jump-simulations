"""
Combine LaTeX tables from different optimization runs for comparison.

This module allows you to merge tables from two different experiment runs
(e.g., one optimized for BAC, another for composite score) into a single
comparison table with labeled model rows.
"""

import re
from pathlib import Path
from typing import Optional, List, Tuple


def extract_table_data(tex_content: str) -> Tuple[List[str], List[List[str]], str, str]:
    """
    Extract structured data from a LaTeX table.
    
    Parameters
    ----------
    tex_content : str
        Content of the LaTeX table file
        
    Returns
    -------
    tuple
        (header_row, data_rows, caption, label)
        - header_row: List of column headers
        - data_rows: List of [model_name, *values] for each row
        - caption: Table caption
        - label: Table label
    """
    # Extract caption
    caption_match = re.search(r'\\caption\{([^}]+)\}', tex_content)
    caption = caption_match.group(1) if caption_match else ""
    
    # Extract label
    label_match = re.search(r'\\label\{([^}]+)\}', tex_content)
    label = label_match.group(1) if label_match else ""
    
    # Extract header row (after \toprule, before first \midrule)
    header_match = re.search(r'\\toprule\s*\n\s*(.+?)\s*\n\s*\\midrule', tex_content, re.DOTALL)
    if header_match:
        header_line = header_match.group(1).strip()
        header_row = [col.strip() for col in header_line.split('&')]
    else:
        header_row = []
    
    # Extract data rows
    data_rows = []
    
    # Find all sections between \multicolumn lines
    sections = re.findall(
        r'\\multicolumn\{[0-9]+\}\{l\}\{\$\\mathbf\{([^}]+)\}\$\}\s*\\\\\s*\n(.*?)(?=\\midrule|\\bottomrule)',
        tex_content,
        re.DOTALL
    )
    
    for section_name, section_content in sections:
        # Extract model rows from this section
        lines = section_content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('%'):
                continue
            
            # Remove trailing \\ and split by &
            line = line.replace('\\\\', '').strip()
            if not line:
                continue
            
            # Remove any \textbf{} formatting
            line = re.sub(r'\\textbf\{([^}]+)\}', r'\1', line)
            
            parts = [p.strip() for p in line.split('&')]
            if len(parts) > 1:  # Must have model name + at least one value
                data_rows.append(parts)
    
    return header_row, data_rows, caption, label


def combine_tables(
    table1_path: Path,
    table2_path: Path,
    output_path: Path,
    label1: str = "1",
    label2: str = "2",
    new_caption: Optional[str] = None,
    new_label: Optional[str] = None
):
    """
    Combine two LaTeX tables into a single comparison table.
    
    Parameters
    ----------
    table1_path : Path
        Path to first table file
    table2_path : Path
        Path to second table file
    output_path : Path
        Path where combined table will be saved
    label1 : str, default="1"
        Label to append to model names from first table (e.g., "BAC")
    label2 : str, default="2"
        Label to append to model names from second table (e.g., "Comp")
    new_caption : str, optional
        Caption for combined table. If None, uses first table's caption with " - Comparison"
    new_label : str, optional
        Label for combined table. If None, uses first table's label with "_comparison"
    
    Examples
    --------
    >>> combine_tables(
    ...     Path("results/1dfc0517/tables/02_balanced_accuracy_by_delta_n_total_features.tex"),
    ...     Path("results/4d42444f/tables/02_balanced_accuracy_by_delta_n_total_features.tex"),
    ...     Path("combined_tables/balanced_accuracy_comparison.tex"),
    ...     label1="BAC",
    ...     label2="Comp"
    ... )
    """
    # Read both tables
    with open(table1_path, 'r', encoding='utf-8') as f:
        table1_content = f.read()
    
    with open(table2_path, 'r', encoding='utf-8') as f:
        table2_content = f.read()
    
    # Extract data from both tables
    header1, data1, caption1, label1_orig = extract_table_data(table1_content)
    header2, data2, caption2, label2_orig = extract_table_data(table2_content)
    
    # Verify headers match
    if header1 != header2:
        print(f"Warning: Headers don't match exactly")
        print(f"Table 1: {header1}")
        print(f"Table 2: {header2}")
    
    # Use first table's header
    header = header1
    
    # Group data by delta sections
    # Assume data is organized by delta sections with Gaussian, Poisson, PoissonKL in each
    combined_rows = []
    
    # Process in groups of 3 (assuming 3 models per delta)
    models = ['Gaussian', 'Poisson', 'PoissonKL']
    i = 0
    
    while i < len(data1):
        # For each set of 3 models from table1
        for model_idx in range(3):
            if i + model_idx < len(data1):
                row1 = data1[i + model_idx]
                row2 = data2[i + model_idx] if i + model_idx < len(data2) else None
                
                model_name = row1[0]
                
                # Add row from table 1 with label
                combined_rows.append([f"{model_name} ({label1})"] + row1[1:])
                
                # Add row from table 2 with label
                if row2:
                    combined_rows.append([f"{model_name} ({label2})"] + row2[1:])
        
        i += 3
    
    # Create new caption and label
    if new_caption is None:
        new_caption = caption1.replace("Mean (Std)", f"Mean (Std) - {label1} vs {label2}")
    
    if new_label is None:
        new_label = label1_orig + "_comparison"
    
    # Build combined table
    n_cols = len(header)
    
    # Start building the LaTeX table
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{{new_caption}}}",
        f"  \\label{{{new_label}}}",
        f"  \\begin{{tabular}}{{{'l' + 'c' * (n_cols - 1)}}}",
        r"    \toprule",
        "    " + " & ".join(header) + r" \\",
        r"    \midrule",
    ]
    
    # Add data rows grouped by delta
    current_delta = None
    row_idx = 0
    
    # Infer delta values from data (assumes consistent ordering)
    # We'll add section headers every 6 rows (3 models × 2 tables)
    delta_values = []
    
    # Try to infer delta values from original table
    delta_matches = re.findall(r'\\mathbf\{(delta = [0-9.]+)\}', table1_content)
    delta_values = delta_matches if delta_matches else []
    
    delta_idx = 0
    for i in range(0, len(combined_rows), 6):  # 6 rows per delta (3 models × 2 tables)
        # Add delta section header
        if delta_idx < len(delta_values):
            lines.append(f"    \\multicolumn{{{n_cols}}}{{l}}{{$\\mathbf{{{delta_values[delta_idx]}}}$}} \\\\")
            delta_idx += 1
        
        # Add 6 rows (Gaussian_1, Gaussian_2, Poisson_1, Poisson_2, PoissonKL_1, PoissonKL_2)
        for j in range(6):
            if i + j < len(combined_rows):
                row = combined_rows[i + j]
                lines.append("    " + " & ".join(row) + r" \\")
        
        # Add midrule after each delta section (except the last)
        if i + 6 < len(combined_rows):
            lines.append(r"    \midrule")
    
    # Close table
    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"  \begin{tablenotes}",
        r"    \small",
        f"    \\item Note: Comparison of results optimized for different metrics. ({label1}) shows results from first run, ({label2}) shows results from second run.",
        r"  \end{tablenotes}",
        r"\end{table}",
    ])
    
    # Write combined table
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"✓ Combined table saved to: {output_path}")
    return output_path


def combine_tables_by_hash(
    hash1: str,
    hash2: str,
    table_filename: str,
    output_filename: str,
    label1: str = "Run1",
    label2: str = "Run2",
    results_base_dir: Path = Path("results")
):
    """
    Combine tables from two experiment runs identified by their hash.
    
    Parameters
    ----------
    hash1 : str
        Hash identifier of first experiment (e.g., "1dfc0517")
    hash2 : str
        Hash identifier of second experiment (e.g., "4d42444f")
    table_filename : str
        Name of the table file to combine (e.g., "02_balanced_accuracy_by_delta_n_total_features.tex")
    output_filename : str
        Name for the output combined table
    label1 : str, default="Run1"
        Label for first run (e.g., "BAC")
    label2 : str, default="Run2"
        Label for second run (e.g., "Comp")
    results_base_dir : Path, default=Path("results")
        Base directory containing result folders
        
    Examples
    --------
    >>> combine_tables_by_hash(
    ...     "1dfc0517",
    ...     "4d42444f",
    ...     "02_balanced_accuracy_by_delta_n_total_features.tex",
    ...     "balanced_accuracy_BAC_vs_Comp.tex",
    ...     label1="BAC",
    ...     label2="Comp"
    ... )
    """
    # Find directories matching the hashes
    results_base_dir = Path(results_base_dir)
    
    # Look for directories containing the hash
    dirs1 = list(results_base_dir.glob(f"*{hash1}*"))
    dirs2 = list(results_base_dir.glob(f"*{hash2}*"))
    
    if not dirs1:
        raise FileNotFoundError(f"No results directory found for hash: {hash1}")
    if not dirs2:
        raise FileNotFoundError(f"No results directory found for hash: {hash2}")
    
    # Use the first match
    dir1 = dirs1[0]
    dir2 = dirs2[0]
    
    print(f"Using directories:")
    print(f"  Run 1 ({label1}): {dir1}")
    print(f"  Run 2 ({label2}): {dir2}")
    
    # Construct table paths
    table1_path = dir1 / "tables" / table_filename
    table2_path = dir2 / "tables" / table_filename
    
    if not table1_path.exists():
        raise FileNotFoundError(f"Table not found: {table1_path}")
    if not table2_path.exists():
        raise FileNotFoundError(f"Table not found: {table2_path}")
    
    # Create output directory
    output_dir = results_base_dir / "combined_tables"
    output_path = output_dir / output_filename
    
    # Combine tables
    return combine_tables(
        table1_path,
        table2_path,
        output_path,
        label1=label1,
        label2=label2
    )


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) >= 6:
        # Command line usage
        hash1 = sys.argv[1]
        hash2 = sys.argv[2]
        table_filename = sys.argv[3]
        label1 = sys.argv[4]
        label2 = sys.argv[5]
        output_filename = sys.argv[6] if len(sys.argv) > 6 else f"combined_{table_filename}"
        
        combine_tables_by_hash(
            hash1, hash2, table_filename, output_filename,
            label1=label1, label2=label2
        )
    else:
        # Interactive example
        print("Example: Combining balanced accuracy tables from BAC vs Comp optimization")
        print()
        
        combine_tables_by_hash(
            "1dfc0517",
            "4d42444f",
            "02_balanced_accuracy_by_delta_n_total_features.tex",
            "balanced_accuracy_BAC_vs_Comp.tex",
            label1="BAC",
            label2="Comp"
        )

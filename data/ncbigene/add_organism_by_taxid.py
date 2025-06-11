"""
Add organism data to NCBI gene reference files.

This script adds data for organisms by tax_id to the NCBI gene reference files.
It creates:
1. Individual REF_NAMES2NCBIGENE files for each new organism (for database search)
2. Individual REF_NCBIGENE2NAMES files for each new organism (for RAG embeddings)
3. Updates the global REF_NCBIGENE2LABEL file

The organism-specific approach is necessary because the same gene name can refer 
to different gene IDs in different organisms.
"""

import pandas as pd
import compress_pickle
import os
from pathlib import Path

def add_organism_data(tax_ids, 
                     gene_info_file=None, 
                     output_dir=None,
                     output_suffix="_updated", 
                     include_types=['protein-coding'],
                     data_dir=None):
    """
    Add data for organisms by tax_id to the NCBI gene reference files.
    
    This creates:
    1. Individual REF_NAMES2NCBIGENE files for each new organism
    2. Individual REF_NCBIGENE2NAMES files for each new organism (for RAG embeddings)
    3. updates the global REF_NCBIGENE2LABEL file
    
    Args:
        tax_ids (str or list): Tax_id or list of tax_ids to add (strings or integers)
        gene_info_file (str): Path to NCBI gene info file 
        output_dir (str): Directory to save updated reference files
        output_suffix (str): Suffix to add to updated files
        include_types (list): Gene types to include (default: ['protein-coding'])
        data_dir (str): Directory containing existing data files
        
    Returns:
        dict: Information about created files
    """
    
    # Set default paths
    if data_dir is None:
        current_dir = Path(os.getcwd())
        if 'ncbigene' in str(current_dir):
            data_dir = current_dir.parent.parent.parent / 'Data' 
        else:
            data_dir = Path('/Users/luna/Desktop/CRBM/AMAS_proj/Data/')
    
    if output_dir is None:
        output_dir = Path(os.getcwd())
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    
    # Ensure tax_ids are strings
    if isinstance(tax_ids, str):
        tax_ids = [tax_ids]
    tax_ids = [str(tid) for tid in tax_ids]
    print(f"Adding organisms with tax_ids: {tax_ids}")
    
    # Load gene info file
    if gene_info_file is None:
        gene_info_file = data_dir / "ncbi" / "All_Data.gene_info.gz"
    
    print(f"Loading gene info from: {gene_info_file}")
    columns_to_keep = ["#tax_id", "GeneID", "Symbol", "Synonyms", "type_of_gene"]
    df = pd.read_csv(gene_info_file, compression='gzip', sep='\t', usecols=columns_to_keep)
    
    # Filter by gene type and tax_ids
    df = df[df['type_of_gene'].isin(include_types)]
    df['#tax_id'] = df['#tax_id'].astype(str)
    organism_dfs = {}
    
    for tax_id in tax_ids:
        organism_df = df[df['#tax_id'] == tax_id]
        if len(organism_df) == 0:
            print(f"Warning: No genes found for tax_id {tax_id}")
            continue
        organism_dfs[tax_id] = organism_df
        print(f"Found {len(organism_df)} genes for tax_id {tax_id}")
    
    if not organism_dfs:
        print("No organisms found with the specified tax_ids")
        return None
    
    # 1. Create organism-specific files for each tax_id
    created_files = []
    all_new_genes = {}  # For updating global label file
    
    for tax_id, organism_df in organism_dfs.items():
        # Create names to gene ID mapping (for database search)
        names_to_geneid = {}
        # Create gene ID to names mapping (for RAG embeddings)
        geneid_to_names = {}
        
        for idx, row in organism_df.iterrows():
            gene_id = str(row['GeneID'])
            symbol = str(row['Symbol']).strip() if pd.notnull(row['Symbol']) else ''
            synonyms_str = str(row['Synonyms']).strip() if pd.notnull(row['Synonyms']) else ''
            
            # Collect all names for this gene
            names = []
            if symbol and symbol != '-' and symbol != 'nan':
                names.append(symbol)
            if synonyms_str and synonyms_str != '-' and synonyms_str != 'nan':
                # Split by '|' and strip whitespace
                names.extend([s.strip() for s in synonyms_str.split('|') 
                            if s.strip() and s.strip() != symbol and s.strip() != '-'])
            
            # Remove duplicates while preserving order
            seen = set()
            unique_names = []
            for name in names:
                if name and name not in seen:
                    unique_names.append(name)
                    seen.add(name)
            
            # Store gene ID to names mapping (for RAG embeddings)
            geneid_to_names[gene_id] = unique_names
            
            # Add each name to the reverse mapping (for database search)
            for name in unique_names:
                norm_name = name.lower()
                if norm_name not in names_to_geneid:
                    names_to_geneid[norm_name] = []
                if gene_id not in names_to_geneid[norm_name]:
                    names_to_geneid[norm_name].append(gene_id)
            
            # Store for global label file update
            all_new_genes[gene_id] = symbol if symbol and symbol != '-' and symbol != 'nan' else f"gene_{gene_id}"
        
        # Save organism-specific NAMES2NCBIGENE file (for database search)
        output_names2gene = output_dir / f"names2ncbigene_tax{tax_id}_{'_'.join(include_types)}.lzma"
        with open(output_names2gene, 'wb') as handle:
            compress_pickle.dump(names_to_geneid, handle, compression="lzma", set_default_extension=False)
        created_files.append(str(output_names2gene))
        
        # Save organism-specific NCBIGENE2NAMES file (for RAG embeddings)
        output_gene2names = output_dir / f"ncbigene2names_tax{tax_id}_{'_'.join(include_types)}.lzma"
        with open(output_gene2names, 'wb') as handle:
            compress_pickle.dump(geneid_to_names, handle, compression="lzma", set_default_extension=False)
        created_files.append(str(output_gene2names))
        
        print(f"Created files for tax_id {tax_id}: {len(organism_df)} genes, {len(names_to_geneid)} unique names")
        print(f"  - {output_names2gene.name} (for search)")
        print(f"  - {output_gene2names.name} (for RAG embeddings)")
    
    # 2. update global REF_NCBIGENE2LABEL file
    updated_label_file = None
    # Load existing label file if it exists
    existing_label_file = output_dir / f"ncbigene2label_bigg_organisms_{'_'.join(include_types)}.lzma"
    
    if existing_label_file.exists():
        print(f"Loading existing label file: {existing_label_file}")
        with open(existing_label_file, 'rb') as handle:
            existing_labels = compress_pickle.load(handle)
        print(f"Existing labels: {len(existing_labels)} genes")
        
        # Add new genes
        existing_labels.update(all_new_genes)
        print(f"After adding new organisms: {len(existing_labels)} genes")
        
        # Save updated file
        updated_label_file = output_dir / f"ncbigene2label_bigg_organisms_{'_'.join(include_types)}{output_suffix}.lzma"
        with open(updated_label_file, 'wb') as handle:
            compress_pickle.dump(existing_labels, handle, compression="lzma", set_default_extension=False)
        created_files.append(str(updated_label_file))
        print(f"Updated global label file: {updated_label_file.name}")
    else:
        print(f"Creating new global label file")
        updated_label_file = output_dir / f"ncbigene2label_bigg_organisms_{'_'.join(include_types)}{output_suffix}.lzma"
        with open(updated_label_file, 'wb') as handle:
            compress_pickle.dump(all_new_genes, handle, compression="lzma", set_default_extension=False)
        created_files.append(str(updated_label_file))
        print(f"Created global label file: {updated_label_file.name}")
    
    # Summary
    total_genes = sum(len(df) for df in organism_dfs.values())
    total_files = len(created_files)
    
    print(f"\n=== Summary ===")
    print(f"Added {len(tax_ids)} organisms")
    print(f"Total genes: {total_genes}")
    print(f"Created/updated {total_files} files:")
    for f in created_files:
        print(f)
    
    return {
        'tax_ids': tax_ids,
        'total_genes': total_genes,
        'created_files': created_files,
        'updated_label_file': str(updated_label_file) if updated_label_file else None
    }


def get_organism_info(tax_ids, gene_info_file=None, data_dir=None):
    """
    Get information about organisms from gene_info file.
    
    Args:
        tax_ids (list): List of tax_ids to get info for
        gene_info_file (str): Path to NCBI gene info file
        data_dir (str): Directory containing data files
        
    Returns:
        DataFrame with organism information
    """
    
    # Set default paths
    if data_dir is None:
        current_dir = Path(os.getcwd())
        if 'ncbigene' in str(current_dir):
            data_dir = current_dir.parent.parent.parent / 'Data' 
        else:
            data_dir = Path('/Users/luna/Desktop/CRBM/AMAS_proj/Data/')
    
    data_dir = Path(data_dir)
    
    if gene_info_file is None:
        gene_info_file = data_dir / "ncbi" / "All_Data.gene_info.gz"
    
    # Ensure tax_ids are strings
    tax_ids = [str(tid) for tid in tax_ids]
    
    print(f"Loading gene info from: {gene_info_file}")
    columns_to_keep = ["#tax_id", "GeneID", "Symbol", "type_of_gene"]
    df = pd.read_csv(gene_info_file, compression='gzip', sep='\t', usecols=columns_to_keep)
    
    # Filter by tax_ids
    df['#tax_id'] = df['#tax_id'].astype(str)
    organism_df = df[df['#tax_id'].isin(tax_ids)]
    
    # Get summary stats by organism and gene type
    summary = organism_df.groupby(['#tax_id', 'type_of_gene']).size().reset_index(name='gene_count')
    summary_pivot = summary.pivot(index='#tax_id', columns='type_of_gene', values='gene_count').fillna(0)
    
    # Add total column
    summary_pivot['total'] = summary_pivot.sum(axis=1)
    
    print(f"\nOrganism gene counts:")
    print(summary_pivot)
    
    return summary_pivot


def load_organism_names_dict(tax_id, data_dir=None, include_types=['protein-coding']):
    """
    Load organism-specific names dictionary.
    
    Args:
        tax_id (str): Tax ID of organism
        data_dir (str): Directory containing reference files 
        include_types (list): Gene types included in files
        
    Returns:
        dict: Names to gene ID mapping for the organism
    """
    
    if data_dir is None:
        data_dir = Path(os.getcwd())
    
    data_dir = Path(data_dir)
    
    # Load names2gene file
    names_file = data_dir / f"names2ncbigene_tax{tax_id}_{'_'.join(include_types)}.lzma"
    
    if not names_file.exists():
        raise FileNotFoundError(f"Names file not found: {names_file}")
    
    with open(names_file, 'rb') as handle:
        names_dict = compress_pickle.load(handle)
    
    return names_dict


def load_organism_gene2names_dict(tax_id, data_dir=None, include_types=['protein-coding']):
    """
    Load organism-specific gene ID to names dictionary (for RAG embeddings).
    
    Args:
        tax_id (str): Tax ID of organism
        data_dir (str): Directory containing reference files 
        include_types (list): Gene types included in files
        
    Returns:
        dict: Gene ID to names mapping for the organism
    """
    
    if data_dir is None:
        data_dir = Path(os.getcwd())
    
    data_dir = Path(data_dir)
    
    # Load gene2names file
    gene2names_file = data_dir / f"ncbigene2names_tax{tax_id}_{'_'.join(include_types)}.lzma"
    
    if not gene2names_file.exists():
        raise FileNotFoundError(f"Gene2names file not found: {gene2names_file}")
    
    with open(gene2names_file, 'rb') as handle:
        gene2names_dict = compress_pickle.load(handle)
    
    return gene2names_dict


def list_available_organisms(data_dir=None, include_types=['protein-coding']):
    """
    List available organism-specific reference files.
    
    Args:
        data_dir (str): Directory containing reference files
        include_types (list): Gene types to look for
        
    Returns:
        list: List of available tax_ids
    """
    
    if data_dir is None:
        data_dir = Path(os.getcwd())
    
    data_dir = Path(data_dir)
    
    # Look for organism-specific files
    pattern = f"names2ncbigene_tax*_{'_'.join(include_types)}.lzma"
    files = list(data_dir.glob(pattern))
    
    tax_ids = []
    for f in files:
        # Extract tax_id from filename: names2ncbigene_tax{tax_id}_protein_coding.lzma
        parts = f.stem.split('_')
        if len(parts) >= 2 and parts[1].startswith('tax'):
            tax_id = parts[1][3:]  # Remove 'tax' prefix
            tax_ids.append(tax_id)
    
    tax_ids.sort()
    print(f"Available organisms: {len(tax_ids)}")
    for tax_id in tax_ids:
        names_file = data_dir / f"names2ncbigene_tax{tax_id}_{'_'.join(include_types)}.lzma"
        gene2names_file = data_dir / f"ncbigene2names_tax{tax_id}_{'_'.join(include_types)}.lzma"
        has_both = names_file.exists() and gene2names_file.exists()
        print(f"  {tax_id}: {'✓' if has_both else '✗'} (both search and RAG files)")
    
    return tax_ids


if __name__ == "__main__":
    # Example usage
    print("Add organism data to NCBI gene reference files")
    print("Available functions:")
    print("- add_organism_data(tax_ids, ...)")
    print("- get_organism_info(tax_ids, ...)")  
    print("- load_organism_names_dict(tax_id, ...)")
    print("- load_organism_gene2names_dict(tax_id, ...)")
    print("- list_available_organisms(...)")
    
    # Example: add E. coli data
    # result = add_organism_data(['511145'])
    # print(result)

"""
Create organism-specific NCBI gene reference files.

This script creates:
1. A single REF_NCBIGENE2LABEL file for all BiGG organisms
2. Separate REF_NAMES2NCBIGENE files for each organism by tax_id
3. Separate REF_NCBIGENE2NAMES files for each organism by tax_id (needed for RAG embeddings)

The organism-specific approach is necessary because the same gene name can refer 
to different gene IDs in different organisms.
"""

import pandas as pd
import compress_pickle
import os
from pathlib import Path

def create_organism_specific_references(
    data_dir=None,
    output_dir=None,
    gene_info_file=None,
    bigg_organisms_file=None,
    include_types=['protein-coding']
):
    """
    Create organism-specific NCBI gene reference files.
    
    Args:
        data_dir: Directory containing NCBI data files
        output_dir: Directory to save reference files
        gene_info_file: Path to NCBI gene_info file (default: use All_Data.gene_info.gz)
        bigg_organisms_file: Path to BiGG organisms CSV file
        include_types: List of gene types to include (default: ['protein-coding'])
    
    Returns:
        dict: Summary statistics about created files
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
    
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load gene info file
    if gene_info_file is None:
        gene_info_file = data_dir / "ncbi" / "All_Data.gene_info.gz"
    
    print(f"Loading gene info from: {gene_info_file}")
    columns_to_keep = ["#tax_id", "GeneID", "Symbol", "Synonyms", "type_of_gene"]
    df = pd.read_csv(gene_info_file, compression='gzip', sep='\t', usecols=columns_to_keep)
    
    print(f"Loaded {len(df)} gene records")
    print(f"Gene types: {df['type_of_gene'].value_counts().head()}")
    
    # Filter by gene type
    df = df[df['type_of_gene'].isin(include_types)]
    print(f"After filtering by gene type {include_types}: {len(df)} records")
    
    # Load BiGG organisms
    if bigg_organisms_file is None:
        bigg_organisms_file = data_dir.parent / "Models" / "BiggModels" / "bigg_organisms.csv"
    
    if not os.path.exists(bigg_organisms_file):
        print(f"Warning: BiGG organisms file not found at {bigg_organisms_file}")
        print("Using all organisms in the gene info file instead")
        taxids = df['#tax_id'].unique().tolist()
    else:
        print(f"Loading BiGG organisms from: {bigg_organisms_file}")
        bigg_organisms = pd.read_csv(bigg_organisms_file)
        taxids = bigg_organisms['tax_id'].unique().tolist()
        taxids = [str(x) for x in taxids if x and str(x).strip()]  # Remove empty strings and convert to string
    
    print(f"Target tax IDs: {len(taxids)} organisms")
    
    # Filter by BiGG organisms
    df['#tax_id'] = df['#tax_id'].astype(str)
    bigg_df = df[df['#tax_id'].isin(taxids)]
    bigg_df = bigg_df.reset_index(drop=True)
    
    print(f"After filtering by BiGG organisms: {len(bigg_df)} records")
    print(f"Organisms found: {sorted(bigg_df['#tax_id'].unique())}")
    
    # 1. Create single REF_NCBIGENE2LABEL for all BiGG organisms
    output_gene2label = f"ncbigene2label_bigg_organisms_{'_'.join(include_types)}.lzma"
    gene_id_to_symbol = dict(zip(bigg_df['GeneID'], bigg_df['Symbol']))
    
    with open(output_dir / output_gene2label, 'wb') as handle:
        compress_pickle.dump(gene_id_to_symbol, handle, compression="lzma", set_default_extension=False)
    
    print(f"Created REF_NCBIGENE2LABEL: {output_gene2label} ({len(gene_id_to_symbol)} entries)")
    
    # 2. Create organism-specific reference files
    organism_stats = []
    
    for tax_id in sorted(bigg_df['#tax_id'].unique()):
        organism_df = bigg_df[bigg_df['#tax_id'] == tax_id]
        
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
        
        # Save organism-specific NAMES2NCBIGENE file (for database search)
        output_names2gene = f"names2ncbigene_tax{tax_id}_{'_'.join(include_types)}.lzma"
        with open(output_dir / output_names2gene, 'wb') as handle:
            compress_pickle.dump(names_to_geneid, handle, compression="lzma", set_default_extension=False)
        
        # Save organism-specific NCBIGENE2NAMES file (for RAG embeddings)
        output_gene2names = f"ncbigene2names_tax{tax_id}_{'_'.join(include_types)}.lzma"
        with open(output_dir / output_gene2names, 'wb') as handle:
            compress_pickle.dump(geneid_to_names, handle, compression="lzma", set_default_extension=False)
        
        organism_stats.append({
            'tax_id': tax_id,
            'genes': len(organism_df),
            'unique_names': len(names_to_geneid),
            'names2gene_file': output_names2gene,
            'gene2names_file': output_gene2names
        })
        
        print(f"Created files for tax_id {tax_id}: {len(organism_df)} genes, {len(names_to_geneid)} unique names")
        print(f"  - {output_names2gene} (for search)")
        print(f"  - {output_gene2names} (for RAG embeddings)")
    
    # Summary statistics
    total_files = len(organism_stats) * 2 + 1  # 2 files per organism + 1 global label file
    total_genes = sum(stat['genes'] for stat in organism_stats)
    total_names = sum(stat['unique_names'] for stat in organism_stats)
    
    summary = {
        'total_files_created': total_files,
        'label_file': output_gene2label,
        'organism_files': len(organism_stats) * 2,  # 2 files per organism
        'total_genes': total_genes,
        'total_unique_names': total_names,
        'organism_stats': organism_stats
    }
    
    print(f"\n=== Summary ===")
    print(f"Created {total_files} reference files:")
    print(f"  - 1 REF_NCBIGENE2LABEL file: {output_gene2label}")
    print(f"  - {len(organism_stats)} organism-specific REF_NAMES2NCBIGENE files (for search)")
    print(f"  - {len(organism_stats)} organism-specific REF_NCBIGENE2NAMES files (for RAG)")
    print(f"Total genes: {total_genes}")
    print(f"Total unique names: {total_names}")
    print(f"Average names per organism: {total_names / len(organism_stats):.1f}")
    
    return summary


def test_organism_specific_files(output_dir=None, tax_id=None):
    """Test loading and using one of the organism-specific files."""
    
    if output_dir is None:
        output_dir = Path(os.getcwd())
    
    output_dir = Path(output_dir)
    
    # Find organism-specific files
    names2gene_files = list(output_dir.glob("names2ncbigene_tax*_protein-coding.lzma"))
    gene2names_files = list(output_dir.glob("ncbigene2names_tax*_protein-coding.lzma"))
    
    if not names2gene_files or not gene2names_files:
        print("No organism-specific files found")
        return
    
    # Use specified tax_id or first available
    if tax_id:
        test_names2gene_file = output_dir / f"names2ncbigene_tax{tax_id}_protein-coding.lzma"
        test_gene2names_file = output_dir / f"ncbigene2names_tax{tax_id}_protein-coding.lzma"
        if not test_names2gene_file.exists() or not test_gene2names_file.exists():
            print(f"Files for tax_id {tax_id} not found")
            return
    else:
        test_names2gene_file = names2gene_files[0]
        test_gene2names_file = gene2names_files[0]
        tax_id = test_names2gene_file.name.split('_')[1][3:]  # Extract tax_id from filename
    
    print(f"\nTesting organism-specific files for tax_id: {tax_id}")
    print(f"Files: {test_names2gene_file.name}, {test_gene2names_file.name}")
    
    # Load both files
    names2gene_dict = compress_pickle.load(open(test_names2gene_file, 'rb'))
    gene2names_dict = compress_pickle.load(open(test_gene2names_file, 'rb'))
    
    print(f"Loaded {len(names2gene_dict)} name->gene mappings")
    print(f"Loaded {len(gene2names_dict)} gene->names mappings")
    
    print("\nSample name->gene mappings:")
    for i, (name, gene_ids) in enumerate(list(names2gene_dict.items())[:5]):
        print(f"  '{name}' -> {gene_ids}")
        if i >= 4:
            break
    
    print("\nSample gene->names mappings:")
    for i, (gene_id, names) in enumerate(list(gene2names_dict.items())[:5]):
        print(f"  '{gene_id}' -> {names}")
        if i >= 4:
            break
    
    # Test lookup functionality
    print(f"\nTesting search functionality:")
    test_names = ['cox1', 'atpb', 'rpob', 'rpoa']
    for name in test_names:
        if name.lower() in names2gene_dict:
            gene_ids = names2gene_dict[name.lower()]
            print(f"  '{name}' -> {gene_ids}")
        else:
            print(f"  '{name}' -> Not found")


if __name__ == "__main__":
    print("Creating organism-specific NCBI gene reference files...")
    
    # Create the files
    summary = create_organism_specific_references()
    
    # Test one of the files
    if summary['organism_stats']:
        test_tax_id = summary['organism_stats'][0]['tax_id']
        test_organism_specific_files(tax_id=test_tax_id)
    
    print("\nDone!") 
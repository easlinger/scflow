from pybiomart import Server


def convert_gene_species(genes,  from_species="mouse", to_species="human"):
    """
    Convert gene symbols from human to mouse or vice-versa.

    Parameters
    ----------
    genes: list or str
        List of gene symbols to convert.
    to_species: str
        Species to convert to, either 'mouse' or 'human' or another
        species as seen in `pybiomart' dataset prefixes to
        '_gene_ensembl' (e.g., 'dmelanogaster'). To see available
        species, use
        pybiomart.Server(host="http://www.ensembl.org").marts[
            "ENSEMBL_MART_ENSEMBL"].datasets.keys().
    from_species: str
        Species to convert from, either 'mouse' or 'human' or another
        species abbreviation/key as seen in `pybiomart` datasets.

    Returns
    -------
    pd.Series
        Series containing corresponding mouse gene symbols.
    """
    if isinstance(genes, str):
        genes = [genes]
    to_species = "mmusculus" if to_species.lower() in [
        "mouse", "mice"] else "hsapiens" if to_species.lower() in [
            "human", "humans"] else to_species
    from_species = "mmusculus" if from_species.lower() in [
        "mouse", "mice"] else "hsapiens" if from_species.lower() in [
            "human", "humans"] else from_species
    server = Server(host="http://www.ensembl.org")
    mart = server.marts["ENSEMBL_MART_ENSEMBL"]
    human_dataset = mart.datasets[f"{to_species}_gene_ensembl"]
    orthologs = human_dataset.query(attributes=[
        "external_gene_name", f"{from_species}_homolog_associated_gene_name"])
    orthologs = orthologs.dropna().set_index(orthologs.columns[0])
    orthologs = orthologs.loc[orthologs.index.intersection(genes)].iloc[:, 0]
    return orthologs

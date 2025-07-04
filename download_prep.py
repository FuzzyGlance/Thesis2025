import requests
from sentence_transformers import SentenceTransformer
import faiss  
import pickle
from owlready2 import get_ontology
import pandas as pd
import os
import re

## LOCAL NCBI TAXON ONTOLOGY DOWNLOAD
def load_ncbi_ontology():
    if not os.path.exists("ncbitaxon.owl"):
        print("Downloading NCBI taxonomy...")
        url = "http://purl.obolibrary.org/obo/ncbitaxon.owl"
        response = requests.get(url)

    with open("ncbitaxon.owl", "xb") as f:
        f.write(response.content)

    print("loading in NCBI taxonomy...")
    ncbi_ontology = get_ontology("ncbitaxon.owl").load()
    return ncbi_ontology

## CREATING/LOADING TAXON NAMES
if not os.path.exists("taxon_names.pkl"):
    ncbi_ontology = load_ncbi_ontology()

    print("Creating taxon names...")
    taxon_names = [t.label[0] for t in ncbi_ontology.search(label="*")]  # Replace with your ontology loader

    with open("taxon_names.pkl", "xb") as f:
        pickle.dump(taxon_names, f)  # Creates the .pkl file 
else:
    with open("taxon_names.pkl", "rb") as f:
        taxon_names = pickle.load(f) 
        print("loaded in NCBI taxon names...")



## CREATING/LOADING TAXON NAMES/DATA (label, rank, iri)
if not os.path.exists("taxon_data_r.pkl"):
    ncbi_ontology = load_ncbi_ontology()
    
    print("Downloading taxon data...")
    taxon_data = []
    for taxon in ncbi_ontology.search(label="*"):
        label = taxon.label[0] if taxon.label else None
        iri = taxon.iri.split("/")[-1] if taxon.iri else None
        
        # Extract rank
        rank = None
        if hasattr(taxon, "has_rank") and taxon.has_rank:  # Check if list is non-empty
            rank = str(taxon.has_rank[0]).split("/")[-1]  # Extract "species" from full IRI
            rank = rank.split("_")[1]
        taxon_data.append((label, rank, iri))

        with open("taxon_data_r.pkl", "xb") as f:
            pickle.dump(taxon_data, f)
else:
    with open("taxon_data_r.pkl", "rb") as f:
        taxon_data = pickle.load(f)
        print("loaded in NCBI taxonomy data...")


## CREATING FAISS INDEX FOR TAXON NAMES
encoders = [
    "menadsa/BioS-MiniLM",
    "all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
    "pritamdeka/S-BioBert-snli-multinli-stsb",
    "intfloat/e5-small-v2",
    "intfloat/e5-large-v2",
    "intfloat/multilingual-e5-large",
    "juanpablomesa/all-mpnet-base-v2-bioasq-matryoshka",
    "NeuML/pubmedbert-base-embeddings"
    ]
index_names = [re.sub(r"\s*[-/.]\s*", "", e.split("/")[-1].lower()) for e in encoders]

for i in range (len(index_names)): 
    file_name = f"ncbi_faiss_{index_names[i]}.index"
    if not os.path.exists(file_name):
        print("Creating index for", encoders[i])
        encoder = SentenceTransformer(encoders[i])
        embeddings = encoder.encode(taxon_names, show_progress_bar=True) # Create embeddings for the taxon names
        faiss.normalize_L2(embeddings) # Normalize vectors 

        # FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, file_name) # Save index
    else:
        print("Index already exists for", encoders[i])
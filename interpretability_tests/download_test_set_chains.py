import random
import urllib.request
import json

#Load the test chains dictionary - you may have to edit the file path
with open("interpretability_tests\\test_chains.json") as test_chains_file:
    test_chains = json.load(test_chains_file)

base_url = "https://files.rcsb.org/download/"

#Download each PDB file - you can edit the output filepath if needed
for pdb_id in test_chains.keys():
    urllib.request.urlretrieve(f"{base_url}{pdb_id}.pdb", f"{pdb_id}.pdb")
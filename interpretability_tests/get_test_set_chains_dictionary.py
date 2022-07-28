import json
import requests

#Load the test set
with open("interpretability_tests\splits.json") as splits_json:
    splits = json.load(splits_json)

test_set_chains = splits["test"]

core_url = "https://data.rcsb.org/rest/v1/core/polymer_entity_instance/"

required_chains = {}

#Iterate through each chain
for chain in test_set_chains:
    entry_id = chain[:4].upper()
    asym_id = chain[-1].upper()

    r = requests.get(f"{core_url}{entry_id}/{asym_id}")

    data = r.json()

    try:
        length = data["rcsb_polymer_entity_instance_container_identifiers"]["auth_to_entity_poly_seq_mapping"][-1]
    except KeyError:
        continue
    
    #If the chain length is less than 100, add to a dictionary
    if int(length) <= 100:
        required_chains[entry_id] = asym_id
        print(f"{entry_id}_{asym_id}")

#Write the dictionary to a json file - keys are PDB IDs and values are chain IDs
with open("test_chains.json", "w") as outfile:
    json.dump(required_chains, outfile)
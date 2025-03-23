from ruamel.yaml import YAML

yaml = YAML()



yaml_data = """
learning_rate: str
"""

parsed = yaml.load(yaml_data)
print(parsed["learning_rate"], type(parsed["learning_rate"]))

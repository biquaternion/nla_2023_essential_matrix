import yaml

class Config(yaml.YAMLObject):
    yaml_tag = '!Config'

    def __init__(self, intrinsic: dict, cameras: dict):
        self.intrinsic = intrinsic
        self.cameras = cameras

    def __repr__(self):
        return f'{self.__class__.__name__}({self.intrinsic}, {self.cameras})'

def load_config(filename: str) -> Config:
    with open(filename) as file:
        return yaml.load(file)
